"""
用户问："模型是事先预测、事后决定？能不能直接把预测当结果卖了？"

本脚本把决策彻底改成**在线/因果**，不允许回头看整条路径：
  D0 p>0.5 首次触发即卖（最"直接"的用法，无额外参数）
  D1 p 在线追踪止盈：记录 p 的历史最高 pmax，p 从 pmax 回撤 x 就卖（x=5%/10%/20%/30%）
  D2 p 创新高后一根即卖（x→0 的极限，纯"预测到峰就走"）
  D3 p 连续 k 根 >0.5 才卖（去抖）
  D4 与 θ=1% 谁先触发用谁
每个决策只用 j 及之前的信息，SHUF 模型同样跑一遍做对照。
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_online_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
CTX, TOPK, CONF_DROP = 96, 6, 0.01

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H + CTX:
        return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    lr1 = np.zeros(n); lr1[1:] = np.log(c[1:] / c[:-1])
    sd24 = pd.Series(lr1).rolling(24, min_periods=24).std().to_numpy()
    return dict(h=h, l=l, c=c, v=v, t=t, n=n, volr=volr, ret24=ret24, sd24=sd24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None:
        continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))

Xl, yl, ei_l = [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c, v = P['h'], P['l'], P['c'], P['v']
    prevc = np.concatenate([[c[0]], c[:-1]])
    lr1 = np.log(c / prevc)
    for j in range(i, i + H):
        f = [float(c[j] / c[j - k] - 1.0) for k in (12, 24, 48, 96)]
        v12 = float(v[j - 11:j + 1].mean()); v96 = float(v[j - 95:j + 1].mean())
        f.append(float(v12 / v96) if v96 > 0 else 1.0)
        f.append(float(np.std(lr1[j - 23:j + 1])))
        hi96 = float(h[j - 95:j + 1].max()); lo96 = float(l[j - 95:j + 1].min())
        f.append(float((c[j] - lo96) / (hi96 - lo96)) if hi96 > lo96 else 0.5)
        rng = float(h[j] - l[j])
        f.append(float((h[j] - c[j]) / rng) if rng > 0 else 0.0)
        f.append(float(c[j] / h[j - 12] - 1.0))
        is_top = h[j] >= h[max(0, j - TOPK):min(P['n'] - 1, j + TOPK) + 1].max() - 1e-12
        fut = l[j + 1:j + 13]
        ok = len(fut) > 0 and fut.min() <= h[j] * (1.0 - CONF_DROP)
        Xl.append(f); yl.append(1 if (is_top and ok) else 0); ei_l.append(ei)

X = np.asarray(Xl, np.float32); y = np.asarray(yl, np.int8); ei_l = np.asarray(ei_l)
q_of = {e: str(qm_all[e]) for e in range(len(entries))}
qr = np.array([q_of[e] for e in ei_l])
tr = qr < str(qs[VALID_FROM_Q]); va = ~tr
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
p_r = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr]).predict_proba(Xs[va])[:, 1]
rng = np.random.default_rng(11)
p_s = LogisticRegression(max_iter=1000).fit(Xs[tr], rng.permutation(y[tr])).predict_proba(Xs[va])[:, 1]
print('训练完成 %.1fs' % (time.time() - t0), flush=True)

va_e = np.unique(ei_l[va])
pos = {e: np.where(ei_l[va] == e)[0] for e in va_e}
retp = {}
for e in va_e:
    P = paths[entries[e][1]]; i = entries[e][2]; c = P['c']
    retp[e] = c[i:i + H] / c[i] - 1.0
qv = np.array([q_of[e] for e in va_e])


def report(name, nets):
    g = pd.DataFrame({'q': qv, 'net': nets}).groupby('q').net.mean()
    return '  %-30s %9.3f%% %9.3f%% %6d/%d' % (name, nets.mean() * 100, np.median(nets) * 100,
                                               int((g > 0).sum()), len(g))


def d_first(P, th=0.5):
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; k = len(ps) - 1
        idx = np.where(ps > th)[0]
        if len(idx):
            k = int(idx[0]); trig += 1
        out.append(float(retp[e][k]) - COST)
    return np.array(out), trig / len(va_e)


def d_trail_p(P, x=0.1, floor=0.1):
    """p 的在线追踪止盈：pmax 是历史最高 p，p 回撤 x 即卖（全因果）"""
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; pmax = 0.0; k = len(ps) - 1
        for j in range(len(ps)):
            if ps[j] > pmax:
                pmax = ps[j]
            elif pmax > floor and ps[j] < pmax * (1.0 - x):
                k = j; trig += 1; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out), trig / len(va_e)


def d_newhigh(P):
    """p 创新高后下一根即卖：纯"预测到峰就走" """
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; pmax = -1.0; k = len(ps) - 1
        for j in range(len(ps)):
            if ps[j] > pmax:
                pmax = ps[j]
                if j + 1 < len(ps):
                    k = j + 1; trig += 1; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out), trig / len(va_e)


def d_debounce(P, th=0.5, k=3):
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; s = len(ps) - 1
        run = 0
        for j in range(len(ps)):
            run = run + 1 if ps[j] > th else 0
            if run >= k:
                s = j; trig += 1; break
        out.append(float(retp[e][s]) - COST)
    return np.array(out), trig / len(va_e)


def theta_k(th=0.01):
    ks = []
    for e in va_e:
        P = paths[entries[e][1]]; i = entries[e][2]
        h, l, c = P['h'], P['l'], P['c']
        eh = h[i:i + H] / c[i] - 1.0
        el = l[i:i + H] / c[i] - 1.0
        hi = 0.0; s = H - 1
        for j in range(H):
            if eh[j] > hi:
                hi = eh[j]
            if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
                s = j; break
        ks.append(s)
    return np.array(ks)


def d_whichever(P, x=0.1, floor=0.1):
    kth = theta_k(0.01)
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; pmax = 0.0; k = int(kth[n])
        for j in range(len(ps)):
            if ps[j] > pmax:
                pmax = ps[j]
            elif pmax > floor and ps[j] < pmax * (1.0 - x) and j < k:
                k = j; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out), (kth < H - 1).mean()


L = []
L.append('把预测直接当卖出信号（全在线/因果，无事后挑选）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集 %d 笔 / %d 季 | 成本 0.04%% 单边' % (len(va_e), len(set(qv))))
L.append('')
L.append('  %-30s %10s %10s %8s %8s' % ('策略', '均值/笔', '中位/笔', '季正', '触发率'))
L.append(report('固定持满 96 根（基线）', np.array([retp[e][-1] - COST for e in va_e])) + '       -')
L.append(report('θ=1% 跟踪止盈（V20）', np.array([float(retp[e][k]) - COST for e, k in zip(va_e, theta_k(0.01))]))
         + '   %5.1f%%' % ((theta_k(0.01) < H - 1).mean() * 100))
for th in (0.3, 0.5):
    n_, r_ = d_first(p_r, th)
    L.append(report('REAL p>%.1f 首次触发即卖' % th, n_) + '   %5.1f%%' % (r_ * 100))
    n_, r_ = d_first(p_s, th)
    L.append(report('SHUF p>%.1f 首次触发即卖' % th, n_) + '   %5.1f%%' % (r_ * 100))
n_, r_ = d_newhigh(p_r)
L.append(report('REAL p创新高后一根即卖', n_) + '   %5.1f%%' % (r_ * 100))
for x in (0.05, 0.1, 0.2, 0.3):
    n_, r_ = d_trail_p(p_r, x)
    L.append(report('REAL p追踪止盈 x=%.0f%%' % (x * 100), n_) + '   %5.1f%%' % (r_ * 100))
n_, r_ = d_debounce(p_r, 0.5, 3)
L.append(report('REAL p连续3根>0.5 卖', n_) + '   %5.1f%%' % (r_ * 100))
n_, r_ = d_whichever(p_r, 0.1)
L.append(report('REAL p追踪x=10% 与 θ谁先', n_) + '   -')
L.append('')
L.append('读法：以上全部只用 j 及之前的信息；"p创新高后一根即卖"是最接近"预测到峰就走"的实现。')
L.append('诚实边界：样本重叠、零滑点、成本单边 0.04% 未核实往返；标签含未来 12 根（正确）。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
