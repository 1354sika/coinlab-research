"""
用户要求：把模型的"警觉"（不是绝对概率，而是相对自身基线的抬升）直接当决策，看结果。

之前的错法：用绝对阈值 p>0.3/0.5 —— p 平时只有 0.056，峰处 0.095，绝对阈值没有意义。
本次：只看 z 分数
    z_t = (p_t - 过去 W 根 p 的均值) / 过去 W 根 p 的标准差
    z 抬升 = 模型"突然比刚才警觉了"，这与绝对水平无关。
决策（全部在线，只用 j 及之前）：
  A1 z > 阈值 即卖（th = 0.2 / 0.3 / 0.5 / 1.0）
  A2 z 触及阈值后回落（z 从最高点回撤 50%）即卖
  A3 z > 阈值 且 p > 0.06（要求有一点绝对水平）
  A4 z > 阈值 与 θ=1% 谁先
每个都用 REAL 与 SHUF 双跑。
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/alert_decision_REAL.txt'
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
n_e = len(entries)
q_of = {e: str(qm_all[e]) for e in range(n_e)}
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


def report(name, nets, trig=None):
    g = pd.DataFrame({'q': qv, 'net': nets}).groupby('q').net.mean()
    s = '  %-32s %9.3f%% %9.3f%% %6d/%d' % (name, nets.mean() * 100, np.median(nets) * 100,
                                            int((g > 0).sum()), len(g))
    if trig is not None:
        s += ' %7.1f%%' % (trig * 100)
    return s


def zscores(ps, win=24):
    z = np.zeros(len(ps))
    for j in range(win, len(ps)):
        m = ps[j - win:j].mean(); s = ps[j - win:j].std() + 1e-6
        z[j] = (ps[j] - m) / s
    return z


def d_z(P, th=0.3, win=24, floor_p=0.0):
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; z = zscores(ps, win); k = len(ps) - 1
        for j in range(win, len(ps)):
            if z[j] > th and ps[j] > floor_p:
                k = j; trig += 1; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out), trig / len(va_e)


def d_z_trail(P, th=0.3, win=24, back=0.5):
    """z 触及 th 后，z 从最高点回落 back 比例即卖"""
    out = []; trig = 0
    for e in va_e:
        ps = P[pos[e]]; z = zscores(ps, win); k = len(ps) - 1
        zmax = 0.0; armed = False
        for j in range(win, len(ps)):
            if z[j] > zmax:
                zmax = z[j]
            if z[j] > th:
                armed = True
            elif armed and z[j] < zmax * (1 - back):
                k = j; trig += 1; break
        out.append(float(retp[e][k]) - COST)
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


def d_z_or_theta(P, th=0.3, win=24):
    kth = theta_k(0.01); out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zscores(ps, win); k = int(kth[n])
        for j in range(win, min(len(ps), k)):
            if z[j] > th:
                k = j; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out), (np.array([0]))


L = []
L.append('把"警觉"（z 分数）直接当决策  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集 %d 笔 / %d 季 | 成本 0.04%% 单边 | z 用过去 24 根滚动均值与标准差' % (len(va_e), len(set(qv))))
L.append('')
L.append('  %-32s %10s %10s %8s %8s' % ('策略', '均值/笔', '中位/笔', '季正', '触发率'))
L.append(report('固定持满 96 根（基线）', np.array([retp[e][-1] - COST for e in va_e])))
L.append(report('θ=1% 跟踪止盈（V20）', np.array([float(retp[e][k]) - COST for e, k in zip(va_e, theta_k(0.01))]),
                (theta_k(0.01) < H - 1).mean()))
for th in (0.2, 0.3, 0.5, 1.0):
    n_, r_ = d_z(p_r, th); L.append(report('REAL z>%.1f 即卖' % th, n_, r_))
    n_, r_ = d_z(p_s, th); L.append(report('SHUF z>%.1f 即卖' % th, n_, r_))
n_, r_ = d_z_trail(p_r, 0.3, 24, 0.5); L.append(report('REAL z>0.3 后回落50%卖', n_, r_))
n_, r_ = d_z(p_r, 0.3, 24, 0.06); L.append(report('REAL z>0.3 且 p>0.06 卖', n_, r_))
n_, _ = d_z_or_theta(p_r, 0.3); L.append(report('REAL z>0.3 与 θ=1% 谁先', n_))
L.append('')
L.append('读法：z 衡量"比刚才警觉了多少"，不是"绝对有多确信"。')
L.append('      REAL 明显优于 SHUF 且为正，才说明"警觉"本身含可交易信息。')
L.append('诚实边界：样本重叠、零滑点、成本 0.04% 单边未核实往返。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
