"""
"喂法"检验 v2：认峰信号能不能变成钱（修正 v1 的决策缺陷）

v1 的缺陷：阈值 p>0.5 触发率只有 4.5%，等于"几乎不交易、持到窗口末"，结果自然接近基线 -0.195%。
本版换决策方式，直接看"认峰"的最大可用价值：
  D1 argmax(p)：每笔在模型最确信"这是峰"的那一根卖（若 p>th）
  D2 分位法：每笔取 p 最高 10% 里的第一根卖
  D3 认峰 + θ=1% 兜底：若最确信那根早于 θ 触发根，就用它；否则按 θ=1%
  D4 oracle 对照：事后最优出场（上界）
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_label2_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
CTX, TOPK, CONF_DROP = 96, 6, 0.01

from sklearn.linear_model import LogisticRegression

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('load %.1fs' % (time.time() - t0), flush=True)


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
print('入场 %d  %.1fs' % (len(entries), time.time() - t0), flush=True)

Xl, yl, ei_l = [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c, v = P['h'], P['l'], P['c'], P['v']
    prevc = np.concatenate([[c[0]], c[:-1]])
    lr1 = np.log(c / prevc)
    for j in range(i, i + H):
        if j - CTX < 0:
            continue
        f = [float(c[j] / c[j - k] - 1.0) for k in (12, 24, 48, 96)]
        v12 = float(v[j - 11:j + 1].mean()); v96 = float(v[j - 95:j + 1].mean())
        f.append(float(v12 / v96) if v96 > 0 else 1.0)
        f.append(float(np.std(lr1[j - 23:j + 1])))
        hi96 = float(h[j - 95:j + 1].max()); lo96 = float(l[j - 95:j + 1].min())
        f.append(float((c[j] - lo96) / (hi96 - lo96)) if hi96 > lo96 else 0.5)
        rng = float(h[j] - l[j])
        f.append(float((h[j] - c[j]) / rng) if rng > 0 else 0.0)
        f.append(float(c[j] / h[j - 12] - 1.0))
        lo_ = max(0, j - TOPK); hi_ = min(P['n'] - 1, j + TOPK)
        is_top = h[j] >= h[lo_:hi_ + 1].max() - 1e-12
        fut = l[j + 1:j + 13]
        ok = len(fut) > 0 and fut.min() <= h[j] * (1.0 - CONF_DROP)
        Xl.append(f); yl.append(1 if (is_top and ok) else 0); ei_l.append(ei)

X = np.asarray(Xl, np.float32); y = np.asarray(yl, np.int8); ei_l = np.asarray(ei_l)
q_of = {e: str(qm_all[e]) for e in range(len(entries))}
qr = np.array([q_of[e] for e in ei_l])
tr = qr < str(qs[VALID_FROM_Q]); va = ~tr
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
m = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr])
p = m.predict_proba(Xs[va])[:, 1]
print('训练完成 AUC 见下  %.1fs' % (time.time() - t0), flush=True)


def auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b]); r = pd.Series(allv).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


yv = y[va]
A = auc(p[yv == 1], p[yv == 0])

va_e = np.unique(ei_l[va])
pos = {e: np.where(ei_l[va] == e)[0] for e in va_e}
retp = {}
for e in va_e:
    P = paths[entries[e][1]]; i = entries[e][2]; c = P['c']
    retp[e] = c[i:i + H] / c[i] - 1.0
qv = np.array([q_of[e] for e in va_e])


def report(name, nets):
    g = pd.DataFrame({'q': qv, 'net': nets}).groupby('q').net.mean()
    return '  %-26s %9.3f%% %9.3f%% %6d/%d' % (name, nets.mean() * 100, np.median(nets) * 100,
                                               int((g > 0).sum()), len(g))


def d_argmax(th):
    nets = []
    for e in va_e:
        ps = p[pos[e]]
        k = int(np.argmax(ps))
        nets.append(float(retp[e][k]) - COST if ps[k] > th else float(retp[e][-1]) - COST)
    return np.array(nets)


def d_topq():
    nets = []
    for e in va_e:
        ps = p[pos[e]]
        k = int(np.argmax(ps))
        nets.append(float(retp[e][k]) - COST if ps[k] >= np.percentile(ps, 90) else float(retp[e][-1]) - COST)
    return np.array(nets)


def theta_sell(th=0.01):
    """θ 跟踪止盈（同 V20 口径）"""
    out = []
    for e in va_e:
        P = paths[entries[e][1]]; i = entries[e][2]
        h, l, c = P['h'], P['l'], P['c']
        gp = c[i:i + H] / c[i] - 1.0
        eh = h[i:i + H] / c[i] - 1.0
        el = l[i:i + H] / c[i] - 1.0
        hi = 0.0; s = H - 1
        for j in range(H):
            if eh[j] > hi:
                hi = eh[j]
            if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
                s = j; break
        out.append((float(gp[s]) - COST, s))
    return np.array([x[0] for x in out]), np.array([x[1] for x in out])


def d_peak_plus_theta(th_p=0.7):
    base, k0 = theta_sell(0.01)
    nets = []
    for n, (e, kbase) in enumerate(zip(va_e, k0)):
        ps = p[pos[e]]
        k = int(np.argmax(ps))
        if ps[k] > th_p and k < kbase:
            nets.append(float(retp[e][k]) - COST)
        else:
            nets.append(base[n])
    return np.array(nets)


def oracle():
    out = []
    for e in va_e:
        P = paths[entries[e][1]]; i = entries[e][2]
        out.append(float(P['h'][i:i + H].max() / P['c'][i] - 1.0) - COST)
    return np.array(out)


L = []
L.append('"喂法"检验 v2：认峰信号能否变成钱  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('同 v1 的标签/特征/划分；换决策方式。验证集 %d 笔 / %d 季' % (len(va_e), len(set(qv))))
L.append('验证集 AUC = %.3f' % A)
L.append('')
L.append('  %-26s %10s %10s %8s' % ('策略', '均值/笔', '中位/笔', '季正'))
L.append(report('固定持满 96 根（基线）', np.array([retp[e][-1] - COST for e in va_e])))
L.append(report('θ=1% 跟踪止盈（V20）', theta_sell(0.01)[0]))
for th in (0.3, 0.5, 0.7):
    L.append(report('argmax(p) 卖, p>%.1f' % th, d_argmax(th)))
L.append(report('p 最高10% 分位卖', d_topq()))
for th in (0.5, 0.7):
    L.append(report('认峰(p>%.1f) + θ=1%%兜底' % th, d_peak_plus_theta(th)))
L.append(report('oracle 上界（事后最优）', oracle()))
L.append('')
L.append('参考：V20 原始结果 +0.053%/笔 7/9；A2（每季重训创新高概率）+0.253%/笔 7/9。')
L.append('诚实边界：同一批入场点、样本重叠、零滑点零成交率；标签含未来 12 根信息（正确）；')
L.append('  特征只用 j 及之前（无前视）。认峰 AUC 高只说明"可识别"，不等于"可获利"。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
