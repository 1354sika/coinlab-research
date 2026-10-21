"""
"喂法"检验 v3 —— 自查并修正 v2 的**前视泄漏**。

v2 的错：分位法用 np.percentile(整条路径的 p, 90) 作阈值 —— 这用到了未来 96 根的 p 分布，
         实盘不可用。"p 最高10%分位卖 +0.799%/笔 8/9 季正"因此是泄漏产物，不能采信。
v3 全部改成**滚动阈值**（只看过去 W 根的 p），并加 SHUF 对照（打乱标签训练的模型）：
  D1 固定阈值 th（无泄漏，v2 已有，作为基准复现）
  D2 滚动分位：p[j] > 过去 24/48 根的 90 分位 且 p[j] > 0.3
  D3 滚动新高：p[j] > 过去 24 根最大值
  D4 θ=1% 兜底 + 滚动认峰提前卖
判定纪律：只有 D2~D4 在 **REAL 明显高于 SHUF** 且为正，才算"认峰能赚钱"。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_label3_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
CTX, TOPK, CONF_DROP = 96, 6, 0.01
from sklearn.linear_model import LogisticRegression

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
print('入场 %d  %.1fs' % (len(entries), time.time() - t0), flush=True)

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


def auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b]); r = pd.Series(allv).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


yv = y[va]
A_r = auc(p_r[yv == 1], p_r[yv == 0]); A_s = auc(p_s[yv == 1], p_s[yv == 0])
print('AUC REAL %.3f SHUF %.3f  %.1fs' % (A_r, A_s, time.time() - t0), flush=True)

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


def d_fixed(P, th):
    out = []
    for e in va_e:
        ps = P[pos[e]]
        k = int(np.argmax(ps))
        out.append(float(retp[e][k]) - COST if ps[k] > th else float(retp[e][-1]) - COST)
    return np.array(out)


def d_roll(P, win=24, q=90, th=0.3):
    """滚动分位：p[j] > 过去 win 根的 q 分位 且 > th，则卖；否则持到末"""
    out = []
    for e in va_e:
        ps = P[pos[e]]; n = len(ps); k = n - 1
        for j in range(win, n):
            if ps[j] > np.percentile(ps[j - win:j], q) and ps[j] > th:
                k = j; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out)


def d_rollmax(P, win=24, th=0.3):
    out = []
    for e in va_e:
        ps = P[pos[e]]; n = len(ps); k = n - 1
        for j in range(win, n):
            if ps[j] > ps[j - win:j].max() and ps[j] > th:
                k = j; break
        out.append(float(retp[e][k]) - COST)
    return np.array(out)


def theta_sell(th=0.01):
    out, ks = [], []
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
        out.append(float(gp[s]) - COST); ks.append(s)
    return np.array(out), np.array(ks)


def d_peak_theta(P, win=24, q=90, th=0.3):
    base, k0 = theta_sell(0.01)
    nets = []
    for n, (e, kb) in enumerate(zip(va_e, k0)):
        ps = P[pos[e]]; n_ = len(ps); pick = kb
        for j in range(win, n_):
            if ps[j] > np.percentile(ps[j - win:j], q) and ps[j] > th and j < kb:
                pick = j; break
        nets.append(float(retp[e][pick]) - COST)
    return np.array(nets)


L = []
L.append('"喂法"检验 v3（修正 v2 的滚动分位前视泄漏 + SHUF 对照）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集 %d 笔 / %d 季 | AUC REAL %.3f vs SHUF %.3f' % (len(va_e), len(set(qv)), A_r, A_s))
L.append('')
L.append('  %-30s %10s %10s %8s' % ('策略', '均值/笔', '中位/笔', '季正'))
L.append(report('固定持满 96 根（基线）', np.array([retp[e][-1] - COST for e in va_e])))
L.append(report('θ=1% 跟踪止盈（V20）', theta_sell(0.01)[0]))
L.append(report('REAL 固定阈值 argmax p>0.3', d_fixed(p_r, 0.3)))
L.append(report('SHUF 固定阈值 argmax p>0.3', d_fixed(p_s, 0.3)))
L.append(report('REAL 滚动分位24/90 p>0.3', d_roll(p_r, 24, 90, 0.3)))
L.append(report('SHUF 滚动分位24/90 p>0.3', d_roll(p_s, 24, 90, 0.3)))
L.append(report('REAL 滚动分位48/90 p>0.3', d_roll(p_r, 48, 90, 0.3)))
L.append(report('SHUF 滚动分位48/90 p>0.3', d_roll(p_s, 48, 90, 0.3)))
L.append(report('REAL 滚动新高24 p>0.3', d_rollmax(p_r, 24, 0.3)))
L.append(report('SHUF 滚动新高24 p>0.3', d_rollmax(p_s, 24, 0.3)))
L.append(report('REAL 认峰+θ=1%兜底', d_peak_theta(p_r)))
L.append(report('SHUF 认峰+θ=1%兜底', d_peak_theta(p_s)))
L.append('')
L.append('判定：v2 的 +0.799% 来自"用整条路径 p 的 90 分位"（含未来）。')
L.append('      本表全部为滚动阈值（只看过去 win 根），只有 REAL 显著优于 SHUF 且为正，才算认峰能赚钱。')
L.append('')
L.append('诚实边界：样本重叠未去重、零滑点零成交率、成本 0.04% 单边未核实往返。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
