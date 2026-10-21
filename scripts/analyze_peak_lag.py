"""
认峰的"滞后"测量：AUC 0.684 为什么不能变成钱？

v3 的事实：认峰 AUC 0.684（SHUF 0.505，真信号），但按 p 卖反而更差（-0.385% vs 不卖 -0.195%）。
最可能的解释：模型 p 的峰值出现在**真实见顶之后**（信号滞后），所以"确信是峰"时已经跌了一段。
本脚本直接测这个滞后：
  idx_peak = 路径内 high 最高的那根
  idx_pmax = 模型 p 最高的那根
  lag = idx_pmax - idx_peak   （>0 = 模型晚了 lag 根；<0 = 提前）
另测 p 在顶峰前后各根的平均值曲线，看信号到底"提前"还是"滞后"。
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_lag_REAL.txt'
H, GAP, W = 96, 24, 48
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
p = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr]).predict_proba(Xs[va])[:, 1]
print('训练完成 %.1fs' % (time.time() - t0), flush=True)

va_e = np.unique(ei_l[va])
pos = {e: np.where(ei_l[va] == e)[0] for e in va_e}
lags, pk_ret, p_at = [], [], {k: [] for k in range(-6, 7)}
for e in va_e:
    P = paths[entries[e][1]]; i = entries[e][2]
    h, c = P['h'], P['c']
    hh = h[i:i + H] / c[i] - 1.0
    ip = int(np.argmax(hh))
    ps = p[pos[e]]
    ipm = int(np.argmax(ps))
    lags.append(ipm - ip); pk_ret.append(float(hh[ip]))
    for k in range(-6, 7):
        j = ip + k
        if 0 <= j < len(ps):
            p_at[k].append(ps[j])

lags = np.array(lags)
L = []
L.append('认峰滞后测量：为什么 AUC 0.684 不能变成钱  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集 %d 笔 | 峰 = 路径内 high 最高那根 | p = 认峰模型输出' % len(va_e))
L.append('')
L.append('=== 一、模型 p 最高点相对真实峰的滞后（根，1 根=5 分钟）')
L.append('  中位 %+d 根  P25 %+d  P75 %+d  P90 %+d'
         % (int(np.median(lags)), int(np.percentile(lags, 25)),
            int(np.percentile(lags, 75)), int(np.percentile(lags, 90))))
L.append('  提前（lag<0） %.1f%%   同时（lag=0） %.1f%%   晚 1~3 根 %.1f%%   晚 >3 根 %.1f%%'
         % ((lags < 0).mean() * 100, (lags == 0).mean() * 100,
            ((lags > 0) & (lags <= 3)).mean() * 100, (lags > 3).mean() * 100))
L.append('')
L.append('=== 二、峰前后各根的模型 p 均值（k=0 即峰所在根）')
for k in range(-6, 7):
    if p_at[k]:
        L.append('  k=%+d  p 均值 %.3f  (n=%d)' % (k, np.mean(p_at[k]), len(p_at[k])))
L.append('')
L.append('=== 三、读法')
L.append('  若滞后中位 >0 且 p 曲线在 k>0 才升，则"确信是峰"时已经跌了几根 ——')
L.append('  这就是 AUC 高（全局排序对）却决策亏（卖点在峰后）的原因。')
L.append('')
L.append('诚实边界：样本重叠、零滑点；标签含未来 12 根（正确）；特征严格只用 j 及之前。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
