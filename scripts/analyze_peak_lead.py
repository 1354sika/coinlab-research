"""
直接回答：模型是"事先觉得是峰"，还是"峰出现时才觉得"？

方法一（提前量实验）：把特征整体前移 lag 根，用 t-lag 的信息预测"t 是否峰"。
    AUC 随 lag 的衰减曲线 = 信息能提前多少根。
    lag=0 就是现状（0.684）。若 lag=1 掉到 0.55，说明信息几乎全在"当根"。
方法二（概率曲线 z 分数）：以非峰样本的 p 为基线，
    看峰前 k 根的 p 相对基线抬高多少个标准差、从哪一根开始显著。
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_lead_REAL.txt'
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

n_e = len(entries)
X = np.asarray(Xl, np.float32)
y = np.asarray(yl, np.int8)
ei_l = np.asarray(ei_l)
F = X.shape[1]
assert len(X) == n_e * H, (len(X), n_e, H)
X3 = X.reshape(n_e, H, F)
y3 = y.reshape(n_e, H)
q3 = np.array([str(qm_all[e]) for e in range(n_e)])


def auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b]); r = pd.Series(allv).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


L = []
L.append('模型是"事先觉得是峰"还是"峰出现时才觉得"?  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集口径：前 %d 季训练，后 9 季验证；样本 %d 笔 × %d 根' % (VALID_FROM_Q, n_e, H))
L.append('')
L.append('=== 一、提前量实验：特征整体前移 lag 根，还能认出"这根是峰"吗')
L.append('  %-6s %10s %10s' % ('lag(根)', '验证 AUC', '相对 lag=0'))
base_auc = None
rows = []
for lag in (0, 1, 2, 3, 6, 12):
    Xs = np.empty_like(X3)
    Xs[:, lag:, :] = X3[:, :H - lag, :]
    if lag > 0:
        Xs[:, :lag, :] = X3[:, :1, :]
    Xf = Xs.reshape(-1, F)
    yf = y3.reshape(-1)
    qf = np.repeat(q3, H)
    jj = np.tile(np.arange(H), n_e)
    keep = jj >= lag
    trm = (qf < str(qs[VALID_FROM_Q])) & keep
    vam = (qf >= str(qs[VALID_FROM_Q])) & keep
    mu = Xf[trm].mean(0); sd = Xf[trm].std(0) + 1e-9
    m = LogisticRegression(max_iter=800).fit((Xf[trm] - mu) / sd, yf[trm])
    pv = m.predict_proba((Xf[vam] - mu) / sd)[:, 1]
    a = auc(pv[yf[vam] == 1], pv[yf[vam] == 0])
    if lag == 0:
        base_auc = a
    rows.append((lag, a))
    L.append('  %-6d %10.3f %9.1f%%' % (lag, a, a / base_auc * 100 if base_auc else 0))
L.append('')

# 二、概率曲线 z 分数（用 lag=0 模型）
Xs = X3.reshape(-1, F); yf = y3.reshape(-1); qf = np.repeat(q3, H)
trm = qf < str(qs[VALID_FROM_Q]); vam = ~trm
mu = Xs[trm].mean(0); sd = Xs[trm].std(0) + 1e-9
m = LogisticRegression(max_iter=800).fit((Xs[trm] - mu) / sd, yf[trm])
pv = m.predict_proba((Xs[vam] - mu) / sd)[:, 1]
yv = yf[vam]
qva = qf[vam]
ei_va = np.repeat(np.arange(n_e), H)[vam]
jj_va = np.tile(np.arange(H), n_e)[vam]

pos_p = pv[yv == 1]; neg_p = pv[yv == 0]
L.append('=== 二、p 的水平（验证集）')
L.append('  非峰样本平均 p = %.3f （标准差 %.3f）' % (neg_p.mean(), neg_p.std()))
L.append('  峰样本平均 p   = %.3f' % pos_p.mean())
L.append('  差距 = %.3f 个 p 单位 = %.1f 个非峰标准差'
         % (pos_p.mean() - neg_p.mean(), (pos_p.mean() - neg_p.mean()) / (neg_p.std() + 1e-9)))
L.append('')

# 以每个 entry 的真实峰为 0，看 p 的曲线
L.append('=== 三、以"真实峰"为 0 的 p 曲线（每笔各自的峰）')
L.append('  %-6s %10s %12s' % ('k(根)', 'p 均值', '相对基线z'))
mean_pos = neg_p.mean(); sd_neg = neg_p.std() + 1e-9
for k in range(-6, 4):
    vals = []
    for e in np.unique(ei_va):
        sel = ei_va == e
        if sel.sum() != H:
            continue
        hh = paths[entries[e][1]]['h'][entries[e][2]:entries[e][2] + H] / paths[entries[e][1]]['c'][entries[e][2]] - 1.0
        ip = int(np.argmax(hh))
        if 0 <= ip + k < H:
            vals.append(pv[sel][ip + k])
    if vals:
        L.append('  %-6d %10.3f %12.2f' % (k, np.mean(vals), (np.mean(vals) - mean_pos) / sd_neg))
L.append('')
L.append('读法：')
L.append('  lag 实验：AUC 在 lag=1 掉多少 = "提前一根还认得多少"。掉到 0.55 以下说明信息几乎全在当根。')
L.append('  z 曲线：峰前 k 根的 p 若明显高于基线，说明有"预感"；只有 k=0 才高，说明"当根才认"。')
L.append('诚实边界：样本重叠；特征含 r12/r24 等回看量，lag 实验是整体平移，不完全等于实盘提前决策。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
