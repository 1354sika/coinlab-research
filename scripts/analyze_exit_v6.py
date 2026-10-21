"""
出场 V6：修正训练方式的三个错误
错1 标签：从「是否创新高」(二分类) 改为「离顶多远 regret」(回归)
         regret = 未来 H 根最高价 / 当前价 - 1，直接预测「继续持有多赚多少」
错2 样本：从「随机入场」改为「浮盈>0 的真实出场场景」
         真实场景人只在有浮盈时才考虑走；负浮盈样本教的是另一件事
错3 特征：加 ATR 归一化，让 1% 在 BTC 和 PEPE 上可比
"""
import numpy as np, pandas as pd, time, os
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v6_REAL.txt'
H     = 96          # 持有窗口 96 根 = 8 小时
W     = 32          # 状态窗口
COST  = 0.0004      # maker 往返
GAP   = 12          # 每 12 根(1h)采一个样本
MIN_Q = 4           # 前 4 个季度做初始训练

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))

def build(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(float); l = g.low.to_numpy(float); c = g.close.to_numpy(float)
    v = g.volume.to_numpy(float); t = g.open_time.to_numpy()
    n = len(c)
    if n < W + H + 300: return None
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    # t 时刻「未来 H 根最高价」= max(h[t], ..., h[t+H-1])
    fmax = pd.Series(h[::-1]).rolling(H, min_periods=H).max().to_numpy()[::-1]
    idx = np.arange(W, n - H, GAP)
    if len(idx) == 0: return None
    ent = idx - 6
    g_now = c[idx] / c[ent] - 1.0
    ok = (g_now > 0) & np.isfinite(atr[idx]) & (atr[idx] > 0) & np.isfinite(fmax[idx])
    idx = idx[ok]; g_now = g_now[ok]
    if len(idx) == 0: return None
    atn = atr[idx] / c[idx]                # ATR 占价格比
    offs = np.arange(-W + 1, 1)
    pr = c[idx[:, None] + offs]
    Xp = (pr / c[idx][:, None] - 1.0) / atn[:, None]
    Xv = volr[idx[:, None] + offs]
    X = np.column_stack([Xp, Xv, g_now / atn, np.full(len(idx), 6.0)])
    y = fmax[idx] / c[idx] - 1.0 - COST   # 继续持有的最大可得多赚（扣成本）
    return (X.astype(np.float32), y.astype(np.float32),
            g_now.astype(np.float32), t[idx], atn.astype(np.float32))

Xl, Yl, Gl, Tl, Al = [], [], [], [], []
for sym, g in df.groupby('symbol', sort=False):
    r = build(g)
    if r is None: continue
    Xl.append(r[0]); Yl.append(r[1]); Gl.append(r[2]); Tl.append(r[3]); Al.append(r[4])
del df
X = np.vstack(Xl); Y = np.concatenate(Yl); G = np.concatenate(Gl)
T = np.concatenate(Tl); A = np.concatenate(Al)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print('samples %d  prep %.1fs' % (len(X), time.time()-t0))

qm = pd.to_datetime(T, unit='ms').to_period('Q')
qs = sorted(qm.unique())
print('quarters %d: %s .. %s' % (len(qs), qs[0], qs[-1]))

lines = []
lines.append('出场 V6（标签=regret 回归 / 样本=浮盈>0 / 特征=ATR 归一化）')
lines.append('H=%d根  W=%d根  扣成本 %.2f%%  样本 %d' % (H, W, COST*100, len(X)))
lines.append('')
lines.append('%-10s %10s %10s %10s %10s %10s %8s' %
             ('季度','模型卖/笔','oracle续持','错过中位','捕获率','卖出占比','n'))

mo_all = []; or_all = []; reg_all = []
qres = []
for i in range(MIN_Q, len(qs)):
    q = qs[i]
    te = (qm == q)
    if te.sum() < 200: continue
    tr = qm < q
    if tr.sum() < 5000: continue
    mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                        max_depth=6, min_samples_leaf=60,
                                        l2_regularization=1.0, random_state=0)
    mdl.fit(X[tr], Y[tr])
    ph = mdl.predict(X[te])
    # 决策：预测「继续持有多赚」< 当前浮盈 → 卖
    g_te = G[te]
    sell = ph < g_te
    # 实际收益：卖 → g_te-COST；持 → 真实 fmax 收益（oracle，乐观上界）
    r_sell = g_te - COST
    r_hold = Y[te] + COST + 0.0
    R = np.where(sell, r_sell, r_hold)
    reg = r_hold - R   # 相对 oracle 的错过
    qres.append((q, R.mean(), r_hold.mean(), np.median(reg), sell.mean(), te.sum()))
    mo_all.append(R); or_all.append(r_hold); reg_all.append(reg)

for q, m, o, md, sp, n in qres:
    lines.append('%-10s %9.3f%% %9.3f%% %9.3f%% %9s %9.1f%% %8d' %
                 (q, m*100, o*100, md*100, '—', sp*100, n))

M = np.concatenate(mo_all); O = np.concatenate(or_all); RG = np.concatenate(reg_all)
pos = sum(1 for _, m, _, _, _, _ in qres if m > 0)
lines.append('')
lines.append('季度为正比例 %d/%d (%.0f%%)' % (pos, len(qres), 100.0*pos/len(qres)))
lines.append('模型卖 均值 %.4f%%/笔   继续持 oracle 均值 %.4f%%/笔' % (M.mean()*100, O.mean()*100))
lines.append('错过 regret 中位 %.4f%%  捕获率 %.1f%%' % (np.median(RG)*100, 100.0*(1 - np.median(RG)/O.mean()) if O.mean() else 0))
lines.append('卖出占比 %.1f%%' % (100.0*np.mean(M != O)))

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
