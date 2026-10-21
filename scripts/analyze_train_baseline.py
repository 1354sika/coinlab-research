# -*- coding: utf-8 -*-
"""
问题：现在能不能筛出样本、训模型了？
做法：按"他的杠杆口径"设标签——8h 内标的涨 >= 5%（10x 杠杆 => 本金 +50%）。
      特征只用 K 线（他说"他只看 K 线"）；时间切分训练/验证。
输出 coinlab/evidence/train_baseline_REAL.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_502.csv'
OUT = 'D:/中信证券/coinlab/evidence/train_baseline_REAL.txt'
H = 96; STEP = 96; TH = 0.05

D = pd.read_csv(CSV, usecols=['symbol','open_time','open','high','low','close','volume'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','low','close','volume'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

X_parts, y_parts, t_parts = [], [], []
for sym, g in D.groupby('symbol', sort=False):
    o = g.open.to_numpy(); h = g.high.to_numpy(); l = g.low.to_numpy()
    c = g.close.to_numpy(); v = g.volume.to_numpy(); t = g.open_time.to_numpy()
    n = len(o)
    if n < 300 + H + 2:
        continue
    idx = np.arange(300, n - H, STEP)
    if len(idx) == 0:
        continue
    from numpy.lib.stride_tricks import sliding_window_view
    fmax = sliding_window_view(h, H).max(axis=1)
    r24 = c / np.concatenate((np.full(288, np.nan, dtype=np.float32), c[:-288])) - 1
    r6h = c / np.concatenate((np.full(72, np.nan, dtype=np.float32), c[:-72])) - 1
    vma = pd.Series(v).rolling(288, min_periods=288).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, np.nan)
    sd = pd.Series(np.log(np.maximum(c, 1e-9))).diff().rolling(H, min_periods=H).std().to_numpy()
    lo = pd.Series(l).rolling(H, min_periods=H).min().to_numpy()
    hi = pd.Series(h).rolling(H, min_periods=H).max().to_numpy()
    pos = np.where(hi > lo, (c - lo) / (hi - lo), 0.5)
    F = np.column_stack([r24[idx], r6h[idx], volr[idx], sd[idx], pos[idx]])
    fut = fmax[idx] / np.maximum(o[idx], 1e-12) - 1
    X_parts.append(F); y_parts.append((fut >= TH).astype(np.int8)); t_parts.append(t[idx])

X = np.vstack(X_parts); y = np.concatenate(y_parts); tt = np.concatenate(t_parts)
m = np.isfinite(X).all(axis=1)
X, y, tt = X[m], y[m], tt[m]
print('样本', len(y), '正', int(y.sum()), '%.1fs' % (time.time()-t0), flush=True)

L = []
L.append('筛选 + 基线模型（标签按杠杆口径：8h 内标的涨 >= 5%%，10x => 本金 +50%%）   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据：klines_5m_502.csv（304 币）；每币每 8h 一个样本；特征只用 K 线、只用过去')
L.append('样本 %d  正样本 %d (%.2f%%)' % (len(y), int(y.sum()), y.mean() * 100))
L.append('')

cut = np.percentile(tt, 70)
tr = tt < cut; va = ~tr
names = ['r24', 'r6h', 'volr', 'sd96', 'pos96']

def fit_lr(Xtr, ytr, Xva, yva, l2=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd
    A = np.column_stack([np.ones(len(A)), A]); B = np.column_stack([np.ones(len(B)), B])
    w = np.zeros(A.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - ytr) / len(A) + l2 * np.r_[0, w[1:]] / len(A)
        w -= 0.5 * g
    p = 1 / (1 + np.exp(-np.clip(B @ w, -30, 30)))
    return w, p

def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan

L.append('=== 一、单特征 AUC（验证集）')
for i, nm in enumerate(names):
    L.append('  %-6s AUC %.3f' % (nm, auc(X[va, i], y[va])))
L.append('')
L.append('=== 二、五特征逻辑回归（时间切分：前 70%% 训练 / 后 30%% 验证）')
w, p = fit_lr(X[tr], y[tr], X[va], y[va])
L.append('  验证集 AUC %.3f   正样本基准率 %.2f%%' % (auc(p, y[va]), y[va].mean() * 100))
for q in (0.90, 0.95, 0.99):
    thr = np.quantile(p, q)
    sel = p >= thr
    L.append('  取 p 前 %2.0f%%：命中率 %.2f%%（基准 %.2f%%，提升 %4.1f 倍）  覆盖正样本 %.1f%%' % (
        (1 - q) * 100, y[va][sel].mean() * 100, y[va].mean() * 100,
        y[va][sel].mean() / max(y[va].mean(), 1e-9), y[va][sel].sum() / max(y[va].sum(), 1) * 100))
L.append('')
L.append('=== 三、读法')
L.append('  AUC 0.55~0.60：弱但可能有边际价值；0.60~0.65 可用；<0.52 等于没有。')
L.append('  命中率提升倍数才是实盘意义：取前 1%% 若命中率 3 倍于基准，才值得做筛选器。')
L.append('')
L.append('诚实边界：')
L.append('  1. 标签是"8h 内 high 触到 +5%%"，浮盈口径，不是可成交收益。')
L.append('  2. 样本重叠（同币相邻 8h 不重叠，但同波行情跨样本仍相关）；未做事件去重。')
L.append('  3. 特征 5 个、逻辑回归线性模型；未调参、未做特征工程。')
L.append('  4. 零滑点、未计成交率；304 币含小市值，流动性未评估。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
