# -*- coding: utf-8 -*-
"""
问题：602 个"8h 内到过 +50%"的正样本，能不能训出能用的入场模型？
做法：对全池每个 8h 入场点提 5 个"只用过去"的 K 线特征，算单特征 AUC（正 vs 负）。
输出 coinlab/evidence/positives_features.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_502.csv'
OUT = 'D:/中信证券/coinlab/evidence/positives_features.txt'
H = 96; STEP = 96

D = pd.read_csv(CSV, usecols=['symbol','open_time','open','high','low','close','volume'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','low','close','volume'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

feats, labels, syms, ts = [], [], [], []
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
    fmax = sliding_window_view(h, H).max(axis=1)     # fmax[i] = max(h[i:i+H])
    r24 = c / np.concatenate((np.full(288, np.nan, dtype=np.float32), c[:-288])) - 1
    r6h = c / np.concatenate((np.full(72, np.nan, dtype=np.float32), c[:-72])) - 1
    vma = pd.Series(v).rolling(288, min_periods=288).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, np.nan)
    lr = pd.Series(np.log(np.maximum(c, 1e-9))).diff().rolling(H, min_periods=H).std().to_numpy()
    lo = pd.Series(l).rolling(H, min_periods=H).min().to_numpy()
    hi = pd.Series(h).rolling(H, min_periods=H).max().to_numpy()
    pos = np.where(hi > lo, (c - lo) / (hi - lo), 0.5)

    F = np.column_stack([r24[idx], r6h[idx], volr[idx], lr[idx], pos[idx]])
    fut = fmax[idx] / np.maximum(o[idx], 1e-12) - 1
    lab = (fut >= 0.5).astype(np.int8)
    feats.append(F); labels.append(lab); syms.append(np.full(len(idx), sym)); ts.append(t[idx])
    print('  %-12s %6d 正 %3d  %.0fs' % (sym, len(idx), int(lab.sum()), time.time()-t0), flush=True)

F = np.vstack(feats); y = np.concatenate(labels); sy = np.concatenate(syms); tt = np.concatenate(ts)
names = ['r24', 'r6h', 'volr', 'sd96', 'pos96']
print('总入场点', len(y), '正样本', int(y.sum()), '%.1fs' % (time.time()-t0), flush=True)

L = []
L.append('"到过 +50%%"正样本的可分性（能否训出场模型）   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据：304 币 / 5m；入场点 %d；正样本(8h内到过+50%%) %d' % (len(y), int(y.sum())))
L.append('特征全部只用 j 及之前（无前视）：r24 24h涨幅 / r6h 6h涨幅 / volr 量比 / sd96 波动率 / pos96 96根内价格位置')
L.append('')
L.append('=== 一、单特征区分力（AUC，0.5=无用）')
def auc(x, y):
    m = np.isfinite(x)
    x, yy = x[m], y[m]
    order = np.argsort(x)
    r = np.empty(len(x)); r[order] = np.arange(1, len(x)+1)
    n1 = yy.sum(); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[yy == 1].sum() - n1*(n1+1)/2) / (n1*n0)
for i, nm in enumerate(names):
    a = auc(F[:, i], y)
    L.append('  %-6s AUC %6.3f   正样本均值 %+9.4f  负样本均值 %+9.4f' % (
        nm, a, np.nanmean(F[y == 1, i]), np.nanmean(F[y == 0, i])))
L.append('')
L.append('=== 二、正样本的时间分布（每年/每币）')
q = pd.PeriodIndex(pd.to_datetime(tt[y == 1], unit='ms'), freq='Q').astype(str)
vc = pd.Series(q).value_counts().sort_index()
for k, v in vc.items():
    L.append('  %s  %4d' % (k, v))
L.append('')
L.append('=== 三、读法')
L.append('  AUC 最高若 < 0.60：单特征分不开，需要组合;若 >0.65 才有"训模型"的意义。')
L.append('  正样本 %d 个，去重后（同币同波行情）更少；按经验只能训 3~5 特征的小模型。' % int(y.sum()))
L.append('')
L.append('诚实边界：')
L.append('  1. 标签是"浮盈到过 50%%"（high 口径），不是可实现收益；零滑点、成交率未计。')
L.append('  2. 样本重叠（每 8h 切一刀），真实独立事件远少于正样本数。')
L.append('  3. 未做时间切分（训练/验证），AUC 是样本内的。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
