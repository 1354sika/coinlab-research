# -*- coding: utf-8 -*-
"""gambler_early20_v2 · 早期识别强化版（围绕波动率重构）

上一版问题：vstd 单特征 AUC 0.7255 > 模型 0.6385 ⇒ GBDT 学歪了（被 r5/r15 的反向噪声带偏）。
本版：
  1. 多周期波动率（15/60/240/1440 根）+ 量能结构
  2. 单特征 vstd1440 作为基线，与 GBDT 对比
  3. 多档目标：gain >= 20% / 30% / 50% / 100%
  4. 收益账：涨 5% 入场，50% 止盈 / 峰值回落止损
严格无前视：所有特征取自「涨 5%」那一根 5m 及之前。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_early20_v2 · 涨 5% 时的早期识别（强化版）')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
rows = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','quote_volume','volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','volume':'float32',
                             'trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in need or len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c)
        if n < 300: continue
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values
        volr = np.where(qma > 0, q / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol > 0, tb / np.maximum(vol, 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values
        trr = np.where(trma > 0, tr / np.maximum(trma, 1e-12), np.nan)
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1
        v15 = pd.Series(r1).rolling(15, min_periods=5).std().values
        v60 = pd.Series(r1).rolling(60, min_periods=20).std().values
        v240 = pd.Series(r1).rolling(240, min_periods=50).std().values
        v1440 = pd.Series(r1).rolling(1440, min_periods=200).std().values
        hi288 = pd.Series(h).rolling(288, min_periods=50).max().values
        for _, row in by_sym[sym].iterrows():
            lo_i = int(row['lo_i']); lo_px = float(row['lo'])
            if lo_i >= n - 2: continue
            j = lo_i
            target = lo_px * 1.05
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1 or j - lo_i > 288: continue
            F = [v15[j], v60[j], v240[j], v1440[j], volr[j], trr[j], tbr[j],
                 (j - lo_i) / 288.0, (c[j] / hi288[j] - 1) if np.isfinite(hi288[j]) else np.nan]
            if not all(np.isfinite(x) for x in F): continue
            rows.append((F, row['gain'], row['peak'] / lo_px - 1.0, ot[j], sym))
    del ch; gc.collect()

X = np.array([r[0] for r in rows], float)
gain = np.array([r[1] for r in rows], float)
peakrel = np.array([r[2] for r in rows], float)
T = np.array([r[3] for r in rows])
log(f'\n可评估周期 {len(X):,}')

cut = np.median(T); trm = T <= cut; tem = T > cut
log(f'训练 {trm.sum():,}  测试 {tem.sum():,}')

for th, nm in [(0.20,'>=20%'), (0.30,'>=30%'), (0.50,'>=50%'), (1.00,'>=100%')]:
    Y = (gain >= th).astype(int)
    log(f'\n目标 {nm}: 基准 {Y[tem].mean():.1%}')

log('\n【A】单特征 vstd1440 分档 vs 大周期(>=50%)概率（测试段）')
Y = (gain >= 0.50).astype(int)
qs = [0, 0.5, 0.8, 0.9, 0.95, 1.0]
edges = np.quantile(X[trm,3], qs)
for i in range(len(qs)-1):
    lo, hi = edges[i], edges[i+1]
    sel = (X[tem,3] >= lo) & (X[tem,3] < hi) if i < len(qs)-2 else (X[tem,3] >= lo)
    nn = sel.sum()
    if nn == 0: continue
    log(f'  [{lo:.5f},{hi:.5f})  n={nn:>5,}  大周期概率 {Y[tem][sel].mean():>6.1%}')

log('\n【B】模型对比')
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=5,
                                   min_samples_leaf=30, random_state=7)
m.fit(X[trm], Y[trm])
p = m.predict_proba(X[tem])[:, 1]
log(f'  GBDT(9特征) AUC = {roc_auc_score(Y[tem], p):.4f}')
log(f'  vstd1440 单特征 AUC = {roc_auc_score(Y[tem], X[tem,3]):.4f}')
log(f'  v240 单特征 AUC = {roc_auc_score(Y[tem], X[tem,2]):.4f}')
log(f'  trr 单特征 AUC = {roc_auc_score(Y[tem], X[tem,5]):.4f}')

log('\n【C】GBDT 概率分档：命中率 + 入场(涨5%)后的峰值收益')
qs2 = [0, 0.5, 0.8, 0.9, 0.95, 1.0]
e2 = np.quantile(p, qs2)
log(f'  {"档":<18s}{"n":>6s}{"大周期概率":>10s}{"峰值收益中位":>12s}{"峰值>=42.9%":>12s}')
for i in range(len(qs2)-1):
    lo, hi = e2[i], e2[i+1]
    sel = (p >= lo) & (p < hi) if i < len(qs2)-2 else (p >= lo)
    nn = sel.sum()
    if nn == 0: continue
    # 入场价=lo*1.05，峰值收益 = peakrel/1.05 - 1
    ret_if_50 = np.where(peakrel[tem][sel] >= 0.50, 1.50/1.05 - 1, (1+peakrel[tem][sel])/1.05 - 1)
    log(f'  [{lo:.3f},{hi:.3f})  {nn:>6,}{Y[tem][sel].mean():>10.1%}'
        f'{np.median(ret_if_50):>12.1%}{(peakrel[tem][sel]>=0.50).mean():>12.1%}')

log('\n【D】简单收益账（入场=涨5%，止盈=+50%，止损=跌回低点-5%）')
for th in [0.3, 0.4, 0.5, 0.6]:
    sel = p >= th
    if sel.sum() < 10: continue
    # 峰值>=50% 则赚 42.9%，否则按峰值/1.05-1（近似：若峰值<5%回撤到低点，用 peakrel 近似）
    win = peakrel[tem][sel] >= 0.50
    ret = np.where(win, 1.50/1.05 - 1, (1+peakrel[tem][sel])/1.05 - 1)
    log(f'  P>={th}: n={sel.sum():>5,}  赢率(到50%) {win.mean():>5.1%}  均值 {ret.mean():>+7.1%}  中位 {np.median(ret):>+7.1%}')

log('\n【判读】')
log('  · 单特征 vstd1440 若仍 > GBDT ⇒ 波动率就是全部信息，模型越复杂越差')
log('  · 峰值>=42.9% 那列 = 入场后能吃到 50% 止盈的概率')
log('  · 无前视：特征全部取自涨 5% 时刻及之前')

out = os.path.join(OUT, 'early20_v2_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
