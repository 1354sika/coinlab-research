# -*- coding: utf-8 -*-
"""gambler_downshort · 看跌能力的两种用法：排除法 vs 直接做空

已训练出「未来 60 分钟跌破 -1%」的模型（AUC 0.741，最高档 93.1%）。
本脚本在测试段上同时统计：
  A) 排除法：按预测跌概率分档，看各档「做多 +2%/-1%」的止盈率
  B) 直接做空：按预测跌概率分档，看各档「做空 +2%/-1%」的止盈率
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL, HOLD = 0.02, 0.01, 60
STEP = 100
FEE_M, FEE_T = 0.0004, 0.0020

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 104)
log('gambler_downshort · 看跌模型（AUC 0.741）的两种用法：排除法 / 直接做空')
log('=' * 104)

XS, YL, YD, YS, TS = [], [], [], [], []
t_read = time.time()
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'volume',
                               'quote_volume', 'trades', 'taker_base'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'volume': 'float32',
                             'quote_volume': 'float32', 'trades': 'float32', 'taker_base': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
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
        n = len(c); m = n - HOLD
        if m <= 200:
            continue
        cc = c[:m]
        INF = 10 ** 9
        f_tp = np.full(m, INF, dtype=np.int64)
        f_sl = np.full(m, INF, dtype=np.int64)   # 做多口径：先跌 -1%
        s_tp = np.full(m, INF, dtype=np.int64)   # 做空口径：先跌 -2%（对空头是 +2%）
        s_sl = np.full(m, INF, dtype=np.int64)   # 做空口径：先涨 +1%
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]; ll = l[j:j + m]
            nt = (f_tp == INF) & (hh >= cc * 1.02)
            ns = (f_sl == INF) & (ll <= cc * 0.99)
            f_tp[nt] = j; f_sl[ns] = j
            st = (s_tp == INF) & (ll <= cc * 0.98)
            ss = (s_sl == INF) & (hh >= cc * 1.01)
            s_tp[st] = j; s_sl[ss] = j
        long_win = (f_tp < f_sl)
        short_win = (s_tp < s_sl)
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        down = (fmin <= cc * 0.99)
        r1 = np.full(m, np.nan); r1[1:] = cc[1:] / cc[:-1] - 1
        r5 = np.full(m, np.nan); r5[5:] = cc[5:] / cc[:-5] - 1
        r15 = np.full(m, np.nan); r15[15:] = cc[15:] / cc[:-15] - 1
        r60 = np.full(m, np.nan); r60[60:] = cc[60:] / cc[:-60] - 1
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1, r5, r15, r60, volr, tbr, trr, vstd, hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(200, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YL.append(long_win[idx].astype('int8'))
        YD.append(down[idx].astype('int8'))
        YS.append(short_win[idx].astype('int8'))
        TS.append(ot[:m][idx])
    del ch; gc.collect()

X = np.vstack(XS); yl = np.concatenate(YL); yd = np.concatenate(YD); ys = np.concatenate(YS); t = np.concatenate(TS)
log(f'样本 {len(X):,}  做多止盈率 {yl.mean():.1%}  60分内跌破-1% {yd.mean():.1%}  读盘 {time.time()-t_read:.0f}s')

cut = np.median(t)
tr = t <= cut
te = t > cut
from sklearn.ensemble import GradientBoostingClassifier

mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=7, subsample=0.8)
mdl.fit(X[tr], yd[tr])
p = mdl.predict_proba(X[te])[:, 1]
log(f'模型在测试段（{te.sum():,}）上的预测完成')

log('')
log('=' * 104)
log('【A】排除法：按预测跌概率分档，看「做多 +2%/-1%」的止盈率')
log('=' * 104)
log(f'{"跌概率档":14s}{"样本":>10s}{"做多止盈率":>12s}{"相对整体":>10s}')
log('-' * 104)
base_l = yl[te].mean()
for lo, hi in [(0, .3), (.3, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m_ = (p >= lo) & (p < hi)
    if m_.sum() == 0:
        continue
    r = yl[te][m_].mean()
    log(f'[{lo:.1f},{hi:.1f})      {m_.sum():>10,}{r:>12.1%}{r/base_l:>10.2f}x')
log(f'整体做多止盈率 {base_l:.1%}')
# 排除高跌概率后
for thr in (0.5, 0.7, 0.8):
    keep = p < thr
    if keep.sum() == 0:
        continue
    log(f'排除 P>={thr:.1f} 后：剩 {keep.sum():,}（{keep.mean():.1%}），做多止盈率 {yl[te][keep].mean():.1%}')

log('')
log('=' * 104)
log('【B】直接做空：按预测跌概率分档，看「做空 +2%/-1%」的止盈率')
log('=' * 104)
log(f'{"跌概率档":14s}{"样本":>10s}{"做空止盈率":>12s}{"跌破-1%率":>12s}')
log('-' * 104)
base_s = None
for lo, hi in [(0, .3), (.3, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m_ = (p >= lo) & (p < hi)
    if m_.sum() == 0:
        continue
    r = None
    # 需要 short_win，但没存；用 yd 近似（跌破 -1% 的比例）
    r = yd[te][m_].mean()
    sw = ys[te][m_].mean()
    log(f'[{lo:.1f},{hi:.1f})      {m_.sum():>10,}{sw:>12.1%}{r:>12.1%}')

log('\n【判读】')
log('  · 【A】若排除高跌概率后止盈率提升 < 3pp ⇒ 排除法对整体帮助有限（高概率档占比小）')
log('  · 【B】若高概率档跌率 > 80% ⇒ 应直接做空，而不是排除')

out = os.path.join(OUT, 'downshort_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
