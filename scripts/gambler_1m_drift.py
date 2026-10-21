# -*- coding: utf-8 -*-
"""gambler_1m_drift · 1m 粒度：未来 60 分钟的「累计方向」可测吗？

与之前两个实验的区别：
  upside_1m 的标签 = max(high) >= +5%（「摸到」= 波动）
  sym_1m   统计的也是「摸到」
  本脚本的标签 = close[t+60] / close[t] - 1（收盘对收盘 = 方向）

三件事：
  1. 基础：未来 60 分钟 close-to-close 涨跌概率与幅度
  2. 按波动率分档：波动大的时候，方向偏哪边
  3. GBDT 预测方向，AUC；按概率分档看胜率与幅度
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 20

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_1m_drift · 未来 60 分钟「累计方向」可测吗（close-to-close）')
log('=' * 100)

FS, RF, TS = [], [], []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','quote_volume','volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','volume':'float32',
                             'trades':'float32','taker_base':'float32'},
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
        if m <= 300:
            continue
        cc = c[:m]
        fwd = c[HOLD:] / cc - 1.0           # close-to-close 60 分钟
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,volr,tbr,trr,vstd,vstd1440,hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        FS.append(F[idx].astype('float32'))
        RF.append(fwd[idx].astype('float32'))
        TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(FS); rf = np.concatenate(RF); t = np.concatenate(TS)
del FS, RF, TS; gc.collect()
log(f'样本 {len(X):,}')

cut = np.median(t); trm = t <= cut; tem = t > cut
log(f'训练段 {trm.sum():,}  测试段 {tem.sum():,}')

log('\n【一】基础：未来 60 分钟 close-to-close 方向')
log(f'  涨的概率 {(rf[tem]>0).mean():.2%}   均值 {rf[tem].mean():+.4%}   中位 {np.median(rf[tem]):+.4%}')
log(f'  涨>=1% {(rf[tem]>=0.01).mean():.2%}   跌<=-1% {(rf[tem]<=-0.01).mean():.2%}')
log(f'  涨>=2% {(rf[tem]>=0.02).mean():.2%}   跌<=-2% {(rf[tem]<=-0.02).mean():.2%}')

log('\n【二】按 vstd 分档：波动大的时候方向偏哪？')
qs = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
v = X[:,7]
edges = np.quantile(v[trm], qs)
log(f'  {"vstd档":<26s}{"n":>9s}{"涨概率":>8s}{"均值":>9s}{"中位":>9s}{"涨>=1%":>8s}{"跌<=-1%":>9s}')
for i in range(len(qs)-1):
    lo, hi = edges[i], edges[i+1]
    sel = (v[tem] >= lo) & (v[tem] < hi) if i < len(qs)-2 else (v[tem] >= lo)
    nn = sel.sum()
    if nn == 0: continue
    log(f'  [{lo:.5f},{hi:.5f})  {nn:>9,}{(rf[tem][sel]>0).mean():>8.1%}{rf[tem][sel].mean():>+9.3%}'
        f'{np.median(rf[tem][sel]):>+9.3%}{(rf[tem][sel]>=0.01).mean():>8.1%}{(rf[tem][sel]<=-0.01).mean():>9.1%}')

log('\n【三】模型：预测 ret_fwd > 0')
Y = (rf > 0).astype(np.int8)
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                   min_samples_leaf=50, random_state=7)
m.fit(X[trm], Y[trm])
p = m.predict_proba(X[tem])[:, 1]
log(f'  AUC = {roc_auc_score(Y[tem], p):.4f}   基准概率 {Y[tem].mean():.2%}')

log('\n【四】按预测概率分档：方向胜率与幅度')
qs2 = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
e2 = np.quantile(p, qs2)
log(f'  {"档":<18s}{"n":>9s}{"涨概率":>8s}{"均值":>9s}{"中位":>9s}{"涨>=1%":>8s}{"跌<=-1%":>9s}')
for i in range(len(qs2)-1):
    lo, hi = e2[i], e2[i+1]
    sel = (p >= lo) & (p < hi) if i < len(qs2)-2 else (p >= lo)
    nn = sel.sum()
    if nn == 0: continue
    log(f'  [{lo:.3f},{hi:.3f})  {nn:>9,}{(rf[tem][sel]>0).mean():>8.1%}{rf[tem][sel].mean():>+9.3%}'
        f'{np.median(rf[tem][sel]):>+9.3%}{(rf[tem][sel]>=0.01).mean():>8.1%}{(rf[tem][sel]<=-0.01).mean():>9.1%}')

log('\n【五】单特征 AUC 对照（预测方向）')
for i, nm in enumerate(['r1','r5','r15','r60','volr','tbr','trr','vstd','vstd1440','hour']):
    try:
        a = roc_auc_score(Y[tem], X[tem][:, i])
        log(f'  {nm:8s} {a:.4f}' + ('  ← 反向' if a < 0.5 else ''))
    except Exception:
        pass

log('\n【判读】')
log('  · 若按 vstd 分档「涨概率」≈50% ⇒ 波动大不改变方向（纯波动）')
log('  · 若模型 AUC < 0.55 ⇒ 累计方向不可测')
log('  · 若高档「涨概率」显著 > 50% 且均值 > 0 ⇒ 方向有微弱可测性')

out = os.path.join(OUT, 'drift_1m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
