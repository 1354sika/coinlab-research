# -*- coding: utf-8 -*-
"""gambler_early20 · 周期早期识别

问题：周期"涨到 5%"的时候，能不能判断它会不会涨到 50%？
        如果能，就在 5% 处入场，50% 处撤（或峰值回落撤），而不是等涨到 20% 才确认。

数据：cycles.csv（已切好的周期：lo_i 低点索引、lo 低点价、gain 峰值涨幅）
      klines_5m_3y.csv（60 币 3 年 5m）
标签：该周期 gain >= 50%（大周期）vs < 50%
样本：每个周期在「close >= lo*1.05」的第一根 5m 上取特征
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
cyc['big'] = (cyc['gain'] >= 0.5).astype(int)
log('=' * 100)
log('gambler_early20 · 涨 5% 时能不能判断它会涨到 50%')
log('=' * 100)
log(f'周期 {len(cyc):,} 个  币 {cyc.sym.nunique()}  大周期(gain>=50%) {cyc.big.sum():,} ({cyc.big.mean():.1%})')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
FS, Y, TS = [], [], []
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
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values
        volr = np.where(qma > 0, q / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol > 0, tb / np.maximum(vol, 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values
        trr = np.where(trma > 0, tr / np.maximum(trma, 1e-12), np.nan)
        r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
        vstd = pd.Series(r).rolling(15, min_periods=5).std().values
        for _, row in by_sym[sym].iterrows():
            lo_i = int(row['lo_i']); lo_px = float(row['lo'])
            if lo_i >= n - 1: continue
            target = lo_px * 1.05
            j = lo_i
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1: continue
            if j - lo_i > 288: continue          # 涨 5% 用了超过 24h，跳过
            F = [c[j]/c[max(j-5,0)]-1, c[j]/c[max(j-15,0)]-1, c[j]/c[max(j-60,0)]-1,
                 volr[j], tbr[j], trr[j], vstd[j], (j-lo_i)/288.0]
            if not all(np.isfinite(x) for x in F): continue
            FS.append(F); Y.append(row['big']); TS.append(ot[j])
    del ch; gc.collect()

X = np.array(FS, dtype='float64'); Y = np.array(Y); T = np.array(TS)
log(f'\n可评估周期 {len(X):,}（涨 5% 在 24h 内的）  大周期占比 {Y.mean():.1%}')
if len(X) < 500:
    log('样本不足，停止')
else:
    cut = np.median(T); trm = T <= cut; tem = T > cut
    log(f'训练 {trm.sum():,}  测试 {tem.sum():,}')
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                       min_samples_leaf=30, random_state=7)
    m.fit(X[trm], Y[trm])
    p = m.predict_proba(X[tem])[:, 1]
    log(f'\nAUC（涨5%时预测会到50%）= {roc_auc_score(Y[tem], p):.4f}  基准 {Y[tem].mean():.1%}')
    for th in [0.3, 0.5, 0.7]:
        sel = p >= th
        if sel.sum() == 0: continue
        log(f'  P>={th}: 报出 {sel.sum():,} ({sel.mean():.1%})  命中(大周期) {Y[tem][sel].mean():.1%}')
    log('\n单特征 AUC:')
    for i, nm in enumerate(['r5','r15','r60','volr','tbr','trr','vstd','bars_since_lo']):
        try:
            log(f'  {nm:14s} {roc_auc_score(Y[tem], X[tem][:,i]):.4f}')
        except Exception:
            pass

log('\n【判读】')
log('  · AUC > 0.65 ⇒ 涨 5% 时能分辨大周期/小周期，可在 5% 处入场、50% 处撤')
log('  · AUC ≈ 0.5 ⇒ 早期识别不了，只能等确认（涨到 20%）')

out = os.path.join(OUT, 'early20_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
