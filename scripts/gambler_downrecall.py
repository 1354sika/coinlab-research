# -*- coding: utf-8 -*-
"""gambler_downrecall · 回答「每一次跌能不能都高概率判出来」

关键区分：
  精确率 precision = 模型报「要跌」时，真的跌的比例（93% 那个数字）
  召回率 recall    = 所有真的跌里，模型报出来的比例（这才是「每一次跌」）

输出：不同阈值下的 覆盖率/精确率/召回率/做空事件命中率；
以及「要抓到 50%、80% 的下跌，精确率会掉到多少」。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
STEP = 100

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_downrecall · 每一次跌都能判出来吗：精确率 vs 召回率')
log('=' * 100)

XS, YD1, YD2, TS = [], [], [], []
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
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 200:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        d1 = (fmin <= cc * (1 - 0.01)).astype('int8')
        d2 = (fmin <= cc * (1 - 0.02)).astype('int8')
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
        YD1.append(d1[idx]); YD2.append(d2[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); yd1 = np.concatenate(YD1); yd2 = np.concatenate(YD2); t = np.concatenate(TS)
del XS, YD1, YD2, TS; gc.collect()
log(f'样本 {len(X):,}  跌1%正例 {yd1.mean():.1%}  跌2%正例 {yd2.mean():.1%}')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
cut = np.median(t)
tr = t <= cut; te = t > cut
mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=7, subsample=0.8)
mdl.fit(X[tr], yd1[tr])
p = mdl.predict_proba(X[te])[:, 1]
log(f'测试段 AUC(跌1%) = {roc_auc_score(yd1[te], p):.3f}   AUC(跌2%) = {roc_auc_score(yd2[te], p):.3f}')

N = te.sum()
D1 = yd1[te].sum(); D2 = yd2[te].sum()
log(f'\n测试段 {N:,} 个时刻，其中真跌 -1% 的 {D1:,} 个（{D1/N:.1%}），真跌 -2% 的 {D2:,} 个（{D2/N:.1%}）')
log('  精确率 = 报的里面真跌比例；召回率 = 真跌里被抓到的比例（这才是「每一次跌」）')

log(f'\n【按阈值】')
log(f'{"阈值":6s}{"报出":>8s}{"占全样本":>9s}{"精确率":>8s}{"召回率":>8s}{"跌2%命中":>9s}')
log('-' * 100)
for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    m = p >= thr
    if m.sum() == 0:
        continue
    log(f'P>={thr:.2f}{m.sum():>8,}{m.sum()/N:>9.1%}{yd1[te][m].mean():>8.1%}'
        f'{yd1[te][m].sum()/D1:>8.1%}{yd2[te][m].mean():>9.1%}')

log(f'\n【换个问法】要抓到 X% 的下跌，精确率会掉到多少？')
log(f'{"目标召回率":12s}{"需要的阈值":12s}{"实际召回":10s}{"精确率":10s}{"做空事件命中(跌2%)":>18s}')
log('-' * 100)
order = np.argsort(-p)
cum = np.cumsum(yd1[te][order])
for target in [0.25, 0.50, 0.75, 0.90]:
    k = int(np.searchsorted(cum, target * D1))
    k = min(max(k + 1, 1), N)
    thr = p[order][k - 1]
    m = p >= thr
    log(f'{target:>10.0%}{thr:>12.2f}{yd1[te][m].sum()/D1:>10.1%}{yd1[te][m].mean():>10.1%}'
        f'{yd2[te][m].mean():>18.1%}')

log('\n【判读】')
log('  · 若召回率 50% 时精确率 < 30% ⇒ 「每一次跌」做不到，只能抓最极端的少数')
log('  · 对照：当前 P>=0.9 的召回率就是上面 P>=0.90 那一行，看它有多低')

out = os.path.join(OUT, 'downrecall_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
