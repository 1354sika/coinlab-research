# -*- coding: utf-8 -*-
"""gambler_owncycle · 币自己的过去周期对现在的影响

上一版只到 4 小时（r240）。本实验补上「自身长周期」：
  r1440(24h) / r4320(3d) / r10080(7d) 涨跌
  距 7 天最高点的回撤 dd7、距 7 天最低点的涨幅 up7
  24h 波动率 vstd1440、24h 量比 volr1440
标签：未来 60 分钟跌 1% / 跌 2%
对照：22 特征三模型融合（AUC 0.7497 / 0.8210）
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
log('gambler_owncycle · 币自身过去周期（24h/3d/7d 位置）→ 未来 60 分钟下跌')
log('=' * 100)

XS, YD1, YD2, TS = [], [], [], []
FEATS = ['r1','r5','r15','r60','r240','r1440','r4320','r10080',
         'dd7','up7','vstd','vstd1440','volr','volr1440','tbr','trr','hour']
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 12000:      # 至少 8 天，否则 7d 特征为空
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
        if m <= 12000:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        d1 = (fmin <= cc * 0.99).astype('int8')
        d2 = (fmin <= cc * 0.98).astype('int8')
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60, r240 = ret(1), ret(5), ret(15), ret(60), ret(240)
        r1440, r4320, r10080 = ret(1440), ret(4320), ret(10080)
        hi7 = pd.Series(h).rolling(10080, min_periods=1440).max().values[:m]
        lo7 = pd.Series(l).rolling(10080, min_periods=1440).min().values[:m]
        dd7 = cc / np.maximum(hi7, 1e-12) - 1
        up7 = cc / np.maximum(lo7, 1e-12) - 1
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        qma1440 = pd.Series(q).rolling(1440, min_periods=240).mean().values[:m]
        volr1440 = np.where(qma1440 > 0, q[:m] / np.maximum(qma1440, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd1440 = pd.Series(r1).rolling(1440, min_periods=240).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,r1440,r4320,r10080,
                             dd7,up7,vstd,vstd1440,volr,volr1440,tbr,trr,hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(10080, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YD1.append(d1[idx]); YD2.append(d2[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); yd1 = np.concatenate(YD1); yd2 = np.concatenate(YD2); t = np.concatenate(TS)
del XS, YD1, YD2, TS; gc.collect()
log(f'样本 {len(X):,}  特征 {X.shape[1]}  跌1%正例 {yd1.mean():.1%}  跌2%正例 {yd2.mean():.1%}')
log(f'（只保留有 ≥8 天历史、且样本从第 7 天起算的币，导致样本变少属正常）')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

cut = np.median(t)
tr = t <= cut; te = t > cut
N = te.sum(); D1 = yd1[te].sum()
log(f'训练 {tr.sum():,}  测试 {N:,}（真跌1% {D1:,}={D1/N:.1%}）')

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=7, subsample=0.8)
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=7)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, C=0.5))
probs = {}
for nm, mdl in [('GBDT', gb), ('RF', rf), ('LR', lr)]:
    mdl.fit(X[tr], yd1[tr])
    pp = mdl.predict_proba(X[te])[:, 1]
    probs[nm] = pp
    log(f'  {nm:5s} AUC(跌1%)={roc_auc_score(yd1[te], pp):.4f}  AUC(跌2%)={roc_auc_score(yd2[te], pp):.4f}')

p = (probs['GBDT'] + probs['RF'] + probs['LR']) / 3.0
log(f'\n【融合（自身周期特征）】AUC(跌1%)={roc_auc_score(yd1[te], p):.4f}  AUC(跌2%)={roc_auc_score(yd2[te], p):.4f}')
log('  对照（22 特征无长周期）：AUC(跌1%)=0.7497  AUC(跌2%)=0.8210')

log('\n【召回率-精确率】')
log(f'{"目标召回":10s}{"阈值":8s}{"精确率":10s}{"跌2%命中":>10s}')
log('-' * 60)
order = np.argsort(-p)
cum = np.cumsum(yd1[te][order])
for target in [0.25, 0.50, 0.75, 0.90]:
    k = min(max(int(np.searchsorted(cum, target * D1)) + 1, 1), N)
    thr = p[order][k - 1]
    mm = p >= thr
    log(f'{target:>8.0%}{thr:>8.2f}{yd1[te][mm].mean():>10.1%}{yd2[te][mm].mean():>10.1%}')

log('\n【GBDT 特征重要性 Top12】')
for f, v in sorted(zip(FEATS, gb.feature_importances_), key=lambda x: -x[1])[:12]:
    log(f'  {f:10s} {v:.3f}')

log('\n【判读】')
log('  · 自身 24h/3d/7d 周期特征若进 Top 与提升 AUC > 0.01 ⇒ 周期历史有用')
log('  · 若仍约 0.75 ⇒ 长周期位置也不带增量信息')

out = os.path.join(OUT, 'owncycle_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
