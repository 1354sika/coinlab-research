# -*- coding: utf-8 -*-
"""gambler_down24h_big · 24h 窗口改用「大跌幅」标签（跌 5% / 跌 10%）

上一版 24h 窗口的「跌 1%」基础概率 75.5%（几乎必然），AUC 只有 0.668，没有区分度。
本版把标签换成「未来 24h 内跌 5% / 跌 10%」，看 24 小时尺度的征兆对「真崩」有没有预测力。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 1440
STEP = 200

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_down24h_big · 24h 窗口 + 大跌幅标签（-5% / -10%）')
log('=' * 100)

XS, Y5, Y10, TS = [], [], [], []
FEATS = ['r1','r15','r60','r240','r1440','vstd','vstd1440','volr','volr1440','tbr','trr','hour']
FEATS += [f'seg{i}' for i in range(12)]

for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 20000:
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
        if m <= 3000:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        y5 = (fmin <= cc * 0.95).astype('int8')
        y10 = (fmin <= cc * 0.90).astype('int8')
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r15, r60, r240, r1440 = ret(1), ret(15), ret(60), ret(240), ret(1440)
        segs = [ret(120 * (i + 1)) - ret(120 * i) if i > 0 else ret(120) for i in range(12)]
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
        F = np.column_stack([r1,r15,r60,r240,r1440,vstd,vstd1440,volr,volr1440,tbr,trr,hour] + segs)
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(1440, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        Y5.append(y5[idx]); Y10.append(y10[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); y5 = np.concatenate(Y5); y10 = np.concatenate(Y10); t = np.concatenate(TS)
del XS, Y5, Y10, TS; gc.collect()
log(f'样本 {len(X):,}  特征 {X.shape[1]}  未来24h跌5%正例 {y5.mean():.1%}  跌10%正例 {y10.mean():.1%}')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

cut = np.median(t)
tr = t <= cut; te = t > cut
N = te.sum()
log(f'训练 {tr.sum():,}  测试 {N:,}（跌5% {y5[te].sum():,}={y5[te].mean():.1%}，跌10% {y10[te].sum():,}={y10[te].mean():.1%}）')

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=7, subsample=0.8)
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=7)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, C=0.5))
probs = {}
for nm, mdl in [('GBDT', gb), ('RF', rf), ('LR', lr)]:
    mdl.fit(X[tr], y5[tr])
    pp = mdl.predict_proba(X[te])[:, 1]
    probs[nm] = pp
    log(f'  {nm:5s} AUC(跌5%)={roc_auc_score(y5[te], pp):.4f}  AUC(跌10%)={roc_auc_score(y10[te], pp):.4f}')

p = (probs['GBDT'] + probs['RF'] + probs['LR']) / 3.0
log(f'\n【融合（24h窗口+大跌幅）】AUC(跌5%)={roc_auc_score(y5[te], p):.4f}  AUC(跌10%)={roc_auc_score(y10[te], p):.4f}')
log('  对照：60 分钟窗口跌1% = 0.7497，24h 窗口跌1% = 0.6677（基础概率 75% 太高）')

log('\n【跌5% 的召回率-精确率】')
log(f'{"目标召回":10s}{"阈值":8s}{"精确率":10s}{"跌10%命中":>10s}')
log('-' * 60)
D5 = y5[te].sum()
order = np.argsort(-p)
cum = np.cumsum(y5[te][order])
for target in [0.25, 0.50, 0.75, 0.90]:
    k = min(max(int(np.searchsorted(cum, target * D5)) + 1, 1), N)
    thr = p[order][k - 1]
    mm = p >= thr
    log(f'{target:>8.0%}{thr:>8.2f}{y5[te][mm].mean():>10.1%}{y10[te][mm].mean():>10.1%}')

log('\n【GBDT 特征重要性 Top12】')
for f, v in sorted(zip(FEATS, gb.feature_importances_), key=lambda x: -x[1])[:12]:
    log(f'  {f:10s} {v:.3f}')

out = os.path.join(OUT, 'down24h_big_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
