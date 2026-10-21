# -*- coding: utf-8 -*-
"""gambler_downall · 穷尽现有 1m 数据的看跌能力：增强特征 + 多模型融合

特征从 9 个扩到 ~22 个：多周期涨跌、K 线形态、量价背离、波动率比、周期相位。
模型：GBDT + RandomForest + LogisticRegression 三模型软投票。
输出：AUC（跌1%/跌2%）、各阈值召回率-精确率、以及"抓 50% 下跌时的精确率"。
对照：上一版单 GBDT（AUC 0.741，抓 50% 时精确率 45.8%）。
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
log('gambler_downall · 穷尽 K 线内信息：22 特征 + 三模型融合 → 每次下跌能否判出')
log('=' * 100)

XS, YD1, YD2, TS = [], [], [], []
FEATS = ['r1','r5','r15','r30','r60','r240','volr','tbr','trr','vstd',
         'us','ls','dn_streak','div','vratio','tbr_chg',
         'sin24','cos24','sin7','cos7','hlr','body']
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','open','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','open':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        o = g['open'].values.astype('float64')
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
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        d1 = (fmin <= cc * 0.99).astype('int8')
        d2 = (fmin <= cc * 0.98).astype('int8')
        span = np.maximum(cc, 1e-12)
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r30, r60, r240 = ret(1), ret(5), ret(15), ret(30), ret(60), ret(240)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        tbma = pd.Series(tbr).rolling(30, min_periods=10).mean().values
        tbr_chg = tbr - tbma
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd60 = pd.Series(r1).rolling(60, min_periods=20).std().values
        vratio = vstd / np.maximum(vstd60, 1e-12)
        us = (h[:m] - np.maximum(c[:m], o[:m])) / span          # 上影线
        ls = (np.minimum(c[:m], o[:m]) - l[:m]) / span          # 下影线
        hlr = (h[:m] - l[:m]) / span                            # 振幅
        body = np.abs(c[:m] - o[:m]) / span
        dn = (r1 < 0).astype('float64')
        dn_streak = pd.Series(dn).rolling(5, min_periods=1).sum().values
        div = np.sign(r15) * np.sign(np.log(np.maximum(volr, 1e-12)))   # 量价背离散
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        dow = ((ot[:m] // 86400000) % 7).astype('float64')
        F = np.column_stack([r1,r5,r15,r30,r60,r240,volr,tbr,trr,vstd,
                             us,ls,dn_streak,div,vratio,tbr_chg,
                             np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                             np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                             hlr,body])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YD1.append(d1[idx]); YD2.append(d2[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); yd1 = np.concatenate(YD1); yd2 = np.concatenate(YD2); t = np.concatenate(TS)
del XS, YD1, YD2, TS; gc.collect()
log(f'样本 {len(X):,}  特征 {X.shape[1]}  跌1%正例 {yd1.mean():.1%}  跌2%正例 {yd2.mean():.1%}  读盘 {time.time()-t0:.0f}s')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

cut = np.median(t)
tr = t <= cut; te = t > cut
N = te.sum(); D1 = yd1[te].sum(); D2 = yd2[te].sum()
log(f'训练 {tr.sum():,}  测试 {N:,}（真跌1% {D1:,}={D1/N:.1%}，真跌2% {D2:,}={D2/N:.1%}）')

log('\n【三模型】')
t_fit = time.time()
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=7, subsample=0.8)
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=7)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, C=0.5))
probs = {}
for nm, mdl in [('GBDT', gb), ('RF', rf), ('LR', lr)]:
    mdl.fit(X[tr], yd1[tr])
    pp = mdl.predict_proba(X[te])[:, 1]
    probs[nm] = pp
    log(f'  {nm:5s} AUC(跌1%)={roc_auc_score(yd1[te], pp):.4f}  AUC(跌2%)={roc_auc_score(yd2[te], pp):.4f}')
log(f'  训练总耗时 {time.time()-t_fit:.0f}s')

p = (probs['GBDT'] + probs['RF'] + probs['LR']) / 3.0
log(f'\n【融合（三模型平均）】AUC(跌1%)={roc_auc_score(yd1[te], p):.4f}  AUC(跌2%)={roc_auc_score(yd2[te], p):.4f}')
log('  对照上一版单 GBDT：AUC(跌1%)=0.7414  AUC(跌2%)=0.8130')

log('\n【按阈值】')
log(f'{"阈值":8s}{"报出":>9s}{"占样本":>8s}{"精确率":>8s}{"召回率":>8s}{"跌2%命中":>9s}')
log('-' * 100)
for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    m = p >= thr
    if m.sum() == 0:
        continue
    log(f'P>={thr:.2f}{m.sum():>9,}{m.sum()/N:>8.1%}{yd1[te][m].mean():>8.1%}'
        f'{yd1[te][m].sum()/D1:>8.1%}{yd2[te][m].mean():>9.1%}')

log('\n【召回率-精确率曲线】')
log(f'{"目标召回":10s}{"阈值":8s}{"实际召回":10s}{"精确率":10s}{"跌2%命中":>10s}')
log('-' * 100)
order = np.argsort(-p)
cum = np.cumsum(yd1[te][order])
for target in [0.25, 0.50, 0.75, 0.90]:
    k = min(max(int(np.searchsorted(cum, target * D1)) + 1, 1), N)
    thr = p[order][k - 1]
    m = p >= thr
    log(f'{target:>8.0%}{thr:>8.2f}{yd1[te][m].sum()/D1:>10.1%}{yd1[te][m].mean():>10.1%}'
        f'{yd2[te][m].mean():>10.1%}')

# 特征重要性
log('\n【GBDT 特征重要性 Top12】')
for f, v in sorted(zip(FEATS, gb.feature_importances_), key=lambda x: -x[1])[:12]:
    log(f'  {f:10s} {v:.3f}')

log('\n【判读】')
log('  · 与上一版比：AUC 若提升 < 0.01 ⇒ K 线内信息已榨干，换特征没用')
log('  · 抓 50% 下跌时精确率若仍 < 50% ⇒ 「每次跌」在 K 线层面做不到')

out = os.path.join(OUT, 'downall_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
