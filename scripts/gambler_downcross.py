# -*- coding: utf-8 -*-
"""gambler_downcross · 穷尽方式之最后一条：跨币联动（BTC/ETH 领先信号）

现有 22 个自身特征已榨干（AUC 0.741→0.750，+0.008）。
本实验加 12 个「大盘同期特征」：BTC/ETH 的 r1/r5/r15/r60/vstd/volr，
按同一分钟对齐到每个山寨币样本上，看能否提升看跌能力。
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
LEAD = ['BTCUSDT', 'ETHUSDT']

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_downcross · 跨币联动：BTC/ETH 同期特征加入后，看跌能力能否提升')
log('=' * 100)

# ---------- 第一遍：收集 BTC/ETH 的分钟特征 ----------
log('第一遍：扫 BTC/ETH 分钟序列…')
lead_raw = {s: [] for s in LEAD}
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','quote_volume'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','quote_volume':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in LEAD:
            continue
        g = g.sort_values('open_time')
        lead_raw[sym].append((g['open_time'].values.astype('int64'),
                              g['close'].values.astype('float64'),
                              g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

def mk_feat(ot, c, q):
    n = len(c)
    def ret(k):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1; return r
    r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
    qma = pd.Series(q).rolling(60, min_periods=10).mean().values
    volr = np.where(qma > 0, q / np.maximum(qma, 1e-12), np.nan)
    vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
    F = np.column_stack([r1, r5, r15, r60, vstd, volr])
    return ot, F

lead_grid = {}
for sym, parts in lead_raw.items():
    ot = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    o2, F = mk_feat(ot, c, q)
    lead_grid[sym] = (o2, F)
    log(f'  {sym}: {len(o2):,} 分钟')
del lead_raw; gc.collect()

# ---------- 第二遍：其他币样本 + 对齐 ----------
log('第二遍：其他币样本 + 对齐 BTC/ETH…')
XS, YD1, YD2, TS = [], [], [], []
FEATS = ['r1','r5','r15','r60','volr','tbr','trr','vstd','hour',
         'b_r1','b_r5','b_r15','b_r60','b_vstd','b_volr',
         'e_r1','e_r5','e_r15','e_r60','e_vstd','e_volr']
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym in LEAD or len(g) < 5000:
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
        if m <= 300:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        d1 = (fmin <= cc * 0.99).astype('int8')
        d2 = (fmin <= cc * 0.98).astype('int8')
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        own = np.column_stack([r1,r5,r15,r60,volr,tbr,trr,vstd,hour])
        idx = np.arange(300, m, STEP)
        ok = np.isfinite(own[idx]).all(axis=1) & np.isfinite(fmin[idx])
        idx = idx[ok]
        if len(idx) == 0:
            continue
        tt = ot[idx]
        lead_feats = []
        for sym2 in LEAD:
            lo, F2 = lead_grid[sym2]
            pos = np.searchsorted(lo, tt)
            pos = np.clip(pos, 0, len(lo) - 1)
            # 只接受同一分钟
            same = lo[pos] == tt
            Ff = F2[pos]
            Ff[~same] = np.nan
            lead_feats.append(Ff)
        F = np.hstack([own[idx]] + lead_feats).astype('float32')
        good = np.isfinite(F).all(axis=1)
        if good.sum() == 0:
            continue
        XS.append(F[good]); YD1.append(d1[idx][good]); YD2.append(d2[idx][good]); TS.append(tt[good])
    del ch; gc.collect()

X = np.vstack(XS); yd1 = np.concatenate(YD1); yd2 = np.concatenate(YD2); t = np.concatenate(TS)
del XS, YD1, YD2, TS; gc.collect()
log(f'样本 {len(X):,}  特征 {X.shape[1]}  跌1%正例 {yd1.mean():.1%}  跌2%正例 {yd2.mean():.1%}')

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
log(f'\n【融合+跨币】AUC(跌1%)={roc_auc_score(yd1[te], p):.4f}  AUC(跌2%)={roc_auc_score(yd2[te], p):.4f}')
log('  对照（同特征无跨币，三模型融合）：AUC(跌1%)=0.7497  AUC(跌2%)=0.8210')

log('\n【召回率-精确率】')
log(f'{"目标召回":10s}{"阈值":8s}{"精确率":10s}{"跌2%命中":>10s}')
log('-' * 60)
order = np.argsort(-p)
cum = np.cumsum(yd1[te][order])
for target in [0.25, 0.50, 0.75, 0.90]:
    k = min(max(int(np.searchsorted(cum, target * D1)) + 1, 1), N)
    thr = p[order][k - 1]
    m = p >= thr
    log(f'{target:>8.0%}{thr:>8.2f}{yd1[te][m].mean():>10.1%}{yd2[te][m].mean():>10.1%}')

log('\n【GBDT 特征重要性 Top10（看跨币是否被用到）】')
for f, v in sorted(zip(FEATS, gb.feature_importances_), key=lambda x: -x[1])[:10]:
    log(f'  {f:8s} {v:.3f}')

out = os.path.join(OUT, 'downcross_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
