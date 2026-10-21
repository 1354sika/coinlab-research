# -*- coding: utf-8 -*-
"""gambler_train.py — 正式训练：涨 / 崩 / 操盘 三模型
口径：判据是期望净收益；本脚本只量"识别力"（AUC / 85% 精确率下的覆盖与召回）
切分：时间留出（前 70% 训练 / 后 30% 测试），不做随机切分（避免前视）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

t0 = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
os.makedirs(MD, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_train.txt')

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_train · 正式训练（时间留出：前70%训练 / 后30%测试）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'volume', 'quote_volume', 'taker_quote']
dt = {'symbol': 'category', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
      'low': 'float32', 'volume': 'float32', 'quote_volume': 'float32', 'taker_quote': 'float32'}
log('读取', SRC)
df = pd.read_csv(SRC, usecols=cols, dtype=dt)
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0:.0f}s')

FNAMES = ['ret3', 'ret6', 'ret12', 'ret24', 'ret48', 'ret96', 'ret192', 'ret288',
          'volr12', 'volr48', 'volr192',
          'vol12', 'vol48', 'vol192',
          'hh24', 'll24', 'hh96', 'll96', 'hh288', 'll288',
          'trr', 'trr_dev', 'up12', 'dn12']
STEP = 6            # 每 6 根（30 分钟）取一个样本，控内存
Xs, ys_up, ys_dn, ys_x2, tsv = [], [], [], [], []

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

def rmin(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).min().values

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 500:
        continue
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    v = g['volume'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64')
    tq = g['taker_quote'].values.astype('float64')
    ot = g['open_time'].values

    F = {}
    for k in (3, 6, 12, 24, 48, 96, 192, 288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
    for k in (12, 48, 192):
        m = rmean(v, k)
        F['volr%d' % k] = np.where(m > 0, v / np.maximum(m, 1e-12), np.nan)
        mq = rmean(q, k)
        F['vol%d' % k] = np.where(mq > 0, q / np.maximum(mq, 1e-12), np.nan)
    for k in (24, 96, 288):
        F['hh%d' % k] = rmax(h, k) / c - 1.0
        F['ll%d' % k] = rmin(l, k) / c - 1.0
    tr = tq / np.maximum(q, 1e-12)
    F['trr'] = tr
    F['trr_dev'] = tr - rmean(tr, 48)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    up = (r1 > 0).astype('float64'); dn = (r1 < 0).astype('float64')
    F['up12'] = rmean(up, 12); F['dn12'] = rmean(dn, 12)

    fu12 = np.full(n, np.nan); fu12[:-12] = c[12:] / c[:-12] - 1.0
    fu24 = np.full(n, np.nan); fu24[:-24] = c[24:] / c[:-24] - 1.0
    fu48 = np.full(n, np.nan); fu48[:-48] = c[48:] / c[:-48] - 1.0

    idx = np.arange(0, n - 48, STEP)
    X = np.stack([F[k][idx] for k in FNAMES], axis=1).astype('float32')
    yup = (fu12[idx] >= 0.05).astype('int8')
    yx2 = (fu48[idx] >= 1.00).astype('int8')
    ydn = (fu12[idx] <= -0.10).astype('int8')
    Xs.append(X); ys_up.append(yup); ys_x2.append(yx2); ys_dn.append(ydn); tsv.append(ot[idx])

del df; gc.collect()
X = np.concatenate(Xs); y_up = np.concatenate(ys_up); y_x2 = np.concatenate(ys_x2)
y_dn = np.concatenate(ys_dn); ts = np.concatenate(tsv)
del Xs, ys_up, ys_x2, ys_dn, tsv; gc.collect()

ok = ~np.isnan(X).any(axis=1)
X, y_up, y_x2, y_dn, ts = X[ok], y_up[ok], y_x2[ok], y_dn[ok], ts[ok]
log(f'样本 {len(X):,}  特征 {X.shape[1]}  耗时 {time.time()-t0:.0f}s')

cut = np.quantile(ts, 0.70)
tr = ts < cut; te = ~tr
log(f'训练 {tr.sum():,}  测试 {te.sum():,}  切分时刻 {pd.to_datetime(int(cut), unit="ms")}')

def thr_table(y, p, name):
    base = 100 * y.mean()
    log(f'--- {name}  测试基准正例率 {base:.3f}%   AUC={roc_auc_score(y, p):.4f}')
    best = None
    for thr in np.arange(0.50, 1.0, 0.01):
        sel = p >= thr
        n = int(sel.sum())
        if n == 0:
            continue
        prec = y[sel].mean()
        if prec >= 0.85 and (best is None or n > best[3]):
            best = (thr, prec, y[sel].sum() / max(y.sum(), 1), n)
    if best:
        log(f'  ★ 85%精确率最低阈值={best[0]:.2f}  报出 {best[3]:,}（覆盖 {100*best[3]/len(y):.2f}%）  精确率 {100*best[1]:.1f}%  召回 {100*best[2]:.1f}%')
    else:
        log('  ✗ 无阈值达到 85% 精确率')

MODELS = {}
for tagname, y in (('涨 +5%/1h', y_up), ('崩 -10%/1h', y_dn), ('操盘局 +100%/4h', y_x2)):
    log('=' * 100)
    log(f'【{tagname}】训练 {int(y[tr].sum()):,} 正例 / {int(tr.sum()):,}')
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=None,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       random_state=42)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    thr_table(y[te], p, tagname)
    fn = os.path.join(MD, 'gambler_%s.joblib' % tagname.replace(' ', '_').replace('/', '_').replace('%', 'pct'))
    joblib.dump({'model': m, 'features': FNAMES}, fn)
    log('  模型已存', fn)
    MODELS[tagname] = (roc_auc_score(y[te], p), fn)

log('=' * 100)
log('汇总（测试段 AUC）')
for k, (a, fn) in MODELS.items():
    log(f'  {k:22s} AUC={a:.4f}')
log(f'TOTAL {time.time()-t0:.0f}s')

open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
