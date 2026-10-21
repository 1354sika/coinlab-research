# -*- coding: utf-8 -*-
"""gambler_train2.py — 修正版：用 304 币全量（klines_5m_502）重训
上一版（60 大币）操盘局 +100%/4h 只有 3 个正例 → 样本选错。
本版：顺序扫 502 文件，累积到 N_BARS_LIMIT 为止（文件按 symbol 分块），取前 150 个币。
标签三档：涨 +5%/1h、崩 -10%/1h、操盘局 +50%/4h（比 +100% 现实）
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
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_train2.txt')
N_COIN_LIMIT = 150

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_train2 · 304 币全量重训（修上一版"操盘局无正例"）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'volume', 'quote_volume', 'taker_quote']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
      'low': 'float32', 'volume': 'float32', 'quote_volume': 'float32', 'taker_quote': 'float32'}

parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=1_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:N_COIN_LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= N_COIN_LIMIT:
                break           # 币已取满，遇到下一个币 ⇒ 前面 150 个币的数据已读全
            seen = seen + new[:N_COIN_LIMIT - len(seen)]
    ch = ch[ch['symbol'].isin(seen)]
    parts.append(ch)
    log(f'  已读 {len(parts)*100:,} 万行  币 {len(seen)}  耗时 {time.time()-t0:.0f}s')

df = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
df['symbol'] = df['symbol'].astype('category')
log(f'总计行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0:.0f}s')

FNAMES = ['ret3', 'ret6', 'ret12', 'ret24', 'ret48', 'ret96', 'ret192', 'ret288',
          'volr12', 'volr48', 'volr192',
          'vol12', 'vol48', 'vol192',
          'hh24', 'll24', 'hh96', 'll96', 'hh288', 'll288',
          'trr', 'trr_dev', 'up12', 'dn12']
STEP = 6
Xs, y1, y2, y3, tsv, sv = [], [], [], [], [], []
SYMS = []

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

def rmin(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).min().values

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 2000:
        continue
    si = len(SYMS); SYMS.append(str(sym))
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
        m = rmean(v, k); F['volr%d' % k] = np.where(m > 0, v / np.maximum(m, 1e-12), np.nan)
        mq = rmean(q, k); F['vol%d' % k] = np.where(mq > 0, q / np.maximum(mq, 1e-12), np.nan)
    for k in (24, 96, 288):
        F['hh%d' % k] = rmax(h, k) / c - 1.0
        F['ll%d' % k] = rmin(l, k) / c - 1.0
    tr = tq / np.maximum(q, 1e-12)
    F['trr'] = tr; F['trr_dev'] = tr - rmean(tr, 48)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F['up12'] = rmean((r1 > 0).astype('float64'), 12)
    F['dn12'] = rmean((r1 < 0).astype('float64'), 12)

    fu12 = np.full(n, np.nan); fu12[:-12] = c[12:] / c[:-12] - 1.0
    fu48 = np.full(n, np.nan); fu48[:-48] = c[48:] / c[:-48] - 1.0
    idx = np.arange(0, n - 48, STEP)
    Xs.append(np.stack([F[k][idx] for k in FNAMES], axis=1).astype('float32'))
    y1.append((fu12[idx] >= 0.05).astype('int8'))
    y2.append((fu12[idx] <= -0.10).astype('int8'))
    y3.append((fu48[idx] >= 0.50).astype('int8'))
    tsv.append(ot[idx]); sv.append(np.full(len(idx), si, dtype='int32'))

del df; gc.collect()
X = np.concatenate(Xs); Y1 = np.concatenate(y1); Y2 = np.concatenate(y2)
Y3 = np.concatenate(y3); TS = np.concatenate(tsv); SI = np.concatenate(sv)
del Xs, y1, y2, y3, tsv, sv; gc.collect()
ok = ~np.isnan(X).any(axis=1)
X, Y1, Y2, Y3, TS, SI = X[ok], Y1[ok], Y2[ok], Y3[ok], TS[ok], SI[ok]
log(f'样本 {len(X):,}  特征 {X.shape[1]}  正例率 涨{100*Y1.mean():.3f}% 崩{100*Y2.mean():.3f}% 局{100*Y3.mean():.3f}%')

cut = np.quantile(TS, 0.70)
tr = TS < cut; te = ~tr
log(f'训练 {tr.sum():,}  测试 {te.sum():,}  切分 {pd.to_datetime(int(cut), unit="ms")}')

def thr_table(y, p, ts, si, name):
    log(f'--- {name}  测试基准 {100*y.mean():.3f}%  AUC={roc_auc_score(y, p):.4f}')
    log(f'    概率分布 max={p.max():.4f}  p99.9={np.quantile(p,0.999):.4f}  p99={np.quantile(p,0.99):.4f}  中位={np.median(p):.5f}')
    best = None
    for thr in np.arange(0.30, 1.0, 0.01):
        sel = p >= thr; n = int(sel.sum())
        if n == 0:
            continue
        prec = y[sel].mean()
        if prec >= 0.85 and (best is None or n > best[3]):
            best = (thr, prec, y[sel].sum() / max(y.sum(), 1), n)
    if best:
        log(f'  ★ 逐点 85%精确率最低阈值={best[0]:.2f}  报出 {best[3]:,}（覆盖 {100*best[3]/len(y):.2f}%）  精确率 {100*best[1]:.1f}%  召回 {100*best[2]:.1f}%')
    else:
        log('  ✗ 逐点：无阈值达到 85% 精确率')
    log('    [事件级] 连续报出（间隔<=1h）算一局，取每局首点')
    for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        ev_y = []
        for c in range(len(SYMS)):
            m = np.flatnonzero((p >= thr) & (si == c))
            if len(m) == 0:
                continue
            o = np.argsort(ts[m]); m = m[o]; t = ts[m]
            brk = np.flatnonzero(np.diff(t) > 12 * 300000)
            starts = np.concatenate([[m[0]], m[brk + 1]]) if len(brk) else m[:1]
            ev_y.extend(y[starts].tolist())
        ev_y = np.array(ev_y, dtype='int8')
        if len(ev_y) == 0:
            log(f'      thr={thr:.1f} 无报出')
        else:
            log(f'      thr={thr:.2f}  局 {len(ev_y):,}  精确率 {100*ev_y.mean():.1f}%')

for tag, y in (('涨 +5%/1h', Y1), ('崩 -10%/1h', Y2), ('操盘局 +50%/4h', Y3)):
    log('=' * 100)
    log(f'【{tag}】训练正例 {int(y[tr].sum()):,} / {int(tr.sum()):,}')
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=42)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    thr_table(y[te], p, TS[te], SI[te], tag)
    fn = os.path.join(MD, 'g2_%s.joblib' % tag.replace(' ', '_').replace('/', '_').replace('%', 'pct').replace('+', 'p'))
    joblib.dump({'model': m, 'features': FNAMES}, fn)
    log('  模型已存', fn)

log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
