# -*- coding: utf-8 -*-
"""gambler_window.py — 显性窗口版：窗口不写死，由数据定起止（上限 24h = 288 根 5m）
设计要点（用户 2026-10-16）：
  - 时间窗口不确定：可能 5 分钟，也可能 24 小时；模型和我们都不知道它何时开始/结束
  - 一切从数据出发 —— 因此把"段"显式构造为特征：
        bars_since_lo24 / bars_since_hi24   距 24h 内最低/最高点的根数（段起点由数据定）
        run_up / run_dn                     当前连续上涨/下跌根数
        seg_ret                             从 24h 低点到当前的涨幅（段内空间）
        seg_volr                            段内平均成交额倍率
  - 多尺度统计（6/12/24/48/96/144/288 根 = 30m~24h）全部喂给模型，让它自己挑尺度
  - 标签同样由数据定义：未来 24h 内是否发生 >=+50% / <=-30% 的行情（不写死 1h/4h）
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
OUT = os.path.join(EV, 'gambler_window.txt')
N_COIN_LIMIT = 150
WMAX = 288          # 窗口上限 24h（根）

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_window · 显性窗口（段由数据定起止，上限 24h）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'volume', 'quote_volume', 'trades', 'taker_quote']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32', 'low': 'float32',
      'volume': 'float32', 'quote_volume': 'float32', 'trades': 'float32', 'taker_quote': 'float32'}
parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=1_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:N_COIN_LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= N_COIN_LIMIT:
                break
            seen = seen + new[:N_COIN_LIMIT - len(seen)]
    parts.append(ch[ch['symbol'].isin(seen)])
df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
df['symbol'] = df['symbol'].astype('category')
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0:.0f}s')

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

def rmin(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).min().values

def argmax_dist(a, k):
    """距窗口内最大值位置的根数（0=当前就是最高）；数据定"段起点"用"""
    s = pd.Series(a)
    idx = s.rolling(k, min_periods=max(2, k // 4)).apply(np.argmax, raw=True).values
    pos = np.arange(len(a), dtype='float64')
    return pos - (pos - k + 1 + np.nan_to_num(idx, nan=0.0))

def argmin_dist(a, k):
    s = pd.Series(a)
    idx = s.rolling(k, min_periods=max(2, k // 4)).apply(np.argmin, raw=True).values
    pos = np.arange(len(a), dtype='float64')
    return pos - (pos - k + 1 + np.nan_to_num(idx, nan=0.0))

def run_len(sign):
    """当前连续同向根数（由数据定，不设上限）"""
    n = len(sign)
    out = np.zeros(n, dtype='float32')
    cur = 0
    for i in range(n):
        if sign[i] != 0 and (i == 0 or sign[i] == sign[i - 1]):
            cur = 1 if i == 0 else cur + 1
        elif sign[i] != 0:
            cur = 1
        else:
            cur = 0
        out[i] = cur
    return out

SCALES = (6, 12, 24, 48, 96, 144, 288)      # 30m / 1h / 2h / 4h / 8h / 12h / 24h
FNAMES = []
for k in SCALES:
    FNAMES += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FNAMES += ['since_hi24', 'since_lo24', 'run_up', 'run_dn', 'seg_ret', 'seg_volr', 'seg_bars',
           'trr_now', 'trr_dev48', 'up_frac24']

STEP = 3
Xs, yA, yB, yC, tsv, sv = [], [], [], [], [], []
SYMS = []

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 1000:
        continue
    si = len(SYMS); SYMS.append(str(sym))
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    v = g['volume'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64')
    tr = g['trades'].values.astype('float64')
    tq = g['taker_quote'].values.astype('float64')
    ot = g['open_time'].values
    F = {}
    for k in SCALES:
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(v, k); F['volr%d' % k] = np.where(mv > 0, v / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
        F['hh%d' % k] = rmax(h, k) / c - 1.0
        F['ll%d' % k] = rmin(l, k) / c - 1.0
    l288 = rmin(l, 288); h288 = rmax(h, 288)
    from numpy.lib.stride_tricks import sliding_window_view
    Wl = sliding_window_view(l, 288); Wh = sliding_window_view(h, 288)
    since_lo = np.full(n, np.nan); since_hi = np.full(n, np.nan)
    since_lo[287:] = 287 - Wl.argmin(axis=1)
    since_hi[287:] = 287 - Wh.argmax(axis=1)
    F['since_lo24'] = since_lo
    F['since_hi24'] = since_hi
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    sg = np.sign(np.nan_to_num(r1, nan=0.0))
    F['run_up'] = run_len(np.where(sg > 0, 1, 0)).astype('float64')
    F['run_dn'] = run_len(np.where(sg < 0, -1, 0)).astype('float64')
    F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0          # 从 24h 低点到现在的涨幅
    with np.errstate(invalid='ignore'):
        qseg = pd.Series(q).rolling(288, min_periods=72).mean().values
        F['seg_volr'] = np.where(qseg > 0, q / np.maximum(qseg, 1e-12), np.nan)
    F['seg_bars'] = since_lo
    trr_now = np.where(rmean(tr, 48) > 0, tr / np.maximum(rmean(tr, 48), 1e-12), np.nan)
    F['trr_now'] = trr_now
    F['trr_dev48'] = trr_now - rmean(np.nan_to_num(trr_now, nan=1.0), 48)
    F['up_frac24'] = rmean((r1 > 0).astype('float64'), 288)

    # 标签：未来 24h 内最大涨幅 / 最大跌幅（由数据定义，不写死 1h/4h）
    fmax = pd.Series(h).rolling(288, min_periods=1).max().shift(-288).values
    fmin = pd.Series(l).rolling(288, min_periods=1).min().shift(-288).values
    yA_ = (fmax / c - 1.0 >= 0.50).astype('int8')     # 未来 24h 内出现 >=+50%
    yB_ = (fmin / c - 1.0 <= -0.30).astype('int8')    # 未来 24h 内出现 <=-30%
    yC_ = (fmax / c - 1.0 >= 0.10).astype('int8')     # 未来 24h 内出现 >=+10%
    idx = np.arange(0, n - WMAX, STEP)
    if len(idx) == 0:
        continue
    Xs.append(np.stack([F[k][idx] for k in FNAMES], axis=1).astype('float32'))
    yA.append(yA_[idx]); yB.append(yB_[idx]); yC.append(yC_[idx])
    tsv.append(ot[idx]); sv.append(np.full(len(idx), si, dtype='int32'))

del df; gc.collect()
X = np.concatenate(Xs); YA = np.concatenate(yA); YB = np.concatenate(yB); YC = np.concatenate(yC)
TS = np.concatenate(tsv); SI = np.concatenate(sv)
ok = ~np.isnan(X).any(axis=1)
X, YA, YB, YC, TS, SI = X[ok], YA[ok], YB[ok], YC[ok], TS[ok], SI[ok]
log(f'样本 {len(X):,}  特征 {X.shape[1]}  正例率 局+50%{100*YA.mean():.3f}% 崩-30%{100*YB.mean():.3f}% 涨+10%{100*YC.mean():.3f}%')

cut = np.quantile(TS, 0.70)
tr = TS < cut; te = ~tr
log(f'训练 {tr.sum():,}  测试 {te.sum():,}  切分 {pd.to_datetime(int(cut), unit="ms")}')

def report(y, p, ts, si, name):
    log(f'--- {name}  测试 {len(y):,}  基准 {100*y.mean():.3f}%  AUC={roc_auc_score(y, p):.4f}')
    log(f'    概率 max={p.max():.3f} p99={np.quantile(p,0.99):.3f} p90={np.quantile(p,0.90):.3f} 中位={np.median(p):.4f}')
    best = None
    for thr in np.arange(0.05, 1.0, 0.01):
        sel = p >= thr; n = int(sel.sum())
        if n == 0:
            continue
        pr = y[sel].mean()
        if pr >= 0.85 and (best is None or n > best[3]):
            best = (thr, pr, y[sel].sum() / max(y.sum(), 1), n)
    if best:
        log(f'  ★ 逐点 85%精确率最低阈值={best[0]:.2f}  报出 {best[3]:,}（覆盖 {100*best[3]/len(y):.3f}%）  精确率 {100*best[1]:.1f}%  召回 {100*best[2]:.1f}%')
    else:
        log('  ✗ 逐点：无阈值达到 85% 精确率')
    for thr in (0.3, 0.5, 0.7, 0.9):
        ev = []
        for c_ in np.unique(si):
            m = np.flatnonzero((p >= thr) & (si == c_))
            if len(m) == 0:
                continue
            o = np.argsort(ts[m]); m = m[o]; t = ts[m]
            brk = np.flatnonzero(np.diff(t) > WMAX * 300000)
            st = np.concatenate([[m[0]], m[brk + 1]]) if len(brk) else m[:1]
            ev.extend(y[st].tolist())
        ev = np.array(ev, dtype='int8')
        log(f'      事件级 thr={thr:.1f}  ' + ('无报出' if len(ev) == 0 else f'局 {len(ev):,}  精确率 {100*ev.mean():.1f}%'))

for tag, y, fn_tag in (('局 +50%/24h', YA, 'up50'), ('崩 -30%/24h', YB, 'dn30'), ('涨 +10%/24h', YC, 'up10')):
    log('=' * 100)
    log(f'【{tag}】训练正例 {int(y[tr].sum()):,} / {int(tr.sum()):,}')
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=42)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    report(y[te], p, TS[te], SI[te], tag)
    fn = os.path.join(MD, 'gw_%s.joblib' % fn_tag)
    joblib.dump({'model': m, 'features': FNAMES, 'wmax': WMAX}, fn)
    log('  模型已存', fn)

log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
