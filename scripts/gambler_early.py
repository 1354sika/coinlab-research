# -*- coding: utf-8 -*-
"""gambler_early.py — 补上核心指标：事件级"<=20% 召回"
口径（用户 2026-10-16 定）：
  · 局 = 局部低点起、24h 内涨>=50%、随后回吐涨幅 50%（half 口径，703 局/303币）
  · 早期识别 = 模型在"该局涨幅还 <=20%"的阶段就报出
输出（这才是判断"早期识别能不能用"的唯一指标）：
  A 事件级召回：<=5% / <=10% / <=20% 各自被报出的局 / 全部局
  B 报出时已涨幅度分布（中位/四分位）
  C 报出后剩余空间（峰值/报出价-1）
"""
import os, gc, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view
import joblib

GAIN = 0.50
WMAX = 288
t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_early.txt')

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_early · 事件级"<=20% 召回"（局=+50%且回吐涨幅50%，303 币全量）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 144, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['since_hi24', 'since_lo24', 'run_up', 'run_dn', 'seg_ret', 'seg_volr', 'up_frac24']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

# 逐币：算特征 + 局 + 标签；保留 (币, 时间, 特征, 局列表)
COINS = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000:
            continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64'); ot = g['open_time'].values
        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        lab = np.zeros(n, dtype='int8'); regs = []; last = -10 ** 9
        for t0 in cand:
            if t0 - last < WMAX:
                continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            if H / base - 1.0 < GAIN:
                continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            thr = base + (H - base) * 0.5
            k2 = min(n - 1, i1 + 96)
            if not (k2 > i1 and l[i1:k2 + 1].min() <= thr):
                continue
            regs.append((int(ot[t0]), int(i1), float(base), float(H)))
            seg = slice(t0, min(n, t0 + 96))
            gsg = c[seg] / base - 1.0
            lab[seg] = ((gsg >= 0.0) & (gsg <= 0.20)).astype('int8')
            last = i1
        F = {}
        for k in (6, 12, 24, 48, 96, 144, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2, k // 4)).max().values / c - 1.0
            F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2, k // 4)).min().values / c - 1.0
        Wl = sliding_window_view(l, WMAX); Wh = sliding_window_view(h, WMAX)
        sl = np.full(n, np.nan); sh = np.full(n, np.nan)
        sl[WMAX - 1:] = WMAX - 1 - Wl.argmin(axis=1); sh[WMAX - 1:] = WMAX - 1 - Wh.argmax(axis=1)
        F['since_lo24'] = sl; F['since_hi24'] = sh
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        sg = np.sign(np.nan_to_num(r1, nan=0.0))
        ru = np.zeros(n); rd = np.zeros(n); cu = cd = 0
        for i in range(n):
            cu = cu + 1 if sg[i] > 0 else 0; cd = cd + 1 if sg[i] < 0 else 0
            ru[i] = cu; rd[i] = cd
        F['run_up'] = ru; F['run_dn'] = rd
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        qm = rmean(q, WMAX); F['seg_volr'] = np.where(qm > 0, q / np.maximum(qm, 1e-12), np.nan)
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)
        pos = np.flatnonzero(lab == 1)[::3]
        neg_all = np.flatnonzero(lab == 0); neg = neg_all[::24]
        if len(neg) > 3000:
            neg = neg[::max(1, len(neg) // 3000)]
        idx = np.concatenate([pos, neg]) if len(pos) else neg
        if len(idx) == 0:
            continue
        X = np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        COINS.append(dict(sym=sym, idx=idx[m], X=X[m], y=lab[idx[m]], ot=ot[idx[m]], n=n,
                          regs=regs, c=c, otfull=ot))
    del ch; gc.collect()

log(f'币 {len(COINS)}  局 {sum(len(c["regs"]) for c in COINS)}  耗时 {time.time()-t0t:.0f}s')

X = np.concatenate([c['X'] for c in COINS]); Y = np.concatenate([c['y'] for c in COINS])
TS = np.concatenate([c['ot'] for c in COINS])
cut = np.quantile(TS, 0.70)
tr = TS < cut
log(f'样本 {len(X):,}  正例 {int(Y.sum()):,}  训练 {int(tr.sum()):,}')
m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=30,
                                    l2_regularization=1.0, random_state=42)
m_.fit(X[tr], Y[tr])
joblib.dump({'model': m_, 'features': FEAT}, os.path.join(MD, 'ge_early.joblib'))
log('模型已存 ge_early.joblib')

# ---- 事件级评估：对每个局，找 t0 之后首次 p>=thr 的时刻 ----
log('')
log('=' * 100)
log('【A】事件级：局在什么涨幅时被首次报出（测试段）')
log('=' * 100)
for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
    hits = {0.05: 0, 0.10: 0, 0.20: 0, 0.50: 0, 'miss': 0}
    first_gains = []; remain = []
    tot = 0
    for c in COINS:
        p = m_.predict_proba(c['X'])[:, 1]
        for (ot0, i1, base, H) in c['regs']:
            if ot0 < cut:
                continue
            tot += 1
            msk = (c['ot'] >= ot0) & (c['ot'] <= c['otfull'][min(i1, c['n'] - 1)]) & (p >= thr)
            if not msk.any():
                hits['miss'] += 1; continue
            j = int(np.flatnonzero(msk)[0])
            px = float(c['c'][np.flatnonzero(c['otfull'] == c['ot'][j])[0]])
            g = px / base - 1.0
            first_gains.append(g); remain.append(H / px - 1.0)
            for k in (0.05, 0.10, 0.20, 0.50):
                if g <= k:
                    hits[k] += 1
    if tot == 0:
        log(f'  thr={thr}: 测试段无局'); continue
    fg = np.array(first_gains) if first_gains else np.array([np.nan])
    rm = np.array(remain) if remain else np.array([np.nan])
    log(f'  thr={thr:.1f}  测试局 {tot}')
    log(f'     <=5% 报出 {hits[0.05]} ({100*hits[0.05]/tot:.1f}%)   <=10% {hits[0.10]} ({100*hits[0.10]/tot:.1f}%)   '
        f'<=20% {hits[0.20]} ({100*hits[0.20]/tot:.1f}%)   <=50% {hits[0.50]} ({100*hits[0.50]/tot:.1f}%)   漏掉 {hits["miss"]} ({100*hits["miss"]/tot:.1f}%)')
    log(f'     报出时已涨幅度 中位 {100*np.nanmedian(fg):.1f}%  p25 {100*np.nanquantile(fg,0.25):.1f}%  p75 {100*np.nanquantile(fg,0.75):.1f}%')
    log(f'     报出后剩余空间 中位 {100*np.nanmedian(rm):.1f}%')

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
