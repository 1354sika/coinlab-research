# -*- coding: utf-8 -*-
"""gambler_early2.py — 早期识别：加散户情绪/级联特征，能否把覆盖率从 30.6% 抬起来
用户要求（2026-10-16）："早期识别这是重中之重" / "成果识别局的覆盖率起码得 90%"

现状：gambler_early 只覆盖 30.6% 的局（涨<=20% 时报出）。
本脚本：把 gambler_crowd 验证过的散户情绪指标（panic/trade_intensity/vol_intensity）
        与级联（同刻异动币数）加进特征，看覆盖率能否提升。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_early2.txt')

GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_early2 · 早期识别：基础 vs +散户情绪+级联')
log('=' * 100)

BASE = []
for k in (6, 12, 24, 48, 96, 288):
    BASE += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
BASE += ['seg_ret', 'up_frac24']
EXT = ['panic', 'trade_intensity', 'vol_intensity', 'fomo', 'vol_price_div']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

RECS = []
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
        regs = []; last = -10 ** 9
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
            regs.append((t0, i1))
            last = i1
        if not regs:
            continue

        F = {}
        for k in (6, 12, 24, 48, 96, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2, k // 4)).max().values / c - 1.0
            F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2, k // 4)).min().values / c - 1.0
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)
        # 散户情绪（24 根 = 2h 窗口）
        v24 = rmean(q, 24); t24 = rmean(tr, 24)
        vi = np.where(v24 > 0, q / np.maximum(v24, 1e-12), 1.0)
        ti = np.where(t24 > 0, tr / np.maximum(t24, 1e-12), 1.0)
        F['trade_intensity'] = ti
        F['vol_intensity'] = vi
        F['fomo'] = np.clip(r1, 0, None) * vi
        F['panic'] = np.clip(-r1, 0, None) * vi
        F['vol_price_div'] = vi / (1.0 + 100 * np.abs(r1))

        # 标签：未来 12 根（1h）内最高价相对当前涨 >= 5%
        fh = np.full(n, np.nan)
        for j in range(n - 12):
            fh[j] = h[j:j + 12].max()
        ylab = ((fh / c - 1.0) >= 0.05).astype('int8')
        idx_all = np.arange(0, n - 12, 6)
        Xb = np.stack([F[k][idx_all] for k in BASE], axis=1).astype('float32')
        Xe = np.stack([F[k][idx_all] for k in EXT], axis=1).astype('float32')
        m = ~np.isnan(Xb).any(axis=1) & ~np.isnan(Xe).any(axis=1)
        ia = idx_all[m]
        RECS.append((sym, Xb[m], Xe[m], ylab[ia], ot[ia], ia, c, h, n, regs))
    del ch; gc.collect()

log(f'币 {len(RECS)}   耗时 {time.time()-t0t:.0f}s')

def evaluate(use_ext):
    nreg = 0; cov = 0; e5 = 0; e10 = 0; e20 = 0
    Xb = np.concatenate([r[1] for r in RECS])
    Xe = np.concatenate([r[2] for r in RECS])
    Y = np.concatenate([r[3] for r in RECS])
    T = np.concatenate([r[4] for r in RECS])
    X = np.hstack([Xb, Xe]) if use_ext else Xb
    cut = np.quantile(T, 0.70); tr = T < cut
    m_ = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    for (sym, Xbc, Xec, yc, otc, iac, c, h, n, regs) in RECS:
        Xc = np.hstack([Xbc, Xec]) if use_ext else Xbc
        pc = m_.predict_proba(Xc)[:, 1]
        for (t0, i1) in regs:
            j = np.flatnonzero(iac >= t0)
            if len(j) == 0:
                continue
            j0 = j[0]
            if otc[j0] < cut:
                continue
            nreg += 1
            # 在 [t0, i1] 内找第一个报出点
            seg = pc[j0:]
            kk = np.flatnonzero(seg >= 0.5)
            if len(kk) == 0:
                continue
            ij = int(iac[j0 + kk[0]])
            if ij > i1:
                continue
            gain = c[ij] / max(c[t0], 1e-12) - 1.0
            cov += 1
            if gain <= 0.05: e5 += 1
            if gain <= 0.10: e10 += 1
            if gain <= 0.20: e20 += 1
    return nreg, cov, e5, e10, e20

log('')
log('=' * 100)
log('【对照】阈值 P>=0.5，测试段局，报出时已涨幅度分档')
log('=' * 100)
for use_ext, tag in ((False, 'A 基础特征'), (True, 'B 基础+散户情绪+级联')):
    nreg, cov, e5, e10, e20 = evaluate(use_ext)
    log(f'  {tag:<24s} 测试局 {nreg:5d}   报出 {cov:5d}（覆盖 {100*cov/max(nreg,1):5.1f}%）'
        f'   其中 <=5% {e5}（{100*e5/max(nreg,1):4.1f}%）  <=10% {e10}（{100*e10/max(nreg,1):4.1f}%）'
        f'  <=20% {e20}（{100*e20/max(nreg,1):4.1f}%）')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
