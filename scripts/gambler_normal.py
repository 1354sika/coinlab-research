# -*- coding: utf-8 -*-
"""gambler_normal.py — 局 vs 正常波动：两种机制分开建模，看谁可预测
用户设想（2026-10-16）：币市波动 = 局 + 正常波动；正常波动是散户之间与币市的行为概率心理博弈。
做法：
  1) 用 g20 口径标出"局内"时刻（局启动后到崩塌前）
  2) 样本分两组：局内 / 局外（正常波动）
  3) 两组各训一个模型（同一套特征，预测"未来 1h 涨>=3%"），对比：
       A AUC（可预测性）
       B 按规则模拟的净收益（判对就是有，判错不下注）
  目的：回答"正常波动到底是不是可建模的博弈，还是噪声"
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_normal.txt')
GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005
TP, SL, TIMEOUT = 0.10, -0.05, 24      # 小目标：正常波动的交易规则

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_normal · 局 vs 正常波动（两种机制分开建模）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['seg_ret', 'up_frac24', 'hour_sin', 'hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

IN, OUT_ = [], []
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
        # 标记局内
        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        inreg = np.zeros(n, dtype=bool); last = -10 ** 9
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
            inreg[t0:min(n, i1 + 96)] = True
            last = i1
        # 特征
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
        hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
        F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)
        # 标签：未来 1h 涨>=3%
        fu = np.full(n, np.nan); fu[:-12] = c[12:] / c[:-12] - 1.0
        ylab = (fu >= 0.03).astype('int8')
        idx_all = np.arange(0, n - 12, 6)
        X = np.stack([F[k][idx_all] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        ia = idx_all[m]; X = X[m]
        reg = inreg[ia]
        yi = ylab[ia]
        # 局内样本全取；局外每 4 个取 1
        keep_in = np.flatnonzero(reg)
        keep_out = np.flatnonzero(~reg)[::4]
        if len(keep_in):
            IN.append((sym, X[keep_in], yi[keep_in], ot[ia[keep_in]], ia[keep_in], c, n))
        if len(keep_out):
            OUT_.append((sym, X[keep_out], yi[keep_out], ot[ia[keep_out]], ia[keep_out], c, n))
    del ch; gc.collect()

def build(L):
    if not L:
        return None
    X = np.concatenate([x[1] for x in L]); Y = np.concatenate([x[2] for x in L])
    T = np.concatenate([x[3] for x in L])
    return X, Y, T

log(f'局内样本组 {len(IN)} 币  局外样本组 {len(OUT_)} 币  耗时 {time.time()-t0t:.0f}s')

def run(L, name):
    B = build(L)
    if B is None or len(B[0]) < 10000:
        log(f'  {name}: 样本不足'); return
    X, Y, T = B
    cut = np.quantile(T, 0.70); tr = T < cut
    if Y[tr].sum() < 100 or Y[~tr].sum() < 50:
        log(f'  {name}: 正例不足（训练 {int(Y[tr].sum())} 测试 {int(Y[~tr].sum())}）'); return
    m_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[~tr])[:, 1]
    auc = roc_auc_score(Y[~tr], p)
    log(f'  {name}: 样本 {len(X):,}  正例率 {100*Y.mean():.2f}%  AUC={auc:.4f}')
    for thr in (0.3, 0.5, 0.7):
        sel = p >= thr; nn = int(sel.sum())
        if nn == 0:
            log(f'    thr={thr}: 无报出'); continue
        log(f'    thr={thr:.1f}  报出 {nn:,}  精确率 {100*Y[~tr][sel].mean():.1f}%  召回 {100*Y[~tr][sel].sum()/max(Y[~tr].sum(),1):.1f}%')

log('')
log('=' * 100)
log('【A】两组可预测性对照 (标签=未来 1h 涨>=3%，时间留出)')
log('=' * 100)
run(IN, '局内（操纵段）')
run(OUT_, '局外（正常波动）')

log('')
log('=' * 100)
log('【B】交易模拟（报出即进场，+10% 止盈 / -5% 止损 / 2h 超时，扣 0.04%+0.05%）')
log('=' * 100)
for L, name in ((IN, '局内'), (OUT_, '局外')):
    B = build(L)
    if B is None:
        continue
    X, Y, T = B
    cut = np.quantile(T, 0.70); tr = T < cut
    if Y[tr].sum() < 100:
        continue
    m_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    log(f'  {name}:')
    for thr in (0.3, 0.5, 0.7):
        rs = []
        for (sym, Xc, yc, otc, iac, c, n) in L:
            p = m_.predict_proba(Xc)[:, 1]
            for j in np.flatnonzero((p >= thr) & (otc >= cut)):
                ij = int(iac[j]); px = float(c[ij])
                seg = c[ij:min(n, ij + TIMEOUT)]
                if len(seg) == 0:
                    continue
                ret = seg / (px * (1 + SLIP)) - 1.0
                a = np.flatnonzero(ret >= TP); b = np.flatnonzero(ret <= SL)
                if len(a) and (not len(b) or a[0] <= b[0]):
                    g = TP
                elif len(b):
                    g = SL
                else:
                    g = ret[-1]
                rs.append(g - FEE - 2 * SLIP)
        if not rs:
            log(f'    thr={thr}: 无交易'); continue
        A = np.array(rs)
        log(f'    thr={thr:.1f}  笔 {len(A):,}  期望净 {100*A.mean():+.3f}%  中位 {100*np.median(A):+.3f}%  胜率 {100*(A>0).mean():.1f}%')

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
