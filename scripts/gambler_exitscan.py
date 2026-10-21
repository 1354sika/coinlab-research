# -*- coding: utf-8 -*-
"""gambler_exitscan.py — 退出规则网格：跟盘 → 何时撤出最赚
用户诉求（2026-10-16）："识别操盘者 → 前期跟盘 → 提前撤出 → 博几倍收益"
                      "搞到就走，不逗留"

做法：
  1) 60 币 5m，标记局（g20 口径）
  2) 训模型（未来 1h 涨>=5%），时间留出
  3) 入场：P >= thr
  4) 退出（网格扫描）：
       A 固定止盈 × 止损 × 超时
       B 移动止损（从最高点回落 x%）
       C 撤退信号（量能萎缩到入场峰值 50% 以下 且 价格自峰回落 >8%）
  5) 输出：期望净收益 / 胜率 / ≥+50% / ≥+100% 占比
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
OUT = os.path.join(EV, 'gambler_exitscan.txt')

GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_exitscan · 退出规则网格（跟盘 → 何时撤出最赚）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['seg_ret', 'up_frac24']

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

        fu = np.full(n, np.nan); fu[:-12] = c[12:] / c[:-12] - 1.0
        ylab = (fu >= 0.05).astype('int8')
        idx_all = np.arange(0, n - 12, 6)
        X = np.stack([F[k][idx_all] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        ia = idx_all[m]; X = X[m]
        RECS.append((sym, X, ylab[ia], ot[ia], ia, c, q, n, inreg))
    del ch; gc.collect()

log(f'币 {len(RECS)}   耗时 {time.time()-t0t:.0f}s')
X = np.concatenate([r[1] for r in RECS]); Y = np.concatenate([r[2] for r in RECS])
T = np.concatenate([r[3] for r in RECS])
cut = np.quantile(T, 0.70); tr = T < cut
m_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, min_samples_leaf=50,
                                    l2_regularization=1.0, random_state=42)
m_.fit(X[tr], Y[tr])
log(f'模型训练完成，训练 {int(tr.sum()):,}  测试 {int((~tr).sum()):,}')

# 预计算测试段所有入场点
ENT = []      # (sym_idx, i_full, close_arr, q_arr, n)
for si, (sym, Xc, yc, otc, iac, c, q, n, inreg) in enumerate(RECS):
    p = m_.predict_proba(Xc)[:, 1]
    for j in np.flatnonzero(otc >= cut):
        if inreg[int(iac[j])]:        # 只看局内
            ENT.append((si, int(iac[j]), c, q, n))
log(f'局内入场点 {len(ENT):,}')

def sim(entries, thr_ps, tp, sl, to, trail=None, retreat=None):
    rs = []
    for (si, ij, c, q, n) in entries:
        px = float(c[ij])
        seg = c[ij:min(n, ij + to)]
        if len(seg) == 0:
            continue
        ret = seg / (px * (1 + SLIP)) - 1.0
        a = np.flatnonzero(ret >= tp); b = np.flatnonzero(ret <= -sl)
        # 移动止损
        if trail is not None:
            peak = np.maximum.accumulate(seg) / px - 1.0
            dd = seg / np.maximum(np.maximum.accumulate(seg), 1e-12) - 1.0
            tb = np.flatnonzero(peak >= trail['arm'])
            if len(tb):
                first = tb[0]
                hit = np.flatnonzero(dd[first:] <= -trail['drop'])
                tb2 = first + hit[0] if len(hit) else None
            else:
                tb2 = None
        else:
            tb2 = None
        # 撤退信号：量能萎缩到峰值 50% 以下 且 价格自峰回落 >8%
        if retreat is not None:
            qs = q[ij:min(n, ij + to)]
            peakq = np.maximum.accumulate(qs)
            peakc = np.maximum.accumulate(seg)
            cond = (qs < 0.5 * peakq) & (seg / np.maximum(peakc, 1e-12) - 1.0 <= -0.08)
            rb = np.flatnonzero(cond)
            rb = rb[0] if len(rb) else None
        else:
            rb = None
        cand = []
        if len(a): cand.append((a[0], tp))
        if len(b): cand.append((b[0], -sl))
        if tb2 is not None: cand.append((tb2, float(seg[tb2]) / px - 1.0))
        if rb is not None: cand.append((rb, float(seg[rb]) / px - 1.0))
        if cand:
            cand.sort()
            g = cand[0][1]
        else:
            g = ret[-1]
        rs.append(g - FEE - 2 * SLIP)
    return np.array(rs) if rs else np.array([])

log('')
log('=' * 100)
log('【A】固定止盈 × 止损 × 超时（局内入场，扣 %.2f%%+%.2f%%）' % (100*FEE, 100*SLIP))
log('=' * 100)
log(f'{"TP":>7s} {"SL":>7s} {"TO":>5s} | {"笔数":>8s} {"期望净%":>9s} {"中位%":>8s} {"胜率%":>7s} {"≥+50%":>7s} {"≥+100%":>7s}')
best = None
for tp in (0.10, 0.20, 0.50, 1.00, 2.00):
    for sl in (0.05, 0.10, 0.15, 0.25):
        for to in (72, 144, 288, 576):     # 6h / 12h / 24h / 48h
            A = sim(ENT, None, tp, sl, to)
            if len(A) == 0:
                continue
            row = (tp, sl, to, len(A), 100*A.mean(), 100*np.median(A), 100*(A>0).mean(),
                   100*(A>=0.5).mean(), 100*(A>=1.0).mean())
            log(f'{tp:7.2f} {sl:7.2f} {to:5d} | {row[3]:8,d} {row[4]:+9.3f} {row[5]:+8.3f} {row[6]:7.1f} {row[7]:7.1f} {row[8]:7.1f}')
            if best is None or row[4] > best[4]:
                best = row
if best:
    log(f'★ 最优：TP={best[0]:.2f} SL={best[1]:.2f} TO={best[2]}  期望净 {best[4]:+.3f}%/笔  笔 {best[3]:,}')

log('')
log('=' * 100)
log('【B】移动止损变体（在最优 TP/SL/TO 基础上）')
log('=' * 100)
if best:
    tp, sl, to = best[0], best[1], best[2]
    for arm, drop in ((0.10, 0.05), (0.20, 0.08), (0.50, 0.15), (1.00, 0.25)):
        A = sim(ENT, None, tp, sl, to, trail={'arm': arm, 'drop': drop})
        if len(A):
            log(f'  触发 {arm:.0%} 后回落 {drop:.0%} 离场：笔 {len(A):,}  期望净 {100*A.mean():+.3f}%  '
                f'胜率 {100*(A>0).mean():.1f}%  ≥+50% {100*(A>=0.5).mean():.1f}%')
    log('')
    log('【C】撤退信号（量能萎缩到峰值 50% 以下 且 自峰回落 >8%）')
    A = sim(ENT, None, tp, sl, to, retreat=True)
    if len(A):
        log(f'  笔 {len(A):,}  期望净 {100*A.mean():+.3f}%  胜率 {100*(A>0).mean():.1f}%  '
            f'≥+50% {100*(A>=0.5).mean():.1f}%  ≥+100% {100*(A>=1.0).mean():.1f}%')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
