# -*- coding: utf-8 -*-
"""gambler_small.py — 小目标策略：+10% / +20% / +30% / +50% × 超时网格
用户问题（2026-10-16）："如果收益率只有 0.3，比如我投一块钱，赚 1.3 块钱呢？"
                      （即 +30%，不是 +50% / +200%）

上一版 gambler_freq 跑了 1.85 小时（Python 循环 + 循环内字符串格式化）。
本版全部向量化：把每个入场点的未来价格堆成矩阵 (N, MAX_HOLD)，一次性比较。

回答：
  1) 小目标的命中率、命中所需时间
  2) 加"超时退出"后，资金占用天数降到多少
  3) 各种 (目标, 超时) 组合的每笔期望与年化
"""
import os, gc, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_small.txt')

GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005
BPD = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_small · 小目标策略（+10%~+50%）× 超时网格')
log('=' * 100)

ENT = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'}, chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000:
            continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); ot = g['open_time'].values

        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        last = -10 ** 9
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
            last = i1
            for j in range(t0, min(n, t0 + WMAX)):
                if c[j] / max(c[t0], 1e-12) - 1.0 <= 0.05:
                    ENT.append((sym, int(j), c, h, n))
                    break
    del ch; gc.collect()

NENT = len(ENT)
log(f'入场点 {NENT:,}   耗时 {time.time()-t0t:.0f}s')

MAXH = 8640
H = np.full((NENT, MAXH), np.nan, dtype='float64')
C = np.full((NENT, MAXH), np.nan, dtype='float64')
PX = np.zeros(NENT)
for i, (sym, j, c, h, n) in enumerate(ENT):
    k = min(MAXH, n - j)
    if k <= 0:
        continue
    H[i, :k] = h[j:j + k]
    C[i, :k] = c[j:j + k]
    PX[i] = c[j]
log(f'价格矩阵 {H.shape}   耗时 {time.time()-t0t:.0f}s')

ok = PX > 0
H = H[ok]; C = C[ok]; PX = PX[ok]
GR = H / PX[:, None] - 1.0          # 相对入场价的高点涨幅
CR = C / PX[:, None] - 1.0          # 相对入场价的收盘涨幅

log('')
log('=' * 100)
log('【A】小目标：命中率 + 命中所需天数（无超时，上限 30 天）')
log('=' * 100)
log(f'{"目标":>7s} | {"命中率%":>8s} {"命中中位天":>10s} {"命中均值天":>10s} | {"未命中到期中位%":>14s} | {"每笔期望净%":>11s}')
for tgt in (0.10, 0.20, 0.30, 0.50, 1.00, 1.30):
    reach = GR >= tgt
    has = np.nan_to_num(reach).any(axis=1)
    arg = np.where(has, np.nan_to_num(reach).argmax(axis=1), MAXH)
    # 未命中：第 30 天收盘
    lastc = np.nan_to_num(CR, nan=0.0)[:, -1]
    ret = np.where(has, tgt, lastc) - FEE - 2 * SLIP
    hit_days = arg[has] / BPD
    miss_med = np.median(lastc[~has]) if (~has).any() else float('nan')
    log(f'{tgt:7.0%} | {100*has.mean():8.1f} {np.median(hit_days) if has.any() else float("nan"):10.2f} '
        f'{hit_days.mean() if has.any() else float("nan"):10.2f} | '
        f'{100*miss_med:14.2f} | {100*ret.mean():11.2f}')

log('')
log('=' * 100)
log('【B】(目标 × 超时) 网格：超时到点就走（拿收盘价）')
log('=' * 100)
TOS = [(48, '4h'), (96, '8h'), (288, '1天'), (576, '2天'), (864, '3天'), (1440, '5天'), (2880, '10天')]
hdr = f'{"目标":>7s} | ' + ' | '.join(f'{tag:>18s}' for _, tag in TOS)
log(hdr)
log(f'{"":>7s} | ' + ' | '.join(f'{"命中% 期望% 占用天":>18s}' for _ in TOS))
best = None
for tgt in (0.10, 0.20, 0.30, 0.50, 1.00, 1.30):
    cells = []
    for to, tag in TOS:
        sub = GR[:, :to]
        reach = np.nan_to_num(sub) >= tgt
        has = reach.any(axis=1)
        arg = np.where(has, reach.argmax(axis=1), to - 1)
        lastc = np.nan_to_num(CR[:, :to], nan=0.0)[:, -1]
        ret = np.where(has, tgt, lastc) - FEE - 2 * SLIP
        days = (arg + 1) / BPD
        cells.append((tag, 100*has.mean(), 100*ret.mean(), days.mean(), ret.mean()))
        score = ret.mean() / max(days.mean(), 1e-6) * 365
        if best is None or score > best[0]:
            best = (score, tgt, tag, 100*has.mean(), 100*ret.mean(), days.mean())
    log(f'{tgt:7.0%} | ' + ' | '.join(f'{c[1]:5.1f} {c[2]:+6.2f} {c[3]:5.1f}' for c in cells))

log('')
log('=' * 100)
log('【C】年化最高的 (目标 × 超时) —— 按"每笔期望 ÷ 占用天数"排序')
log('=' * 100)
rows = []
for tgt in (0.10, 0.20, 0.30, 0.50, 1.00, 1.30):
    for to, tag in TOS:
        reach = np.nan_to_num(GR[:, :to]) >= tgt
        has = reach.any(axis=1)
        arg = np.where(has, reach.argmax(axis=1), to - 1)
        lastc = np.nan_to_num(CR[:, :to], nan=0.0)[:, -1]
        ret = np.where(has, tgt, lastc) - FEE - 2 * SLIP
        days = (arg + 1) / BPD
        rows.append((tgt, tag, 100*has.mean(), 100*ret.mean(), days.mean(),
                     ret.mean() / max(days.mean(), 1e-6) * 365 * 100))
rows.sort(key=lambda r: -r[5])
log(f'{"目标":>7s} {"超时":>5s} | {"命中%":>7s} {"每笔期望%":>9s} {"占用天":>7s} {"年化%":>10s}')
for tgt, tag, hit, exp_, days, ann in rows[:15]:
    log(f'{tgt:7.0%} {tag:>5s} | {hit:7.1f} {exp_:+9.2f} {days:7.2f} {ann:10.1f}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
