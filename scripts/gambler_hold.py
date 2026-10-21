# -*- coding: utf-8 -*-
"""gambler_hold.py — 持有时间扫描：给足时间，到底能不能翻几倍？
用户质疑（2026-10-16）："翻几倍的情况肯定存在，要么就是没发现，要么时间跨度不够，要么持有时间不够"
                      "情绪特征一定是影响市场的"

此前 exitscan 的最大持有只到 48h（576 根），且固定止损 5~25% 会在拉升前被震出。
本脚本回答：如果**给足时间（到 30 天）**并**放宽止损**，翻倍/翻数倍的比例是多少？

做法：
  1) 60 币 5m，标记局；入场 = 局启动 + 已涨 <= 5%（早期）
  2) 对每个入场点，扫持有上限 H ∈ {1d,2d,3d,5d,10d,15d,30d}
        A 不设止损：只看到期收益 + 期间最大收益
        B 移动止损：从最高点回落 x%（10/20/30/50%）才走
        统计：到期收益中位 / 期间最大收益中位 / ≥+50% / ≥+100% / ≥+200% / ≥+500% 占比
  3) 输出：持有时间是"不够"还是"够"了
"""
import os, gc, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_hold.txt')

GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005
HOLDS = [(288, '1天'), (576, '2天'), (864, '3天'), (1440, '5天'),
         (2880, '10天'), (4320, '15天'), (8640, '30天')]

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_hold · 持有时间扫描：给足时间能不能翻几倍？')
log('=' * 100)
log(f'  入场：局启动后、已涨 <= 5% 的早期点（无前视）')
log(f'  扣费 {FEE:.2%}+{SLIP:.2%}')

ENT = []   # (sym, ij, c, h, l, n)
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
            # 入场：局启动之后，第一根 已涨<=5% 的点
            for j in range(t0, min(n, t0 + WMAX)):
                if c[j] / max(c[t0], 1e-12) - 1.0 <= 0.05:
                    ENT.append((sym, int(j), c, h, l, n))
                    break
    del ch; gc.collect()

log(f'入场点 {len(ENT):,}   耗时 {time.time()-t0t:.0f}s')

log('')
log('=' * 100)
log('【A】不设止损：只看到期收益 + 期间最大涨幅')
log('=' * 100)
log(f'{"持有":>6s} | {"到期中位%":>10s} {"到期均值%":>10s} | {"期间最高中位%":>13s} {"期间最高均值%":>13s} | '
    f'{"≥+50%":>7s} {"≥+100%":>8s} {"≥+200%":>8s} {"≥+500%":>8s}')
for H, tag in HOLDS:
    end, peak = [], []
    for (sym, j, c, h, l, n) in ENT:
        j2 = min(n - 1, j + H)
        if j2 <= j:
            continue
        px = c[j]
        end.append(c[j2] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP)
        peak.append(h[j:j2 + 1].max() / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP)
    if not end:
        log(f'{tag:>6s} | 无样本'); continue
    E = np.array(end); P = np.array(peak)
    log(f'{tag:>6s} | {100*np.median(E):10.2f} {100*E.mean():10.2f} | '
        f'{100*np.median(P):13.2f} {100*P.mean():13.2f} | '
        f'{100*(P>=0.5).mean():7.1f} {100*(P>=1.0).mean():8.1f} {100*(P>=2.0).mean():8.1f} {100*(P>=5.0).mean():8.1f}')

log('')
log('=' * 100)
log('【B】移动止损（从最高点回落 x% 就走），持有上限 30 天')
log('=' * 100)
H2 = 8640
log(f'{"回落":>6s} | {"笔数":>8s} {"期望净%":>10s} {"中位%":>9s} {"胜率%":>8s} {"≥+50%":>7s} {"≥+100%":>8s} {"≥+200%":>8s}')
for drop in (0.10, 0.20, 0.30, 0.50, 0.80):
    rs = []
    for (sym, j, c, h, l, n) in ENT:
        j2 = min(n, j + H2)
        if j2 <= j:
            continue
        px = c[j]
        seg = c[j:j2]
        peakc = np.maximum.accumulate(seg)
        dd = seg / np.maximum(peakc, 1e-12) - 1.0
        hit = np.flatnonzero(dd <= -drop)
        if len(hit):
            g = seg[hit[0]] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP
        else:
            g = seg[-1] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP
        rs.append(g)
    if not rs:
        log(f'{drop:6.0%} | 无样本'); continue
    A = np.array(rs)
    log(f'{drop:6.0%} | {len(A):8,d} {100*A.mean():+10.2f} {100*np.median(A):+9.2f} '
        f'{100*(A>0).mean():8.1f} {100*(A>=0.5).mean():7.1f} {100*(A>=1.0).mean():8.1f} {100*(A>=2.0).mean():8.1f}')

log('')
log('=' * 100)
log('【C】对照：不设止损、不设持有上限（一直拿到数据结束）—— 上界')
log('=' * 100)
rs = []
for (sym, j, c, h, l, n) in ENT:
    px = c[j]
    seg = c[j:]
    if len(seg) < 2:
        continue
    rs.append(seg[-1] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP)
A = np.array(rs)
log(f'  笔 {len(A):,}  期望净 {100*A.mean():+.2f}%  中位 {100*np.median(A):+.2f}%  '
    f'胜率 {100*(A>0).mean():.1f}%  ≥+100% {100*(A>=1.0).mean():.1f}%  ≥+200% {100*(A>=2.0).mean():.1f}%')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
