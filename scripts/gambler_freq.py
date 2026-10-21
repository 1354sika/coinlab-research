# -*- coding: utf-8 -*-
"""gambler_freq.py — 机会频率 vs 单笔幅度：每天有几次？该博几倍？
用户判断（2026-10-16）："翻二分之一和一倍就够了。因为这样的机会如果每天都有，那我接受，
                       每一次只能翻二分之一或一倍。但是如果机会没这么多？
                       那我们必须博几倍的，去增加持窗"

做法（全部向量化，不用逐根循环）：
  1) 60 币 3 年 5m，标记局，取"局启动后已涨<=5%"的早期入场点
  2) 统计**机会频率**：有局的天数占比、每天/每周局数
  3) 对每个入场点，测三种目标（无固定止损，靠自然到达）：
       A 目标 +50%     B 目标 +100%    C 目标 +200%
     统计：命中率、命中所需天数（中位/均值）、未命中到期末的收益
  4) 算**资金年化**：期望每笔 × 每年笔数（含资金占用天数）
"""
import os, gc, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_freq.txt')

GAIN = 0.20
WMAX = 288
FEE, SLIP = 0.0004, 0.0005
BAR_PER_DAY = 288
MAX_HOLD = 8640          # 30 天

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_freq · 机会频率 vs 单笔幅度（该博 50% 还是几倍？）')
log('=' * 100)

ENT = []          # (sym, j, c, h, l, n, t0_day)
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
                    ENT.append((sym, int(j), c, h, l, n, int(ot[t0] // 86400000)))
                    break
    del ch; gc.collect()

log(f'入场点（局内、已涨<=5%）{len(ENT):,}   耗时 {time.time()-t0t:.0f}s')

all_days = sorted({e[6] for e in ENT})
day_count = {}
for e in ENT:
    day_count[e[6]] = day_count.get(e[6], 0) + 1

log('')
log('=' * 100)
log('【A】机会频率（60 币口径；全市场 304 币会更多）')
log('=' * 100)
span_days = (all_days[-1] - all_days[0]) + 1
log(f'  时间跨度 {span_days:,} 天')
log(f'  有入场机会的天数 {len(all_days):,}（占 {100*len(all_days)/span_days:.1f}%）')
log(f'  平均每天入场点 {len(ENT)/span_days:.2f} 个')
cnt = np.array(list(day_count.values()))
log(f'  在有局的天里：中位 {np.median(cnt):.0f} 个/天  p90 {np.percentile(cnt,90):.0f}  max {cnt.max()}')
log(f'  ⇒ 关键：{100-100*len(all_days)/span_days:.1f}% 的天数**没有任何入场机会**')

log('')
log('=' * 100)
log('【B】三种目标：命中率 + 命中所需天数（无固定止损，最长持有 30 天）')
log('=' * 100)
log(f'{"目标":>8s} | {"命中率%":>8s} {"命中天数中位":>12s} {"命中天数均值":>12s} | '
    f'{"未命中到期中位%":>15s} | {"综合期望净%":>12s} {"日均占用":>9s}')
for tgt in (0.50, 1.00, 2.00, 3.00):
    hit_days = []; miss_ret = []; all_ret = []; hold_days = []
    for (sym, j, c, h, l, n, d0) in ENT:
        j2 = min(n, j + MAX_HOLD)
        if j2 <= j + 1:
            continue
        px = c[j]
        seg_h = h[j:j2]
        seg_c = c[j:j2]
        reach = np.flatnonzero(seg_h / px - 1.0 >= tgt)
        if len(reach):
            k = int(reach[0])
            hit_days.append(k / BAR_PER_DAY)
            hold_days.append(k)
            all_ret.append(tgt - FEE - 2 * SLIP)
        else:
            k = len(seg_c) - 1
            r = seg_c[k] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP
            miss_ret.append(r)
            hold_days.append(k)
            all_ret.append(r)
    nh = len(hit_days); nm = len(miss_ret)
    tot = nh + nm
    if tot == 0:
        log(f'{tgt:8.0%} | 无样本'); continue
    A = np.array(all_ret); HD = np.array(hold_days)
    log(f'{tgt:8.0%} | {100*nh/tot:8.1f} {np.median(hit_days) if hit_days else float("nan"):12.1f} '
        f'{np.mean(hit_days) if hit_days else float("nan"):12.1f} | '
        f'{100*np.median(miss_ret) if miss_ret else float("nan"):15.2f} | '
        f'{100*A.mean():12.2f} {HD.mean()/BAR_PER_DAY:9.2f}')

log('')
log('=' * 100)
log('【C】年化估算（假设单笔独立、不并行；年化 = 每笔期望 × 每年可做笔数）')
log('=' * 100)
log(f'  每年天数 365；每笔占用天数 = 命中/未命中的平均持有天数')
for tgt in (0.50, 1.00, 2.00):
    rets = []; holds = []
    for (sym, j, c, h, l, n, d0) in ENT:
        j2 = min(n, j + MAX_HOLD)
        if j2 <= j + 1:
            continue
        px = c[j]
        reach = np.flatnonzero(h[j:j2] / px - 1.0 >= tgt)
        if len(reach):
            k = int(reach[0]); rets.append(tgt - FEE - 2 * SLIP); holds.append(k)
        else:
            k = len(c[j:j2]) - 1
            rets.append(c[j + k] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP); holds.append(k)
    A = np.array(rets); Hd = np.array(holds) / BAR_PER_DAY
    if len(A) == 0:
        continue
    per_trade = A.mean()
    days_per_trade = Hd.mean()
    trades_per_year = 365.0 / max(days_per_trade, 1e-6)
    ann = (1 + per_trade) ** trades_per_year - 1
    log(f'  目标 {tgt:.0%}：每笔期望 {100*per_trade:+.2f}%  平均占用 {days_per_trade:.1f} 天  '
        f'每年可做 {trades_per_year:.1f} 笔  ⇒ 年化 {(ann*100 if abs(ann)<1e3 else float("inf")):.1f}%'
        f'{"（>10 倍，不可信）" if ann>10 else ""}')

log('')
log('=' * 100)
log('【D】移动止损版：回落 20% / 30% / 50% 就走（上限 30 天）')
log('=' * 100)
log(f'{"回落":>6s} | {"笔数":>7s} {"期望净%":>10s} {"中位%":>9s} {"胜率%":>7s} {"平均占用天":>10s} '
    f'{"≥+50%":>7s} {"≥+100%":>8s} {"≥+200%":>8s}')
for drop in (0.20, 0.30, 0.50):
    rs = []; hs = []
    for (sym, j, c, h, l, n, d0) in ENT:
        j2 = min(n, j + MAX_HOLD)
        if j2 <= j + 1:
            continue
        px = c[j]
        seg = c[j:j2]
        peakc = np.maximum.accumulate(seg)
        dd = seg / np.maximum(peakc, 1e-12) - 1.0
        hit = np.flatnonzero(dd <= -drop)
        if len(hit):
            k = int(hit[0])
        else:
            k = len(seg) - 1
        rs.append(seg[k] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP)
        hs.append(k)
    A = np.array(rs); Hd = np.array(hs) / BAR_PER_DAY
    log(f'{drop:6.0%} | {len(A):7,d} {100*A.mean():+10.2f} {100*np.median(A):+9.2f} '
        f'{100*(A>0).mean():7.1f} {Hd.mean():10.1f} '
        f'{100*(A>=0.5).mean():7.1f} {100*(A>=1.0).mean():8.1f} {100*(A>=2.0).mean():8.1f}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
