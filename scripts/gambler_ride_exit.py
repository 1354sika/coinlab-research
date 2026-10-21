# -*- coding: utf-8 -*-
"""gambler_ride_exit.py — 战略转向：不预测入场，只判断何时跑
用户洞察（2026-10-16）：
  "无论是大目标还是小目标，目前都无法精准的预测的。
   只能说知道如果抓到了之后，什么时候会跌"

⇒ 这推翻了此前的进攻方向。既然"入场"（预测哪个币会涨）覆盖率只有 24%，
   而"出场"（涨完之后何时见顶）是可观测的，那就应该：
     入场：追涨 —— 已经涨起来才进（不需要预测能力）
     出场：逃顶 —— 用量价信号判断顶部（这才是我们的优势）

比测 6 种出场规则（同一批追涨入场点）：
   A 固定 +10% 止盈 / -5% 止损 / 24h 超时
   B 移动止损：从最高点回落 10% / 20%
   C 量能萎缩：当前量比(6根) < 入场以来最大量比 × 0.5
   D 量价背离：价格创 24 根新高，但量比(6) < 量比(24)
   E 撤退组合：C 或 D 任一触发
   F 峰前三根特征：单根跌幅 > 3% 且 量比 > 5（"放量下跌"）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_ride_exit.txt')

FEE, SLIP = 0.0004, 0.0005
UP_ENTRY = 0.10        # 入场：过去 24 根（2h）涨 >= 10%
TO = 576               # 24h 上限
BPD = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_ride_exit · 追涨入场 + 逃顶出场（不预测入场，只判断何时跑）')
log('=' * 100)
log(f'  入场：过去 2h 涨 >= {UP_ENTRY:.0%}（追涨，无预测）')
log(f'  持有上限 {TO//BPD} 天；扣费 {FEE:.2%}+{SLIP:.2%}')

# 收集入场点及其后续序列
ENT = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 3000:
            continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        ot = g['open_time'].values

        r24 = np.full(n, np.nan); r24[24:] = c[24:] / c[:-24] - 1.0
        mv6 = pd.Series(q).rolling(6, min_periods=2).mean().values
        vr6 = np.where(mv6 > 0, q / np.maximum(mv6, 1e-12), 1.0)
        mv24 = pd.Series(q).rolling(24, min_periods=6).mean().values
        vr24 = np.where(mv24 > 0, q / np.maximum(mv24, 1e-12), 1.0)

        cand = np.flatnonzero(np.nan_to_num(r24) >= UP_ENTRY)
        last = -10 ** 9
        for j in cand:
            if j - last < 24:            # 冷却：同一波不重复入场
                continue
            if j + 5 >= n:
                break
            last = j
            k = min(TO, n - j)
            ENT.append((sym, int(j), c[j:j + k], h[j:j + k], l[j:j + k], q[j:j + k],
                        vr6[j:j + k], vr24[j:j + k], k))
    del ch; gc.collect()

NENT = len(ENT)
log(f'追涨入场点 {NENT:,}   耗时 {time.time()-t0t:.0f}s')

def sim(rule):
    rs = []
    for (sym, j, c, h, l, q, v6, v24, k) in ENT:
        px = c[0]
        if px <= 0 or k < 2:
            continue
        exit_k = None
        if rule[0] == 'fix':          # (fix, tp, sl)
            _, tp, sl = rule
            ret = c / (px * (1 + SLIP)) - 1.0
            a = np.flatnonzero(ret >= tp); b = np.flatnonzero(ret <= -sl)
            if len(a) and (not len(b) or a[0] <= b[0]):
                exit_k = int(a[0]); g = tp
            elif len(b):
                exit_k = int(b[0]); g = -sl
            else:
                exit_k = k - 1; g = float(ret[-1])
        elif rule[0] == 'trail':      # (trail, drop)
            _, drop = rule
            peak = np.maximum.accumulate(c)
            dd = c / np.maximum(peak, 1e-12) - 1.0
            hit = np.flatnonzero(dd <= -drop)
            if len(hit):
                exit_k = int(hit[0]); g = float(c[exit_k] / (px * (1 + SLIP)) - 1.0)
            else:
                exit_k = k - 1; g = float(c[-1] / (px * (1 + SLIP)) - 1.0)
        elif rule[0] == 'volshrink':  # (volshrink, frac)
            _, frac = rule
            peakv = np.maximum.accumulate(v6)
            hit = np.flatnonzero(v6 < frac * peakv)
            hit = hit[hit >= 3]
            if len(hit):
                exit_k = int(hit[0]); g = float(c[exit_k] / (px * (1 + SLIP)) - 1.0)
            else:
                exit_k = k - 1; g = float(c[-1] / (px * (1 + SLIP)) - 1.0)
        elif rule[0] == 'div':        # 量价背离：新高但量能不走高
            hi24 = pd.Series(h).rolling(24, min_periods=6).max().values
            cond = (h >= hi24 - 1e-12) & (v6 < v24)
            hit = np.flatnonzero(cond)
            hit = hit[hit >= 3]
            if len(hit):
                exit_k = int(hit[0]); g = float(c[exit_k] / (px * (1 + SLIP)) - 1.0)
            else:
                exit_k = k - 1; g = float(c[-1] / (px * (1 + SLIP)) - 1.0)
        elif rule[0] == 'combo':      # 量能萎缩 或 量价背离
            peakv = np.maximum.accumulate(v6)
            hi24 = pd.Series(h).rolling(24, min_periods=6).max().values
            cond = (v6 < 0.5 * peakv) | ((h >= hi24 - 1e-12) & (v6 < v24))
            hit = np.flatnonzero(cond)
            hit = hit[hit >= 3]
            if len(hit):
                exit_k = int(hit[0]); g = float(c[exit_k] / (px * (1 + SLIP)) - 1.0)
            else:
                exit_k = k - 1; g = float(c[-1] / (px * (1 + SLIP)) - 1.0)
        elif rule[0] == 'dump':       # 放量下跌：单根跌>3% 且 量比>5
            r1 = np.zeros(k); r1[1:] = c[1:] / c[:-1] - 1.0
            cond = (r1 <= -0.03) & (v6 > 5.0)
            hit = np.flatnonzero(cond)
            if len(hit):
                exit_k = int(hit[0]); g = float(c[exit_k] / (px * (1 + SLIP)) - 1.0)
            else:
                exit_k = k - 1; g = float(c[-1] / (px * (1 + SLIP)) - 1.0)
        else:
            continue
        rs.append((g - FEE - 2 * SLIP, exit_k / BPD,
                   float(h.max() / (px * (1 + SLIP)) - 1.0)))
    return rs

log('')
log('=' * 100)
log('【出场规则对比】（入场完全相同：追涨 +10%/2h）')
log('=' * 100)
log(f'{"规则":<34s} | {"笔数":>7s} {"期望净%":>9s} {"中位%":>8s} {"胜率%":>7s} {"平均持有天":>9s} {"平均最高涨幅%":>12s}')
RULES = [
    (('fix', 0.10, 0.05), 'A 固定 +10%/-5%'),
    (('fix', 0.20, 0.10), 'A 固定 +20%/-10%'),
    (('trail', 0.10), 'B 移动止损 回落10%'),
    (('trail', 0.20), 'B 移动止损 回落20%'),
    (('trail', 0.30), 'B 移动止损 回落30%'),
    (('volshrink', 0.5), 'C 量能萎缩到峰值50%'),
    (('volshrink', 0.3), 'C 量能萎缩到峰值30%'),
    (('div', None), 'D 量价背离（新高但量不走高）'),
    (('combo', None), 'E 组合（C 或 D）'),
    (('dump', None), 'F 放量下跌（跌>3%且量比>5）'),
]
best = None
for rule, tag in RULES:
    rs = sim(rule)
    if not rs:
        log(f'{tag:<34s} | 无样本'); continue
    A = np.array([r[0] for r in rs]); HD = np.array([r[1] for r in rs]); PK = np.array([r[2] for r in rs])
    log(f'{tag:<34s} | {len(A):7,d} {100*A.mean():+9.3f} {100*np.median(A):+8.3f} '
        f'{100*(A>0).mean():7.1f} {HD.mean():9.2f} {100*PK.mean():12.2f}')
    if best is None or A.mean() > best[1]:
        best = (tag, A.mean(), len(A), HD.mean())

log('')
log('=' * 100)
log('【基准对照】')
log('=' * 100)
rs = []
for (sym, j, c, h, l, q, v6, v24, k) in ENT:
    px = c[0]
    if px <= 0 or k < 2:
        continue
    rs.append(c[-1] / (px * (1 + SLIP)) - 1.0 - FEE - 2 * SLIP)
A = np.array(rs)
log(f'  入场后一直持有到 24h（什么都不做）：笔 {len(A):,}  期望净 {100*A.mean():+.3f}%  '
    f'中位 {100*np.median(A):+.3f}%  胜率 {100*(A>0).mean():.1f}%')
if best:
    log(f'  ★ 最优规则：{best[0]}  期望净 {100*best[1]:+.3f}%/笔  笔 {best[2]:,}  平均持有 {best[3]:.2f} 天')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
