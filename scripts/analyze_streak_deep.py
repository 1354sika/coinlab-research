# -*- coding: utf-8 -*-
"""C) 连续段维度深挖（在 B 的教训之后：所有统计都同时报 去重前/去重后 两个口径）
特征：
  C1 段内速率 speed = 累计幅度 / 连续分钟数（每分钟平均）  -> 快速暴跌 vs 缓慢阴跌
  C2 段内加速 accel = 后半段幅度 / 前半段幅度              -> 是否在加速
  C3 段内量比 volr = 段内均量 / 段前60分钟均量
  C4 段长 x 幅度 二维网格
  C5 段结束后第一根（是否已出现反向确认）
目标：未来 60 分钟收益；并回答「加上这些维度，能否把组合从亏损救回来」
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, THR = 60, 0.05
t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
QV = qv.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seg_len = (rows - start + 1).astype(np.float64)
P0 = np.take_along_axis(P, start, axis=0)
streak_ret = P / P0 - 1.0
speed = streak_ret / seg_len                      # 每分钟平均幅度

# 段内加速：前半段/后半段幅度
half = np.maximum((seg_len / 2).astype(np.int64), 1)
mid_row = np.maximum(rows - half, start)
Pmid = np.take_along_axis(P, mid_row, axis=0)
first_half = Pmid / P0 - 1.0
second_half = P / Pmid - 1.0
accel = np.where(np.abs(first_half) > 1e-9, second_half / np.where(first_half == 0, np.nan, first_half), np.nan)

# 段内量比
qv60 = qv.rolling(60).mean().values
volr = QV / qv60

FWD60 = np.full((T, N), np.nan)
FWD60[:-60] = P[60:] / P[:-60] - 1.0
d90 = float(np.nanpercentile(disp_v, 90))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
SIG = (streak_ret < -THR) & sel

# 去重（用于第二口径）
opens = []
for j in range(N):
    ii = np.flatnonzero(SIG[:, j])
    if len(ii) == 0:
        continue
    last = -1
    for k in ii:
        if k <= last:
            continue
        if np.isfinite(FWD60[k, j]):
            opens.append((k, j))
        last = k + HOLD - 1
opens = np.array(opens, int)
mask_open = np.zeros((T, N), bool)
for k, j in opens:
    mask_open[k, j] = True
say('信号 去重前=%d  去重后=%d' % (int(SIG.sum()), len(opens)))


def two(name, feat, bins, sig=SIG):
    say('\n--- %s ---' % name)
    say('  %-18s %8s %11s | %8s %11s' % ('档', 'n(全)', '均值(全)', 'n(去重)', '均值(去重)'))
    for lo, hi, lb in bins:
        m = sig & (feat >= lo) & (feat < hi)
        y = FWD60[m]; y = y[np.isfinite(y)]
        mo = m & mask_open
        yo = FWD60[mo]; yo = yo[np.isfinite(yo)]
        if len(y) < 100:
            continue
        say('  %-18s %8d %10.3f%% | %8d %10.3f%%' % (
            lb, len(y), 100*y.mean(), len(yo), 100*yo.mean() if len(yo) else np.nan))


two('C1 段内速率 speed（每分钟平均跌幅，越负越快）', speed,
    [(-0.05, -0.02, '>=2%/min极快'), (-0.02, -0.01, '1-2%/min快'),
     (-0.01, -0.005, '0.5-1%/min中'), (-0.005, 0, '<0.5%/min慢')])

two('C2 段内加速 accel（后半/前半，>1=加速下跌）', accel,
    [(0, 0.5, '大幅减速'), (0.5, 1.0, '减速'), (1.0, 2.0, '加速'),
     (2.0, 10, '大幅加速'), (10, 1e9, '极端加速')])

two('C3 段内量比 volr', volr,
    [(0, 1, '<1缩量'), (1, 2, '1-2倍'), (2, 5, '2-5倍'), (5, 1e9, '>5倍')])

two('C4a 段长（连续下跌分钟数）', seg_len,
    [(0, 3, '1-2分钟'), (3, 6, '3-5分钟'), (6, 11, '6-10分钟'), (11, 1e9, '>=11分钟')])

say('\n=== C4b 段长 x 幅度 网格（去重后口径）===')
lens = [(0, 3, '1-2m'), (3, 6, '3-5m'), (6, 11, '6-10m'), (11, 1e9, '>=11m')]
deps = [(-0.07, -0.05, '-7~-5%'), (-0.10, -0.07, '-10~-7%'),
        (-0.15, -0.10, '-15~-10%'), (-1, -0.15, '<-15%')]
say('  %-10s' % '段长\\幅度' + ''.join('%14s' % d[2] for d in deps))
for lo1, hi1, l1 in lens:
    row = '  %-10s' % l1
    for lo2, hi2, l2 in deps:
        m = SIG & (seg_len >= lo1) & (seg_len < hi1) & (streak_ret >= lo2) & (streak_ret < hi2) & mask_open
        y = FWD60[m]; y = y[np.isfinite(y)]
        row += '%14s' % ('n=%d %+.2f%%' % (len(y), 100*y.mean()) if len(y) >= 5 else '-')
    say(row)

say('\n=== C5 段距离峰值的时间（跌幅 / 速率 = 已持续分钟占比的替代）===')
say('  说明：若「跌得越快、段越长」的样本更好，应优先做那些；见下表')
best = []
for lo1, hi1, l1 in lens:
    for lo2, hi2, l2 in deps:
        m = SIG & (seg_len >= lo1) & (seg_len < hi1) & (streak_ret >= lo2) & (streak_ret < hi2) & mask_open
        y = FWD60[m]; y = y[np.isfinite(y)]
        if len(y) >= 30:
            best.append((100*y.mean(), len(y), l1, l2))
best.sort(reverse=True)
say('  最优 8 格（去重后，n>=30）:')
for mu, n, l1, l2 in best[:8]:
    say('    %-8s %-10s n=%5d 均值=%+.3f%%' % (l1, l2, n, mu))
say('  最差 4 格:')
for mu, n, l1, l2 in best[-4:]:
    say('    %-8s %-10s n=%5d 均值=%+.3f%%' % (l1, l2, n, mu))

with open(os.path.join(EV, 'streak_deep.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
