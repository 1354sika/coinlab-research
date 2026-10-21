# -*- coding: utf-8 -*-
"""事件对普查：暴涨 → 暴跌（操盘手「局」的完整生命周期）
1. 扫出「4h 内涨幅 >= X」的暴涨事件（同币去重）
2. 对每个事件：看之后 72h 内的最大回撤、回撤发生得多久
3. 统计：配对率（暴涨后是否真暴跌）、幅度关系、时间间隔
4. 防幸存者偏差：同时统计「暴涨后没暴跌」的比例
数据：klines_5m_3y.csv（3 年 5m，60 币）
"""
import os, numpy as np, pandas as pd

DATA = r'D:\中信证券\coinlab\data'
WIN  = 48            # 4 小时 @5m
FWD  = 864           # 72 小时 @5m

def say(*a): print(*a, flush=True)

say('=' * 100)
say('事件对普查：暴涨 → 暴跌（3 年 5m / 60 币）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_3y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='dt', columns='symbol', values='close').astype('float64').ffill(limit=12)
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float64')
T, N = close.shape
say('matrix %d x %d  跨度 %s ~ %s' % (T, N, close.index[0], close.index[-1]))
del df

# 未来 4h 的最大/最小（从 t+1 起算）
rev  = close.iloc[::-1]
wmax = rev.rolling(WIN, min_periods=1).max().iloc[::-1]
wmin = rev.rolling(WIN, min_periods=1).min().iloc[::-1]
fwd_max = (wmax.shift(-1) / close - 1.0).values
fwd_min = (wmin.shift(-1) / close - 1.0).values
del rev, wmax, wmin
say('未来窗口算完')

C = close.values
# 量比（5 分钟成交额 / 过去 60 分钟中位）
volr = (qv / qv.rolling(12, min_periods=6).median()).values
del qv
say('量比算完')

for PUMP_TH in [2.0, 1.0, 0.5]:
    say('\n' + '=' * 100)
    say('【暴涨门槛 4h >= +%.0f%%】' % (PUMP_TH * 100))
    say('=' * 100)

    ev = fwd_max >= PUMP_TH
    rows, cols = np.where(ev)
    if len(rows) == 0:
        say('  无事件'); continue

    # 同币去重：相邻事件间隔 < WIN 只保留第一个
    sel = []
    last_c, last_r = -1, -10 ** 9
    for r, c in zip(rows, cols):
        if c != last_c or (r - last_r) >= WIN:
            sel.append((r, c))
        last_c, last_r = c, r
    say('  原始命中点 %s -> 去重后事件 %d 个'
        % (format(len(rows), ','), len(sel)))

    n_ev = len(sel)
    pairs, novictim, nofuture = 0, 0, 0
    gaps, pump_amp, dump_amp = [], [], []

    for r, c in sel:
        end = min(T - 1, r + WIN)
        seg_end = min(T, end + FWD)
        seg = C[r:seg_end, c]
        fut = C[r + 1:end + 1, c]
        if len(fut) < 5 or len(seg) < 10:
            nofuture += 1
            continue
        peak = np.nanmax(fut)
        trough = np.nanmin(seg)
        dd = trough / peak - 1.0
        if np.isfinite(dd) and dd <= -0.30:
            pairs += 1
            dump_amp.append(dd)
            pump_amp.append(fwd_max[r, c])
            # 修正：从「峰值之后」开始找最低点（原来从事件起点算，得到不合理的 0 小时）
            peak_i = r + 1 + int(np.nanargmax(fut))
            post = C[peak_i:seg_end, c]
            if len(post) > 0:
                trough_i = peak_i + int(np.nanargmin(post))
                gaps.append((trough_i - peak_i) * 5.0 / 60.0)
        else:
            novictim += 1

    if n_ev == 0: continue
    say('  其中 72h 内出现 <= -30%% 回撤的 = %d（%.1f%%）' % (pairs, pairs / n_ev * 100))
    say('  未出现大回撤的             = %d（%.1f%%）' % (novictim, novictim / n_ev * 100))
    if nofuture: say('  数据不足无法判定           = %d' % nofuture)
    if gaps:
        g = np.array(gaps)
        say('  暴涨 -> 最低点 间隔：中位 %.1f 小时（P25 %.1f / P75 %.1f）'
            % (np.median(g), np.percentile(g, 25), np.percentile(g, 75)))
        say('  暴涨幅度中位 %+.1f%%    后续回撤中位 %.1f%%'
            % (np.median(pump_amp) * 100, np.median(dump_amp) * 100))
        for th in [0.3, 0.5, 0.7]:
            m = np.array(dump_amp) <= -th
            say('    回撤 >= %.0f%% 的事件占比 = %.1f%%' % (th * 100, m.mean() * 100))

say('\n【对照】不同暴涨幅度下，"未暴跌"的比例（防自证预言）')
say('=' * 100)
ev = fwd_max >= 0.5
rows, cols = np.where(ev)
sel = []
last_c, last_r = -1, -10 ** 9
for r, c in zip(rows, cols):
    if c != last_c or (r - last_r) >= WIN:
        sel.append((r, c))
    last_c, last_r = c, r
buckets = {(0.5, 1.0): [0, 0], (1.0, 2.0): [0, 0], (2.0, 10.0): [0, 0], (10.0, 1e9): [0, 0]}
for r, c in sel:
    amp = fwd_max[r, c]
    end = min(T - 1, r + WIN)
    seg_end = min(T, end + FWD)
    seg = C[r:seg_end, c]
    fut = C[r + 1:end + 1, c]
    if len(fut) < 5 or len(seg) < 10: continue
    dd = np.nanmin(seg) / np.nanmax(fut) - 1.0
    for k in buckets:
        if k[0] <= amp < k[1]:
            buckets[k][0] += 1
            if np.isfinite(dd) and dd <= -0.30:
                buckets[k][1] += 1
            break
say('  暴涨幅度区间          事件数    其中暴跌(<=-30%%)   暴跌占比')
for k, (tot, hit) in buckets.items():
    if tot == 0: continue
    say('  [%+5.0f%%, %+8.0f%%)   %7d   %10d   %8.1f%%'
        % (k[0] * 100, min(k[1], 9999) * 100, tot, hit, hit / tot * 100))

say('\n【暴跌前 1h 的特征（对照全样本）】')
say('=' * 100)
# 暴跌事件：未来 4h 跌幅 <= -30%
dn = fwd_min <= -0.30
say('  暴跌事件命中点 = %s（未去重）' % format(int(np.nansum(dn)), ','))
vr_d = volr[dn]
vr_d = vr_d[np.isfinite(vr_d)]
vr_a = volr[np.isfinite(volr)]
say('  暴跌前 量比 中位 = %.2f     全样本 量比 中位 = %.2f' % (np.median(vr_d), np.median(vr_a)))
for q in [0.5, 0.9, 0.99]:
    say('    量比 P%.0f：暴跌前 %.2f   全样本 %.2f'
        % (q * 100, np.percentile(vr_d, q * 100), np.percentile(vr_a, q * 100)))

say('\nTOTAL done')
