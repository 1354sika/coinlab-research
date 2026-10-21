# -*- coding: utf-8 -*-
"""接线验证：暴涨后窗口内，做空的期望收益到底是多少？
问题：我们已有的「做空正期望」（p>=0.31 -> +0.0544%/笔）是不是本来就来自「暴涨后的回归」？
做法：
  1. 用 1 年 1m 数据（60 币）找暴涨事件（4h = 240 根内涨幅 >= +50% / +100%）
  2. 标记暴涨后 72h 窗口，并分三档：0-6h / 6-24h / 24-72h
  3. 比较：窗口内各档 vs 窗口外，做空（收益取负）的期望净收益
  4. 叠加量比门槛（>5）看是否进一步区分
"""
import os, numpy as np, pandas as pd

DATA = r'D:\中信证券\coinlab\data'
FEE_M = 0.0002          # maker 单边
WIN   = 240             # 4h @1m
FWD_H = 4320            # 72h @1m

def say(*a): print(*a, flush=True)

say('=' * 100)
say('接线验证：暴涨后窗口内的做空期望（1 年 1m / 60 币）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='dt', columns='symbol', values='close').astype('float64').ffill(limit=30)
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float64')
T, N = close.shape
say('matrix %d x %d  跨度 %s ~ %s' % (T, N, close.index[0], close.index[-1]))
del df

C = close.values

# 未来 4h 最大涨幅
rev  = close.iloc[::-1]
wmax = rev.rolling(WIN, min_periods=1).max().iloc[::-1]
fwd_max = (wmax.shift(-1) / close - 1.0).values
del rev, wmax
say('未来窗口算完')

# 量比（5 分钟成交额 / 过去 60 分钟中位）
volr = (qv / qv.rolling(60, min_periods=20).median()).values
del qv
say('量比算完')

# 未来收益（多个持有期）
FW = {}
for H in [5, 15, 60]:
    FW[H] = (close.shift(-H) / close - 1.0).values

for PUMP_TH in [0.5, 1.0]:
    say('\n' + '=' * 100)
    say('【暴涨门槛：4h 内 >= +%.0f%%】' % (PUMP_TH * 100))
    say('=' * 100)

    pump = fwd_max >= PUMP_TH
    n_ev = 0
    w1 = np.zeros((T, N), dtype=bool)   # 0-6h
    w2 = np.zeros((T, N), dtype=bool)   # 6-24h
    w3 = np.zeros((T, N), dtype=bool)   # 24-72h
    for j in range(N):
        idx = np.flatnonzero(pump[:, j])
        last = -10 ** 9
        for i in idx:
            if i - last >= WIN:
                n_ev += 1
                # 0-6h（360 根）
                e1 = min(T, i + 360);  w1[i:e1, j] = True
                # 6-24h（360~1440）
                e2 = min(T, i + 1440); w2[e1:e2, j] = True
                # 24-72h（1440~4320）
                e3 = min(T, i + FWD_H); w3[e2:e3, j] = True
            last = i
    say('  去重后暴涨事件 = %d 个' % n_ev)
    say('  窗口内样本：0-6h %s ｜6-24h %s ｜24-72h %s ｜窗口外 %s'
        % (format(int(w1.sum()), ','), format(int(w2.sum()), ','),
           format(int(w3.sum()), ','), format(int((~(w1 | w2 | w3)).sum()), ',')))

    say('\n  %-14s %8s %10s %12s %12s %12s' % ('区段', '未来m', '样本', 'E[毛]%', '做空净maker%', '做空净taker%'))
    segs = [('窗口外', ~(w1 | w2 | w3)), ('暴涨后0-6h', w1), ('暴涨后6-24h', w2), ('暴涨后24-72h', w3)]
    for nm, mask in segs:
        for H in [5, 15, 60]:
            r = FW[H]
            m = mask & np.isfinite(r)
            n = int(m.sum())
            if n < 500: continue
            g = -r[m].mean() * 100          # 做空：收益取负
            say('  %-14s %6d    %10s  %+9.4f%%  %+10.4f%%  %+10.4f%%'
                % (nm, H, format(n, ','), -r[m].mean() * 100,
                   g - 2 * FEE_M * 100, g - 2 * 0.0010 * 100))

    # 叠加量比门槛
    say('\n  叠加「量比 > 5」后（做空 60 分钟）：')
    say('  %-22s %10s %12s %12s' % ('区段', '样本', 'E[毛]做空%', '净maker%'))
    vmask = (volr > 5) & np.isfinite(volr)
    for nm, mask in segs:
        m = mask & vmask & np.isfinite(FW[60])
        n = int(m.sum())
        if n < 200:
            say('  %-22s %10s  (样本不足)' % (nm, format(n, ',')))
            continue
        g = -FW[60][m].mean() * 100
        say('  %-22s %10s  %+9.4f%%  %+10.4f%%' % (nm, format(n, ','), g, g - 2 * FEE_M * 100))

say('\nTOTAL done')
