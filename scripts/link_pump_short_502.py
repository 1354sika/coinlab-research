# -*- coding: utf-8 -*-
"""接线验证（扩样本版）：502 币 x 3 年 5m
复用 link_pump_short.py 的逻辑，但币数从 60 -> 502、粒度 1m -> 5m
内存优化：全程 float32
"""
import os, gc, numpy as np, pandas as pd

DATA = r'D:\中信证券\coinlab\data'
FEE_M = 0.0002          # maker 单边
WIN   = 48              # 4h @5m
FWD_H = 864             # 72h @5m

def say(*a): print(*a, flush=True)

say('=' * 100)
say('接线验证（扩样本：502 币 x 3 年 5m）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
say('原始行数 %s  币数 %d' % (format(len(df), ','), df.symbol.nunique()))

close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=12)
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
T, N = close.shape
say('matrix %d x %d (%d 币)  跨度 %s ~ %s' % (T, N, N, close.index[0], close.index[-1]))
del df; gc.collect()

C = close.values

# 未来 4h 最大涨幅
rev  = close.iloc[::-1]
wmax = rev.rolling(WIN, min_periods=1).max().iloc[::-1]
fwd_max = (wmax.shift(-1) / close - 1.0).values.astype('float32')
del rev, wmax; gc.collect()
say('未来窗口算完')

# 量比（5 分钟成交额 / 过去 60 分钟中位）= 1 根 / 12 根中位
volr = (qv / qv.rolling(12, min_periods=6).median()).values.astype('float32')
del qv; gc.collect()
say('量比算完')

FW = {}
for H in [1, 3, 12]:      # 5m / 15m / 60m
    FW[H] = (close.shift(-H) / close - 1.0).values.astype('float32')
say('未来收益算完')

for PUMP_TH in [0.5, 1.0]:
    say('\n' + '=' * 100)
    say('【暴涨门槛：4h 内 >= +%.0f%%】' % (PUMP_TH * 100))
    say('=' * 100)

    pump = fwd_max >= PUMP_TH
    n_ev = 0
    w1 = np.zeros((T, N), dtype=bool)
    w2 = np.zeros((T, N), dtype=bool)
    w3 = np.zeros((T, N), dtype=bool)
    for j in range(N):
        idx = np.flatnonzero(pump[:, j])
        last = -10 ** 9
        for i in idx:
            if i - last >= WIN:
                n_ev += 1
                e1 = min(T, i + 72)      # 0-6h @5m
                e2 = min(T, i + 288)     # 6-24h
                e3 = min(T, i + FWD_H)   # 24-72h
                w1[i:e1, j] = True
                w2[e1:e2, j] = True
                w3[e2:e3, j] = True
            last = i
    say('  去重后暴涨事件 = %d 个（502 币 / 3 年）' % n_ev)
    say('  窗口内样本：0-6h %s ｜6-24h %s ｜24-72h %s ｜窗口外 %s'
        % (format(int(w1.sum()), ','), format(int(w2.sum()), ','),
           format(int(w3.sum()), ','), format(int((~(w1 | w2 | w3)).sum()), ',')))

    say('\n  %-14s %8s %14s %14s %14s' % ('区段', '未来', '样本', 'E[毛]%', '做空净maker%'))
    segs = [('窗口外', ~(w1 | w2 | w3)), ('暴涨后0-6h', w1), ('暴涨后6-24h', w2), ('暴涨后24-72h', w3)]
    for nm, mask in segs:
        for H, lbl in [(1, '5m'), (3, '15m'), (12, '60m')]:
            r = FW[H]
            m = mask & np.isfinite(r)
            n = int(m.sum())
            if n < 2000: continue
            g = -float(r[m].mean()) * 100
            say('  %-14s %8s %14s %+13.4f%% %+13.4f%%'
                % (nm, lbl, format(n, ','), g, g - 2 * FEE_M * 100))

    say('\n  叠加「量比 > 5」（做空 60 分钟）：')
    vmask = (volr > 5) & np.isfinite(volr)
    for nm, mask in segs:
        m = mask & vmask & np.isfinite(FW[12])
        n = int(m.sum())
        if n < 500:
            say('    %-22s %12s  (样本不足)' % (nm, format(n, ','))); continue
        g = -float(FW[12][m].mean()) * 100
        say('    %-22s %12s  E[毛] %+9.4f%%  净maker %+9.4f%%'
            % (nm, format(n, ','), g, g - 2 * FEE_M * 100))
    del pump, w1, w2, w3; gc.collect()

say('\nTOTAL done')
