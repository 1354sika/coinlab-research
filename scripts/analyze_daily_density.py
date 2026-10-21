# -*- coding: utf-8 -*-
"""
问题：用户称"他每三天不到就能实现一次本金 +50%（日内累计）"。
做法：把日内决策密度拉高（每 8h / 2h / 1h 一个决策点，各取波动率 Top5），
      重算"一天内所有笔累计 >= +50%"的天数占比，看能否达到 1/3（每 3 天一次）。
输出 coinlab/evidence/daily_density.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/daily_density.txt'
H = 96; TOP = 5; COST = 0.0008; TP = 0.02

D = pd.read_csv(CSV, usecols=['open_time','symbol','open','high','low','close'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','low','close'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

L = []
L.append('日内到 +50%% 的频率 vs 决策密度   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('出场：TP=2%% 先触及；否则持满 96 根收盘；成本往返 0.08%%；60 币池')
L.append('')
L.append('  决策间隔   每天笔数   达标天数   占比      折合"每几天一次"   日收益中位')
for STEP in (96, 24, 12):
    rows = []
    for sym, g in D.groupby('symbol', sort=False):
        o = g.open.to_numpy(); h = g.high.to_numpy(); c = g.close.to_numpy()
        t = g.open_time.to_numpy(); n = len(o)
        if n < H + STEP + 2:
            continue
        lr = pd.Series(np.log(np.maximum(c, 1e-9))).diff()
        sd = lr.rolling(H, min_periods=H).std().to_numpy()
        idx = np.arange(H, n - H - 1, STEP)
        idx = idx[np.isfinite(sd[idx])]
        if len(idx) == 0:
            continue
        for e in idx:
            ep = float(o[e]); hi = h[e:e+H]; cl = float(c[e+H-1])
            target = ep * (1 + TP)
            hit = np.flatnonzero(hi >= target)
            r = (TP if len(hit) else cl/ep - 1) - COST
            rows.append((int(t[e]), sym, sd[e], r))
    R = pd.DataFrame(rows, columns=['t','sym','vol','r'])
    R = R.sort_values(['t','vol'], ascending=[True, False])
    R['rk'] = R.groupby('t').cumcount()
    R = R[R['rk'] < TOP].copy()
    R['day'] = pd.to_datetime(R['t'], unit='ms').dt.strftime('%Y-%m-%d')
    daily = R.groupby('day')['r'].apply(lambda x: float(np.prod(1 + x.to_numpy()) - 1))
    nd = len(daily); hit = (daily >= 0.5).sum()
    per = nd / max(hit, 1)
    L.append('  %2dh        %6.1f    %5d    %6.2f%%   每 %.2f 天一次      %+6.2f%%' % (
        STEP * 5 // 60, len(R) / nd, hit, hit / nd * 100, per, np.median(daily) * 100))
    print('  STEP=%d done %.0fs' % (STEP, time.time()-t0), flush=True)

L.append('')
L.append('读法：')
L.append('  用户口径"每三天不到一次" = 达标率 >= 33%%。看哪一档能到 33%%。')
L.append('  注意：密度提高同时放大了亏损日，达标率上升不等于能赚钱。')
L.append('')
L.append('诚实边界：')
L.append('  1. 全部回测，TP 用 high 触价、零滑点、挂单成交率未计。')
L.append('  2. 每档都是"波动率 Top5"，未调优；换选币口径数字会变。')
L.append('  3. 60 币池；304 币池（小市值）达标率应更高，未测。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
