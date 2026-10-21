# -*- coding: utf-8 -*-
"""
问题：用户称"他靠的是高杠杆"。加上杠杆后重算日内口径。
杠杆 L 的含义：标的变动 r% -> 本金变动 L*r%；标的逆向 1/L 即爆仓归零。
用 2h 决策密度（每天约 186 笔，对应"每 2.5 天一次"的实测档）。
输出 coinlab/evidence/leverage_REAL.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/leverage_REAL.txt'
H = 96; TOP = 5; COST = 0.0008; TP = 0.02; STEP = 24

D = pd.read_csv(CSV, usecols=['open_time','symbol','open','high','low','close'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','low','close'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

rows = []
for sym, g in D.groupby('symbol', sort=False):
    o = g.open.to_numpy(); h = g.high.to_numpy(); c = g.close.to_numpy()
    t = g.open_time.to_numpy(); n = len(o)
    if n < H + STEP + 2:
        continue
    sd = pd.Series(np.log(np.maximum(c, 1e-9))).diff().rolling(H, min_periods=H).std().to_numpy()
    idx = np.arange(H, n - H - 1, STEP)
    idx = idx[np.isfinite(sd[idx])]
    for e in idx:
        ep = float(o[e]); hi = h[e:e+H]; cl = float(c[e+H-1])
        hit = np.flatnonzero(hi >= ep * (1 + TP))
        rows.append((int(t[e]), sym, float(sd[e]), (TP if len(hit) else cl/ep - 1) - COST))
R = pd.DataFrame(rows, columns=['t','sym','vol','r'])
R = R.sort_values(['t','vol'], ascending=[True, False])
R['rk'] = R.groupby('t').cumcount()
R = R[R['rk'] < TOP].copy()
R['day'] = pd.to_datetime(R['t'], unit='ms').dt.strftime('%Y-%m-%d')
print('笔数', len(R), '天', R['day'].nunique(), '%.1fs' % (time.time()-t0), flush=True)

L = []
L.append('高杠杆下的日内口径（本金 +50%% 即停、次日新本金）   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('标的：2h 决策 × 波动率 Top5（每天约 186 笔）；单笔 TP=2%% 或持满 8h 收盘；成本 0.08%% 往返')
L.append('杠杆 L：标的价格动 r%% -> 本金动 L*r%%；标的逆向 1/L 即本金归零（强平）')
L.append('')
L.append('  杠杆    本金涨到 +50% 所需标的涨      达标天数     爆仓天数     日终值中位     3年终值')
for lev in (1, 5, 10, 20, 50):
    n_hit = n_liq = 0
    daily_end = []
    logeq = 0.0
    for d, g in R.groupby('day'):
        eq = 1.0
        dead = False
        for rr in g['r'].to_numpy():
            eq *= (1 + lev * rr)
            if eq <= 0:
                eq = 0.0; dead = True; break
            if eq >= 1.5:
                break
        if dead:
            n_liq += 1
        elif eq >= 1.5:
            n_hit += 1
        daily_end.append(eq)
        logeq += np.log(max(eq, 1e-12))
    nd = len(daily_end)
    L.append('  %3dx    %8.2f%%                 %5d (%5.1f%%)  %5d (%5.1f%%)  %+8.2f%%   %10.3e' % (
        lev, 50 / lev, n_hit, n_hit / nd * 100, n_liq, n_liq / nd * 100,
        np.median(daily_end) * 100 - 100, np.exp(min(logeq, 700))))
L.append('')
L.append('=== 判据')
L.append('  用户口径：每 3 天不到一次达标（>=33%）。看哪一档杠杆能到 33%，同时爆仓率是多少。')
L.append('  爆仓 = 当日本金被打到 0（真实交易所按维持保证金，通常更早强平，这里是最宽松估计）。')
L.append('')
L.append('诚实边界：')
L.append('  1. 杠杆按"逐笔复利、全额本金"，未模拟逐仓/全仓、维持保证金、资金费率、强平滑点。')
L.append('  2. 单笔 TP 用 high 触价；实盘高杠杆下流动性冲击更大。')
L.append('  3. 60 币池；未含小市值；未计手续费随杠杆放大的影响（成本已含在 r 内，按标的算，杠杆会放大）。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
