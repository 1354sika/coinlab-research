# -*- coding: utf-8 -*-
"""gambler_tp2_1m_full · 502 币 × 86400 分钟 全量：列出所有「先触 +2%」的达标周期

用户口径（2026-10-17）：
  投 1 块赚到 1.2 块；跌掉收益的 50%（即 -1%）止损；一天 15 次以上。

本脚本全量扫描 klines_1m_502.csv（502 币 × 60 天 = 43,200,000 根 1m），
对每一个分钟点判断「未来 60 分钟内先触 +2% 还是先触 -1%」，
输出所有先触 +2% 的达标周期（不重叠），并给出频率与分场景成功率。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL = 0.02, 0.01
HOLD = 60            # 1 小时 = 60 根 1m
FEE_M, FEE_T = 0.0004, 0.0020
DAYS = 60

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2_1m_full · 全量 502 币 × 86400 分钟：先触 +2% 的达标周期')
log('=' * 108)
log(f'口径：TP+{TP:.0%} / SL-{SL:.0%} / 窗口 {HOLD} 分钟 / 不重叠')
log('')

rows = []
tot_pts = 0
tot_tp = 0
tot_sl = 0
tot_to = 0
t_read = time.time()
n_sym = 0

for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64',
                             'close': 'float32', 'high': 'float32', 'low': 'float32',
                             'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        n = len(c)
        m = n - HOLD
        if m <= 0:
            continue
        n_sym += 1
        cc = c[:m]
        INF = 10 ** 9
        first_tp = np.full(m, INF, dtype=np.int64)
        first_sl = np.full(m, INF, dtype=np.int64)
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]
            ll = l[j:j + m]
            nt = (first_tp == INF) & (hh >= cc * (1 + TP))
            ns = (first_sl == INF) & (ll <= cc * (1 - SL))
            first_tp[nt] = j
            first_sl[ns] = j
        win = first_tp < first_sl
        lose = first_sl < first_tp
        tie = (first_tp == first_sl) & (first_tp < INF)
        none = (first_tp == INF) & (first_sl == INF)
        tot_pts += m
        tot_tp += int(win.sum())
        tot_sl += int((lose | tie).sum())
        tot_to += int(none.sum())

        # 不重叠的达标周期：贪心
        idx = np.flatnonzero(win)
        if len(idx) == 0:
            continue
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values
        last_end = -1
        for i in idx:
            if i <= last_end:
                continue
            j = i + int(first_tp[i])
            last_end = j
            entry = cc[i]
            r5 = entry / c[i - 5] - 1 if i >= 5 else np.nan
            r15 = entry / c[i - 15] - 1 if i >= 15 else np.nan
            volr = q[i] / qma[i] if qma[i] > 0 else np.nan
            rows.append((sym, int(ot[i]), int(ot[j]), int(j - i),
                         float(entry), float(c[j]), float(r5), float(r15), float(volr)))

log(f'币 {n_sym}  读盘+判定 {time.time()-t_read:.0f}s')
log(f'总入场点数 {tot_pts:,}  先触TP {tot_tp:,}  先触SL {tot_sl:,}  60分钟内都未触 {tot_to:,}')
log(f'先触TP率 {tot_tp/tot_pts:.2%}  （门槛 34.7% maker）')
log(f'达标周期（不重叠）{len(rows):,} 个   平均每天 {len(rows)/DAYS:,.0f} 个（全 502 币）')

df = pd.DataFrame(rows, columns=['symbol', 't_entry', 't_exit', 'minutes',
                                 'entry', 'exit', 'r5', 'r15', 'volr'])
csv = os.path.join(OUT, 'tp2_1m_cycles.csv')
df.to_csv(csv, index=False, encoding='utf-8')
log(f'\n周期明细已写 {csv}（{len(df):,} 行）')

log('')
log('=' * 108)
log('【达标周期的形态】')
log('=' * 108)
log(f'持有分钟  中位 {df.minutes.median():.0f}  P25 {np.percentile(df.minutes,25):.0f}  '
    f'P75 {np.percentile(df.minutes,75):.0f}  P90 {np.percentile(df.minutes,90):.0f}  最大 {df.minutes.max()}')
log(f'10 分钟内达标的占比 {(df.minutes<=10).mean():.1%}')
log(f'30 分钟内达标的占比 {(df.minutes<=30).mean():.1%}')
log(f'入场前 5m 涨跌幅 中位 {df.r5.median():+.2%}   入场时量比 中位 {df.volr.median():.2f}')
log('')
log('按入场前 15m 涨跌分档：')
df['b'] = pd.cut(df.r15, [-1, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 1],
                 labels=['<-5%', '-5~-2%', '-2~-0.5%', '±0.5%', '+0.5~2%', '+2~5%', '>+5%'])
for k, g in df.groupby('b', observed=True):
    log(f'  {str(k):12s} n={len(g):>7,}  {len(g)/len(df):>6.1%}  持有中位 {g.minutes.median():>4.0f} 分钟')

out = os.path.join(OUT, 'tp2_1m_full_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
