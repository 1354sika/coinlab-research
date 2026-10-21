# -*- coding: utf-8 -*-
"""gambler_stop_dynamic · 动态止损对比

背景：固定 -9.5% 止损 + 24h 超时出场，最差单笔 -71.5%（超时把亏损放大）。
用户口径：止损信号一到就撤，但止损强度每个周期可能不同。

对比 4 种规则（入场 = 涨 5% 时刻，v240 >= p90 的周期）：
  A 固定：  入场价 -9.5%
  B 移动：  从持仓期最高点回落 25%
  C 波动率：入场价 - 2*v240（v240 = 5m 收益率标准差，逐周期动态）
  D 时间：  只止盈，24h 到期按收盘出（不设价格止损）
评价：止盈率 / 每笔均值 / 中位 / 最差 / 固定10%仓位资金曲线 / 最大回撤
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_stop_dynamic · 动态止损对比')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
RES = {k: [] for k in ['A_fixed','B_trail','C_vol','D_time']}
META = []   # (t, v240)

def sim(entry, tp, sl_fixed, h, l, c, j, n, v, mode):
    if mode == 'C_vol':
        sl_fixed = entry * (1 - 2.0 * v)
    hi = entry
    for k in range(j + 1, min(j + 289, n)):
        if h[k] >= tp:
            return tp / entry - 1, 'TP'
        if mode == 'B_trail':
            hi = max(hi, h[k])
            if l[k] <= hi * 0.75:
                return hi * 0.75 / entry - 1, 'TRAIL'
        elif mode != 'D_time':
            if l[k] <= sl_fixed:
                return sl_fixed / entry - 1, 'SL'
    kk = min(j + 288, n - 1)
    return c[kk] / entry - 1, 'TO'

for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in need or len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        n = len(c)
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1
        v240 = pd.Series(r1).rolling(240, min_periods=50).std().values
        for _, row in by_sym[sym].iterrows():
            lo_i = int(row['lo_i']); lo_px = float(row['lo'])
            if lo_i >= n - 2: continue
            j = lo_i
            target = lo_px * 1.05
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1 or j - lo_i > 288: continue
            if not np.isfinite(v240[j]): continue
            entry = c[j]; tp = lo_px * 1.575; sl = lo_px * 0.95
            for mode in RES:
                ret, reason = sim(entry, tp, sl, h, l, c, j, n, v240[j], mode)
                RES[mode].append((ot[j], ret, reason, v240[j]))
    del ch; gc.collect()

for mode, rows in RES.items():
    rows.sort()
    T = np.array([r[0] for r in rows]); V = np.array([r[3] for r in rows])
    R = np.array([r[1] for r in rows]); RS = np.array([r[2] for r in rows])
    cut = np.median(T); trm = T <= cut; tem = T > cut
    th = np.quantile(V[trm], 0.90)
    sel = tem & (V >= th)
    r = R[sel]
    if len(r) == 0: continue
    eq = 1.0 + 0.1 * np.cumsum(r)
    dd = np.maximum.accumulate(eq) - eq
    log(f'\n【{mode}】入场 {len(r):,} 笔')
    log(f'  止盈 {(RS[sel]=="TP").mean():>6.1%}  止损 {(RS[sel]=="SL").mean():>6.1%}  '
        f'移动触发 {(RS[sel]=="TRAIL").mean():>6.1%}  超时 {(RS[sel]=="TO").mean():>6.1%}')
    log(f'  均值 {r.mean():>+7.2%}  中位 {np.median(r):>+7.2%}  最差 {r.min():>+7.1%}')
    log(f'  固定10%仓位：最终 {eq[-1]:>6.2f}x  最大回撤 {dd.max():>6.3f}')

log('\n【判读】')
log('  · 看「最差单笔」和「最大回撤」：动态止损应把 -71.5% 压下去')
log('  · 看「止盈占比」：止损收得太紧会把到 50% 的单子提前打掉')

out = os.path.join(OUT, 'stop_dynamic_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
