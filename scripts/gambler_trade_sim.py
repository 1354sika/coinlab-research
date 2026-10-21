# -*- coding: utf-8 -*-
"""gambler_trade_sim · 按概率行动：从「算命」到「用命」

问题：前面所有实验都只算概率/期望，从没按概率真正"下单-持仓-出场"走一遍，
      所以不知道"在最高概率的时刻入场，最后资金曲线长什么样"。

本脚本：
  1. 用 early20 的模型（v240 波动率单特征，AUC 0.7123）给出每个周期「涨 5%」时刻的概率
  2. 概率 >= 阈值 时入场：入场价 = lo * 1.05
  3. 出场规则：
       止盈：峰值 >= lo*1.575（相对入场 +50%）
       止损：周期结束时价格 <= lo*0.95（相对入场 -9.5%）
       否则按周期结束价出场
  4. 记录逐笔盈亏、累计资金曲线、最大回撤、胜率
局限（必须标注）：用 cycles.csv 的 peak/peak_to_end 近似路径，
                  不判断「止盈和止损谁先到」；未扣滑点。
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
log('gambler_trade_sim · 按概率行动：入场-出场-资金曲线')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
rows = []
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
            rows.append((v240[j], float(row['gain']), float(row['peak']), float(row['peak_to_end']),
                         lo_px, ot[j], sym))
    del ch; gc.collect()

log(f'可评估周期 {len(rows):,}')
v = np.array([r[0] for r in rows]); gain = np.array([r[1] for r in rows])
peak = np.array([r[2] for r in rows]); p2e = np.array([r[3] for r in rows])
lo = np.array([r[4] for r in rows]); T = np.array([r[5] for r in rows])
syms = [r[6] for r in rows]

cut = np.median(T); trm = T <= cut; tem = T > cut
log(f'训练 {trm.sum():,}  测试 {tem.sum():,}')

# 阈值 = 训练段 v240 的 p70 / p80（高波动前 30% / 20%）
for q in [0.6, 0.7, 0.8, 0.9]:
    th = np.quantile(v[trm], q)
    sel = tem & (v >= th)
    nn = sel.sum()
    if nn == 0: continue
    entry = lo[sel] * 1.05
    # 止盈：峰值 >= lo*1.575（即 gain >= 0.50）
    win = gain[sel] >= 0.50
    # 非止盈：出场价 = peak*(1+peak_to_end)
    exit_px = peak[sel] * (1 + p2e[sel])
    ret_win = 1.575/1.05 - 1          # +50%
    ret_lose = np.maximum(exit_px/entry - 1, -0.095)   # 止损 -9.5%
    ret = np.where(win, ret_win, ret_lose)
    # 等权资金曲线（每笔固定 1 单位）
    curve = np.cumsum(np.sort(ret))  # 粗略：按收益排序不代表时间序，仅看分布
    # 时间序资金曲线
    o = np.argsort(T[sel])
    eq = np.cumsum(ret[o])
    dd = np.minimum.accumulate(eq) - eq
    log(f'\n【v240 >= p{q*100:.0f}】入场 {nn:,} 笔')
    log(f'  到 50% 止盈比例 {win.mean():>6.1%}')
    log(f'  每笔均值 {ret.mean():>+7.2%}  中位 {np.median(ret):>+7.2%}')
    log(f'  累计（等权，1 单位/笔） {eq[-1]:>+8.1%}   最大回撤 {dd.max():>7.2%}')
    log(f'  最差单笔 {ret.min():>+7.1%}  最好 {ret.max():>+7.1%}')

# 对比：不止损（跌回也不管，按结束价出）
log('\n【对照：不止损，按周期结束价出场】')
for q in [0.7, 0.8]:
    th = np.quantile(v[trm], q)
    sel = tem & (v >= th)
    nn = sel.sum()
    if nn == 0: continue
    entry = lo[sel] * 1.05
    win = gain[sel] >= 0.50
    exit_px = peak[sel] * (1 + p2e[sel])
    ret = np.where(win, 1.575/1.05 - 1, exit_px/entry - 1)
    o = np.argsort(T[sel])
    eq = np.cumsum(ret[o])
    dd = np.minimum.accumulate(eq) - eq
    log(f'  p{q*100:.0f}: {nn:,} 笔  均值 {ret.mean():>+7.2%}  累计 {eq[-1]:>+8.1%}  最大回撤 {dd.max():>7.2%}  最差 {ret.min():>+7.1%}')

log('\n【判读】')
log('  · 这是按概率行动的第一次闭环：入场、止盈、止损、资金曲线都有')
log('  · 局限：路径用 cycles.csv 的 peak/peak_to_end 近似，不判断先触谁；未扣滑点')

out = os.path.join(OUT, 'trade_sim_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
