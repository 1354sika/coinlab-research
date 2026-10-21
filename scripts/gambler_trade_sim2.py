# -*- coding: utf-8 -*-
"""gambler_trade_sim2 · 逐 bar 模拟：真正按规则「用命」

上一版（trade_sim）的三个口径错误：
  1. cumsum(ret) 不是资金曲线（累加收益率 ≠ 资金）
  2. 路径用 cycles.csv 的 peak/peak_to_end 事后近似，不知道先触止盈还是止损
  3. 最大回撤 0.00% 是假的（cumsum 单调递增）

本版：
  入场：涨 5%（close >= lo*1.05 的第一根 5m）
  止盈：high >= lo*1.575（相对入场 +50%）
  止损：low  <= lo*0.95（相对入场 -9.5%）
  逐 bar 检查谁先到；同一根内双触 → 保守算止损
  超时：288 根（24h）按收盘价出
  资金曲线：每笔固定 10% 仓位（不复利），并给复利版本对照
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
log('gambler_trade_sim2 · 逐 bar 模拟：入场→止盈/止损→资金曲线')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
trades = []   # (entry_time, ret, bars_held, exit_reason, v240)
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
            tp_px = lo_px * 1.575
            sl_px = lo_px * 0.95
            entry = c[j]
            ret = None; reason = 'timeout'; bars = 0
            for k in range(j + 1, min(j + 289, n)):
                bars = k - j
                hit_tp = h[k] >= tp_px
                hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl:
                    ret = sl_px / entry - 1; reason = 'both→SL'; break
                if hit_tp:
                    ret = tp_px / entry - 1; reason = 'TP'; break
                if hit_sl:
                    ret = sl_px / entry - 1; reason = 'SL'; break
            if ret is None:
                kk = min(j + 288, n - 1)
                ret = c[kk] / entry - 1; bars = kk - j
            trades.append((ot[j], ret, bars, reason, v240[j], sym))
    del ch; gc.collect()

tr = pd.DataFrame(trades, columns=['t','ret','bars','reason','v240','sym'])
tr = tr.sort_values('t').reset_index(drop=True)
log(f'\n可模拟周期 {len(tr):,}')

cut = np.median(tr['t'].values)
trm = tr['t'].values <= cut
tem = tr['t'].values > cut
log(f'训练 {trm.sum():,}  测试 {tem.sum():,}')

for q in [0.6, 0.7, 0.8, 0.9]:
    th = np.quantile(tr.loc[trm, 'v240'], q)
    sel = tem & (tr['v240'].values >= th)
    sub = tr[sel]
    if len(sub) == 0: continue
    r = sub['ret'].values
    # 固定 10% 仓位资金曲线
    eq = 1.0 + 0.1 * np.cumsum(r)
    dd = np.maximum.accumulate(eq) - eq
    # 复利 10% 仓位
    eqc = np.cumprod(1 + 0.1 * r)
    ddc = np.maximum.accumulate(eqc) - eqc
    log(f'\n【v240 >= p{q*100:.0f}】入场 {len(sub):,} 笔')
    log(f'  止盈占比 {(sub.reason=="TP").mean():>6.1%}   止损 {(sub.reason=="SL").mean():>6.1%}   '
        f'双触 {(sub.reason=="both→SL").mean():>5.1%}   超时 {(sub.reason=="timeout").mean():>5.1%}')
    log(f'  每笔均值 {r.mean():>+7.2%}  中位 {np.median(r):>+7.2%}  最差 {r.min():>+7.1%}')
    log(f'  固定10%仓位：最终资金 {eq[-1]:>7.2f}x  最大回撤 {dd.max():>7.3f}')
    log(f'  复利10%仓位：最终资金 {eqc[-1]:>7.2f}x  最大回撤 {ddc.max():>7.3f}')

log('\n【全样本对照：不挑 v240】')
r = tr.loc[tem, 'ret'].values
eq = 1.0 + 0.1 * np.cumsum(r)
log(f'  笔数 {len(r):,}  均值 {r.mean():>+7.2%}  固定10%最终 {eq[-1]:>7.2f}x  回撤 {(np.maximum.accumulate(eq)-eq).max():>7.3f}')

log('\n【判读】')
log('  · 这是逐 bar 判断「先触止盈还是先触止损」的真实模拟')
log('  · 未扣滑点；同一根内双触保守算止损')

out = os.path.join(OUT, 'trade_sim2_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
