# -*- coding: utf-8 -*-
"""gambler_cluster.py — 把"局成簇"（220x lift）变成可交易的筛选器
依据 selfbfly（job-47368-27，2,082,130 样本）：
    距上次起局 0~24h  → 后 24h 再起局率 51.72%（基准 0.234%）  **lift 220.6x**
    距上次起局 24~72h → 0.446% (1.90x)   72~168h → 0.121% (0.52x)   >168h → 0.014% (0.06x)

★ 关键认识：这 220x 是**筛选器**，不是普通特征。
  把它塞进全样本模型时 AUC 只从 0.7500 涨到 0.7534——因为 AUC 由 99.8% 的"无局"样本主导，
  稀有但极强的信号在排序指标里被淹没。**它的正确用法是：先缩小战场，再在里面交易。**

本脚本：只在"距上次起局 <=24h"的币/时刻入场，测**期望净收益**（扣 maker 0.04% + 滑点 0.05%）。
  对照：① 距上次起局 24~72h 入场  ② 距上次起局 >72h 入场  ③ 随机同频入场
  规则：+50% 止盈 / -15% 止损 / 24h(288根) 超时
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_cluster.txt')
WDAY = 288
GAME = 0.50
FEE = 0.0004 + 0.0005      # maker 0.04% + 滑点 0.05%（单边合计，双边由回测按次计）
TP = 0.50
SL = -0.15

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_cluster · 局成簇筛选器：只在"距上次起局<=24h"时交易，期望净收益如何？')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','quote_volume':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

def simulate(lo, hi, c, h, l, tp=TP, sl=SL, to=WDAY):
    """入场于 lo，检查 (lo, min(hi,lo+to)] 的价格路径"""
    tp_p = c[lo] * (1 + tp)
    sl_p = c[lo] * (1 + sl)
    end = min(hi, lo + to)
    if end <= lo + 1: return None
    seg_h = h[lo+1:end+1]; seg_l = l[lo+1:end+1]
    hit_tp = np.flatnonzero(seg_h >= tp_p)
    hit_sl = np.flatnonzero(seg_l <= sl_p)
    i_tp = hit_tp[0] if len(hit_tp) else 10**9
    i_sl = hit_sl[0] if len(hit_sl) else 10**9
    if i_tp == 10**9 and i_sl == 10**9:
        return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if i_tp <= i_sl:
        return tp - FEE, 'tp'
    return sl - FEE, 'sl'

BUCKETS = {}
def add(k, r, kind):
    if k not in BUCKETS: BUCKETS[k] = []
    BUCKETS[k].append((r, kind))

rng = np.random.default_rng(0)
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 3: continue
    hs = pd.Series(h)
    fmax = hs.rolling(WDAY, min_periods=1).max().shift(-WDAY).values
    up = fmax / c - 1.0
    # 逐个时刻标出"过去最近一次起局"（无前视：判断该时刻是否为起点，需其之后 288 根，t 时已知）
    last = -10 ** 9
    gap_arr = np.full(n, 10**9, dtype='int64')
    for t in range(n):
        gap_arr[t] = (t - last) if last > -10**8 else 10**9
        t0c = t - WDAY
        if t0c >= 0 and np.isfinite(up[t0c]) and up[t0c] >= GAME and t0c - last > WDAY:
            last = t0c
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r6 = np.full(n, np.nan); r6[6:] = c[6:] / c[:-6] - 1.0
    # 入场条件：还在早期（距低点 <20%）且未在拉升中（过去 6 根 <10%）
    cand = np.flatnonzero((gap_arr <= WDAY) & (over < 0.20) & (np.nan_to_num(r6, nan=0.0) < 0.10))
    last_i = -10 ** 9
    for t in cand:
        if t - last_i < 12: continue          # 同一币 1h 内不重复入场
        r = simulate(t, n - 1, c, h, l)
        if r is None: continue
        add('A 距上次起局<=24h', r[0], r[1]); last_i = t
    cand2 = np.flatnonzero((gap_arr > WDAY) & (gap_arr <= WDAY*3) & (over < 0.20) & (np.nan_to_num(r6, nan=0.0) < 0.10))
    last_i = -10 ** 9
    for t in cand2:
        if t - last_i < 12: continue
        r = simulate(t, n - 1, c, h, l)
        if r is None: continue
        add('B 距上次起局24~72h', r[0], r[1]); last_i = t
    cand3 = np.flatnonzero((gap_arr > WDAY*3) & (over < 0.20) & (np.nan_to_num(r6, nan=0.0) < 0.10))
    last_i = -10 ** 9
    for t in cand3[::37]:                      # 抽样，量大
        if t - last_i < 12: continue
        r = simulate(t, n - 1, c, h, l)
        if r is None: continue
        add('C 距上次起局>72h', r[0], r[1]); last_i = t
    # 随机对照（同频：与 A 组等量）
    na = len(BUCKETS.get('A 距上次起局<=24h', []))
    if na:
        pool = np.arange(WDAY*2, max(WDAY*2+1, n - WDAY))
        if len(pool):
            pick = rng.choice(pool, size=min(len(pool), max(200, len(cand))), replace=False)
            for t in pick:
                r = simulate(int(t), n - 1, c, h, l)
                if r is None: continue
                add('D 随机入场(对照)', r[0], r[1])

log('')
log('=' * 100)
log(f'【结果】规则：+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；已扣 {100*FEE:.2f}%（费+滑点）')
log('=' * 100)
log(f'  {"分组":<22} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
for k in ['A 距上次起局<=24h', 'B 距上次起局24~72h', 'C 距上次起局>72h', 'D 随机入场(对照)']:
    arr = BUCKETS.get(k, [])
    if not arr:
        log(f'  {k:<22} {0:>8}'); continue
    rs = np.array([x[0] for x in arr])
    kinds = [x[1] for x in arr]
    tp_p = 100.0 * np.mean([x == 'tp' for x in kinds])
    sl_p = 100.0 * np.mean([x == 'sl' for x in kinds])
    to_p = 100.0 * np.mean([x == 'timeout' for x in kinds])
    log(f'  {k:<22} {len(rs):>8,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% {tp_p:>6.1f}% {sl_p:>6.1f}% {to_p:>6.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若 A（距上次起局<=24h）期望净 显著 > D（随机），说明"局成簇"可直接变现')
log('  · B/C 是衰减对照：越久没起局，期望应越差')
log('  · 期望净为负 ⇒ 该筛选器还不够，需叠加更多条件（如量能、同步、位置）')

log('')
log('【A 组的止盈档位敏感性】')
for tp in (0.30, 0.50, 1.00):
    rs = []
    for (sym, ot, c, h, l, q) in RAW:
        n = len(c)
        if n < WDAY * 3: continue
        hs = pd.Series(h)
        fmax = hs.rolling(WDAY, min_periods=1).max().shift(-WDAY).values
        up = fmax / c - 1.0
        last = -10 ** 9
        gap_arr = np.full(n, 10**9, dtype='int64')
        for t in range(n):
            gap_arr[t] = (t - last) if last > -10**8 else 10**9
            t0c = t - WDAY
            if t0c >= 0 and np.isfinite(up[t0c]) and up[t0c] >= GAME and t0c - last > WDAY: last = t0c
        lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
        over = c / np.maximum(lo288, 1e-12) - 1.0
        r6 = np.full(n, np.nan); r6[6:] = c[6:] / c[:-6] - 1.0
        cand = np.flatnonzero((gap_arr <= WDAY) & (over < 0.20) & (np.nan_to_num(r6, nan=0.0) < 0.10))
        last_i = -10 ** 9
        for t in cand:
            if t - last_i < 12: continue
            r = simulate(t, n - 1, c, h, l, tp=tp)
            if r is None: continue
            rs.append(r[0]); last_i = t
        del hs; gc.collect()
    if rs:
        rs = np.array(rs)
        log(f'  止盈 {100*tp:>3.0f}%：笔 {len(rs):>7,}  期望净 {100*rs.mean():>7.3f}%  中位 {100*np.median(rs):>7.2f}%  胜率 {100.0*(rs>0).mean():>5.1f}%')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
