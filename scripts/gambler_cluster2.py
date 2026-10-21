# -*- coding: utf-8 -*-
"""gambler_cluster2.py — 成簇筛选器（实时信号版，严格无前视）
修前视后（job-47368-31）发现的硬事实：
  用"未来 288 根涨>=50%"来定义"起点"⇒ 该起点要等 288 根之后才能确认
  ⇒ 任何"距上次起局 <=24h"的入场条件在真实时间轴上都**永不成立**（A 组 0 笔）
  ⇒ selfbfly 的 220x lift 同样是**事后视角**的产物，不可直接变现。

本版改用**实时可判定的启动信号**（t 时刻全部可见，无任何未来信息）：
    sig[t] = (过去 R1 根涨幅 >= G1) 且 (量比 >= V1) 且 (当前价距 288 根低点 <= H1)
  再测："24h 内刚出现过启信号"的币，后续期望净收益如何。
对照：距上次信号 24~72h / >72h / 随机入场。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_cluster2.txt')
WDAY = 288
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

R1G, G1, V1, H1 = 12, 0.20, 3.0, 0.50    # 启信号定义：1h 涨>=20% 且 量比>=3 且 距低点<=50%

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log(f'gambler_cluster2 · 成簇筛选器（实时信号版）')
log(f'  启信号 = 过去{R1G}根涨>={100*G1:.0f}% 且 量比>={V1} 且 距288根低点<={100*H1:.0f}%')
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

def simulate(lo, c, h, l, tp=TP, sl=SL, to=TO):
    n = len(c)
    end = min(n - 1, lo + to)
    if end <= lo + 1: return None
    tp_p = c[lo] * (1 + tp); sl_p = c[lo] * (1 + sl)
    sh = h[lo+1:end+1]; sl_ = l[lo+1:end+1]
    ht = np.flatnonzero(sh >= tp_p); hs = np.flatnonzero(sl_ <= sl_p)
    it = ht[0] if len(ht) else 10**9
    isl = hs[0] if len(hs) else 10**9
    if it == 10**9 and isl == 10**9: return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if it <= isl: return tp - FEE, 'tp'
    return sl - FEE, 'sl'

B = {}
def add(k, r, kind):
    B.setdefault(k, []).append((r, kind))

sig_total = 0
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 3: continue
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    rR = np.full(n, np.nan); rR[R1G:] = c[R1G:] / c[:-R1G] - 1.0
    mv = pd.Series(q).rolling(WDAY * 7, min_periods=WDAY).mean().values   # 7 日均量做基准
    volr = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    sig = np.nan_to_num(rR, nan=-9) >= G1
    sig &= np.nan_to_num(volr, nan=0.0) >= V1
    sig &= over <= H1
    sig_total += int(sig.sum())
    last = -10 ** 9
    gap = np.full(n, 10**9, dtype='int64')
    for t in range(n):
        gap[t] = (t - last) if last > -10**8 else 10**9
        if sig[t] and t - last > R1G: last = t        # 实时信号，t 时刻即可确认 ✓
    early = (over < 0.20) & (np.nan_to_num(rR, nan=0.0) < 0.10)   # 还在早期、未在拉升中
    for lo, hi, k in ((0, WDAY, 'A 距启信号<=24h'),
                      (WDAY, WDAY * 3, 'B 距启信号24~72h'),
                      (WDAY * 3, 10**9, 'C 距启信号>72h')):
        cand = np.flatnonzero((gap > lo) & (gap <= hi) & early)
        last_i = -10 ** 9
        step = 1 if hi <= WDAY * 3 else 37
        for t in cand[::step]:
            if t - last_i < 12: continue
            r = simulate(t, c, h, l)
            if r is None: continue
            add(k, r[0], r[1]); last_i = t
    # 随机对照
    pool = np.arange(WDAY * 3, n - WDAY)
    if len(pool) > 200:
        pick = np.random.default_rng(0).choice(pool, size=min(len(pool), 2000), replace=False)
        for t in pick:
            r = simulate(int(t), c, h, l)
            if r is None: continue
            add('D 随机入场(对照)', r[0], r[1])

log(f'启信号总数 {sig_total:,}')
log('')
log('=' * 100)
log(f'【结果】+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；已扣 {100*FEE:.2f}%')
log('=' * 100)
log(f'  {"分组":<22} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
for k in ['A 距启信号<=24h', 'B 距启信号24~72h', 'C 距启信号>72h', 'D 随机入场(对照)']:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<22} {0:>8}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<22} {len(rs):>8,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 全部特征在 t 时刻可见（实时信号），无前视')
log('  · A > D ⇒ "刚启动过的币更容易再启动"这一成簇效应可交易')
log('  · A ≈ D 或 A < 0 ⇒ 成簇效应在实时口径下不足以赚钱')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
