# -*- coding: utf-8 -*-
"""gambler_inter2.py — 最强交互组合 → 交易模拟（交互能不能变现）
承 gambler_inter（job-47368-40）：
  单格最高：r12 5~10% × r288>=30%  → 真局率 10.45%（18.34x），样本 507
  三阶最强：T4 volr24>=5 & r288>=30% & over>=15% → 7.655%（13.43x），2,652 样本
  多数格子标"冗余"⇒ 这些特征高度共线，超加性只在少数组合出现。

本脚本对最强组合做交易模拟（无前视入场）：
    +50% 止盈 / -15% 止损 / 24h 超时 / 扣 0.09%
对照：候选池随机。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_inter2.txt')
WDAY = 288
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_inter2 · 最强交互组合 → 交易模拟')
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
    ht = np.flatnonzero(h[lo+1:end+1] >= tp_p)
    hs = np.flatnonzero(l[lo+1:end+1] <= sl_p)
    it = ht[0] if len(ht) else 10**9
    isl = hs[0] if len(hs) else 10**9
    if it == 10**9 and isl == 10**9: return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if it <= isl: return tp - FEE, 'tp'
    return sl - FEE, 'sl'

B = {}
def add(k, r, kind):
    B.setdefault(k, []).append((r, kind))

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    ok = np.isfinite(r12) & np.isfinite(volr24) & np.isfinite(r288)
    base = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    RULES = [
        ('R0 候选池',                          base),
        ('T1 volr24>=5 & r12>=5% & over>=15%',  base & (volr24 >= 5) & (r12 >= 0.05) & (over >= 0.15)),
        ('T3 volr24>=10 & r12>=5% & over>=15%', base & (volr24 >= 10) & (r12 >= 0.05) & (over >= 0.15)),
        ('T4 volr24>=5 & r288>=10% & over>=15%', base & (volr24 >= 5) & (r288 >= 0.10) & (over >= 0.15)),
        ('T4b volr24>=5 & r288>=30% & over>=15%', base & (volr24 >= 5) & (r288 >= 0.30) & (over >= 0.15)),
        ('T6 r12 5~10% & r288>=30%',            base & (r12 >= 0.05) & (r12 < 0.10) & (r288 >= 0.30)),
        ('T7 volr24>=10 & r288>=30%',           base & (volr24 >= 10) & (r288 >= 0.30)),
        ('T8 volr24>=5 & r12>=5% & r288>=30% & over>=15%',
         base & (volr24 >= 5) & (r12 >= 0.05) & (r288 >= 0.30) & (over >= 0.15)),
    ]
    for nm, m in RULES:
        cand = np.flatnonzero(ok & m)
        last = -10 ** 9
        for t in cand:
            if t - last < 12: continue
            r = simulate(t, c, h, l)
            if r is None: continue
            add(nm, r[0], r[1]); last = t
    pool = np.flatnonzero(ok & base)
    if len(pool) > 2000:
        pick = np.random.default_rng(3).choice(pool, size=min(len(pool), 3000), replace=False)
        for t in pick:
            r = simulate(int(t), c, h, l)
            if r is None: continue
            add('Z 候选池随机(对照)', r[0], r[1])

log('')
log('=' * 100)
log(f'【交易模拟】+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；扣 {100*FEE:.2f}%')
log('=' * 100)
log(f'  {"规则":<42} {"笔数":>7} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
order = ['R0 候选池', 'T1 volr24>=5 & r12>=5% & over>=15%', 'T3 volr24>=10 & r12>=5% & over>=15%',
         'T4 volr24>=5 & r288>=10% & over>=15%', 'T4b volr24>=5 & r288>=30% & over>=15%',
         'T6 r12 5~10% & r288>=30%', 'T7 volr24>=10 & r288>=30%',
         'T8 volr24>=5 & r12>=5% & r288>=30% & over>=15%', 'Z 候选池随机(对照)']
for k in order:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<42} {0:>7}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<42} {len(rs):>7,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 期望净 > 0 且 > Z ⇒ 交互组合可交易；若仍 ≤ 0.5% ⇒ 交互提升的是"识别"，不是"收益"')
log('  · 全程无前视：volr24/r12/r288/over 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
