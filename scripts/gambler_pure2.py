# -*- coding: utf-8 -*-
"""gambler_pure2.py — 把"提纯规则"做成交易策略，测期望净收益
承 gambler_pure（job-47368-35）：提纯有效
    R1 volr24>=3                         真局率 1.310% (lift 2.30x)  覆盖 47.5%
    R2 volr24>=5                         真局率 2.033% (lift 3.57x)  覆盖 29.8%
    R3 volr24>=3 且 r12>=5%              真局率 3.077% (lift 5.40x)  覆盖  5.6%
    R5 volr24>=5 且 r12>=5% 且 同步>=2    真局率 2.987% (lift 5.24x)  覆盖  2.0%
  （基准真局率 0.570%；"市场同步"与"BTC"几乎无区分力，lift 0.5~1.3）

本脚本：对上述规则做**完整交易模拟**（入场无前视）：
    +50% 止盈 / -15% 止损 / 24h 超时 / 扣 0.09%
对照：同池随机入场。看"提纯"能否把期望净收益拉正。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_pure2.txt')
WDAY = 288
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_pure2 · 提纯规则 → 交易模拟（无前视入场）')
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
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    ok = np.isfinite(r12) & np.isfinite(volr24)
    base_cand = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    RULES = [
        ('R0 全部候选',                    base_cand),
        ('R1 volr24>=3',                   base_cand & (volr24 >= 3)),
        ('R2 volr24>=5',                   base_cand & (volr24 >= 5)),
        ('R3 volr24>=3 且 r12>=5%',         base_cand & (volr24 >= 3) & (r12 >= 0.05)),
        ('R3b volr24>=3 且 r12 5~15%',      base_cand & (volr24 >= 3) & (r12 >= 0.05) & (r12 < 0.15)),
        ('R4 volr24>=5 且 r12>=5%',         base_cand & (volr24 >= 5) & (r12 >= 0.05)),
        ('R5 volr24>=5 且 r12 5~15%',       base_cand & (volr24 >= 5) & (r12 >= 0.05) & (r12 < 0.15)),
        ('R6 volr24>=3 且 r288>=10%',       base_cand & (volr24 >= 3)),
    ]
    for nm, m in RULES:
        cand = np.flatnonzero(ok & m)
        last = -10 ** 9
        for t in cand:
            if t - last < 12: continue
            r = simulate(t, c, h, l)
            if r is None: continue
            add(nm, r[0], r[1]); last = t
    pool = np.flatnonzero(ok & base_cand)
    if len(pool) > 2000:
        pick = np.random.default_rng(2).choice(pool, size=min(len(pool), 3000), replace=False)
        for t in pick:
            r = simulate(int(t), c, h, l)
            if r is None: continue
            add('Z 候选池随机(对照)', r[0], r[1])

log('')
log('=' * 100)
log(f'【交易模拟】+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；扣 {100*FEE:.2f}%')
log('=' * 100)
log(f'  {"规则":<26} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
order = ['R0 全部候选', 'R1 volr24>=3', 'R2 volr24>=5', 'R3 volr24>=3 且 r12>=5%',
         'R3b volr24>=3 且 r12 5~15%', 'R4 volr24>=5 且 r12>=5%', 'R5 volr24>=5 且 r12 5~15%',
         'R6 volr24>=3 且 r288>=10%', 'Z 候选池随机(对照)']
for k in order:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<26} {0:>8}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<26} {len(rs):>8,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

log('')
log('=' * 100)
log('【止盈档位敏感性（对最优规则）】')
log('=' * 100)
for tp in (0.30, 0.50, 1.00):
    allr = []
    for (sym, ot, c, h, l, q) in RAW:
        n = len(c)
        if n < WDAY * 4: continue
        lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
        over = c / np.maximum(lo288, 1e-12) - 1.0
        r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
        base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
        volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
        volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
        if not (np.isfinite(r12).any() and np.isfinite(volr24).any()): continue
        m = (over >= 0.03) & (over <= 0.40) & (volr24 >= 3) & np.isfinite(r12) & (r12 >= 0.05)
        last = -10 ** 9
        for t in np.flatnonzero(np.nan_to_num(m, nan=0).astype(bool)):
            if t - last < 12: continue
            r = simulate(t, c, h, l, tp=tp)
            if r is None: continue
            allr.append(r[0]); last = t
    if allr:
        a = np.array(allr)
        log(f'  {100*tp:>3.0f}% 止盈：笔 {len(a):>7,}  期望净 {100*a.mean():>7.3f}%  '
            f'中位 {100*np.median(a):>7.2f}%  胜率 {100.0*(a>0).mean():>5.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 期望净 > 0 且 > Z(候选池随机) ⇒ 提纯能变现')
log('  · 全程无前视：over/volr24/r12 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
