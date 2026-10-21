# -*- coding: utf-8 -*-
"""gambler_early4.py — 分档交易模拟：over288 各档位下，真正能赚到多少？
承 early3 的发现（job-47368-33，672,744 样本）：
    早期(over 5~20%)  P(未来24h再涨>=50%) = 0.39%
    中段(over 20~50%)  → 3.29%
    已大涨(over >50%)  → 5.63%
  ⇒ "早期"不是好机会，动量反而更强。但"未来最大涨幅"≠"能赚到的钱"（可能先深跌再涨）。

本脚本对 over288 分档 + 放量条件做**完整交易模拟**（无前视入场）：
    +50% 止盈 / -15% 止损 / 24h 超时 / 扣 0.09%
对照：随机同池入场。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_early4.txt')
WDAY = 288
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_early4 · 分档交易模拟（over288 各档，无前视入场）')
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
    if n < WDAY * 3: continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    ok = np.isfinite(r12) & np.isfinite(volr24)
    GROUPS = [
        ('A 早期 over5~20%',              (over >= 0.05) & (over < 0.20)),
        ('B 中段 over20~50%',             (over >= 0.20) & (over < 0.50)),
        ('C 已大涨 over50~100%',          (over >= 0.50) & (over < 1.00)),
        ('D 暴涨 over>100%',              (over >= 1.00)),
        ('E 早期+放量',                    (over >= 0.05) & (over < 0.20) & (volr24 >= 2)),
        ('F 早期+放量+1h涨5~15%',          (over >= 0.05) & (over < 0.20) & (volr24 >= 2) & (r12 >= 0.05) & (r12 < 0.15)),
        ('G 中段+放量',                    (over >= 0.20) & (over < 0.50) & (volr24 >= 2)),
        ('H 已大涨+放量',                  (over >= 0.50) & (volr24 >= 2)),
    ]
    for nm, m in GROUPS:
        cand = np.flatnonzero(ok & m)
        last = -10 ** 9
        for t in cand:
            if t - last < 12: continue
            r = simulate(t, c, h, l)
            if r is None: continue
            add(nm, r[0], r[1]); last = t
    # 随机对照
    pool = np.flatnonzero(ok)
    if len(pool) > 2000:
        pick = np.random.default_rng(1).choice(pool, size=3000, replace=False)
        for t in pick:
            r = simulate(int(t), c, h, l)
            if r is None: continue
            add('Z 随机入场(对照)', r[0], r[1])

log('')
log('=' * 100)
log(f'【交易模拟】+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；扣 {100*FEE:.2f}%')
log('=' * 100)
log(f'  {"分组":<24} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
order = ['A 早期 over5~20%', 'B 中段 over20~50%', 'C 已大涨 over50~100%', 'D 暴涨 over>100%',
         'E 早期+放量', 'F 早期+放量+1h涨5~15%', 'G 中段+放量', 'H 已大涨+放量', 'Z 随机入场(对照)']
for k in order:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<24} {0:>8}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<24} {len(rs):>8,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 期望净 > 0 且显著高于"Z 随机"⇒ 该档位可交易')
log('  · 若"早期"档期望净最低 ⇒ 追早期是错的，动量档才是主战场')
log('  · 全程无前视：over/volr24/r12 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
