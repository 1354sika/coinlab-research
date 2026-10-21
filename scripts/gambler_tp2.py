# -*- coding: utf-8 -*-
"""gambler_tp2 · 用户策略检验：目标 +2% / 止损 -1% / 4h 超时 / 一天 15 次

问题：这个赔率结构（2:1）在什么胜率下才赚钱？真实数据里的胜率是多少？
    盈亏平衡胜率 = (SL + FEE) / (TP + SL) —— maker 0.04% 时约 34.7%，taker 0.20% 时约 40%
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
TP, SL = 0.02, 0.01
HOLD = 48          # 4h = 48 根 5m
FEE_M, FEE_T = 0.0004, 0.0020

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2 · 目标 +2% / 止损 -1% / 4h 超时（用户 2026-10-17 策略）')
log('=' * 108)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}  读盘 {time.time()-t0:.0f}s')

def scan(i, c, h, l, long_side=True):
    n = len(c); end = min(i + HOLD, n)
    if end - i < 3:
        return None
    e = c[i]
    for j in range(i + 1, end):
        if long_side:
            hs = l[j] <= e * (1 - SL); ht = h[j] >= e * (1 + TP)
        else:
            hs = h[j] >= e * (1 + SL); ht = l[j] <= e * (1 - TP)
        if hs and ht:
            return ('双触', -SL)
        if hs:
            return ('止损', -SL)
        if ht:
            return ('止盈', TP)
    r = c[end - 1] / e - 1.0
    if not long_side:
        r = -r
    return ('超时', r)

rng = np.random.default_rng(11)
SCEN = [
    ('A 随机时刻', lambda c, h, l, i: True),
    ('B 1h跌>=3%', lambda c, h, l, i: i >= 12 and c[i] / c[i - 12] - 1 <= -0.03),
    ('C 1h跌>=5%', lambda c, h, l, i: i >= 12 and c[i] / c[i - 12] - 1 <= -0.05),
    ('D 24h涨>=20%', lambda c, h, l, i: i >= 288 and c[i] / c[i - 288] - 1 >= 0.20),
    ('E 1h涨>=5%', lambda c, h, l, i: i >= 12 and c[i] / c[i - 12] - 1 >= 0.05),
]
RES = {n: [] for n, _ in SCEN}

for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < 400:
        continue
    idx = rng.integers(300, n - HOLD - 2, size=200)
    for i in idx:
        for name, f in SCEN[:1]:
            r = scan(i, c, h, l)
            if r:
                RES[name].append(r)
    # 条件场景：全量扫（样本少）
    for name, f in SCEN[1:]:
        for i in range(300, n - HOLD - 2, 3):
            try:
                if not f(c, h, l, i):
                    continue
            except Exception:
                continue
            r = scan(i, c, h, l)
            if r:
                RES[name].append(r)
            if len(RES[name]) >= 20000:
                break

log('')
log(f'{"场景":16s}{"笔数":>8s}{"止盈%":>8s}{"止损%":>8s}{"双触%":>7s}{"超时%":>7s}'
    f'{"毛期望%":>9s}{"净maker":>9s}{"净taker":>9s}')
log('-' * 108)
hdr = None
for name, _ in SCEN:
    rows = RES[name]
    if not rows:
        log(f'{name:16s}{"无样本":>8s}')
        continue
    kinds = np.array([x[0] for x in rows])
    rets = np.array([x[1] for x in rows])
    n = len(rows)
    tp = (kinds == '止盈').mean() * 100
    sl = (kinds == '止损').mean() * 100
    db = (kinds == '双触').mean() * 100
    to = (kinds == '超时').mean() * 100
    gross = rets.mean() * 100
    log(f'{name:16s}{n:>8,}{tp:>8.1f}{sl:>8.1f}{db:>7.1f}{to:>7.1f}'
        f'{gross:>9.2f}{gross-FEE_M*100:>9.2f}{gross-FEE_T*100:>9.2f}')

log('')
log('=' * 108)
log('【盈亏平衡胜率】')
log('=' * 108)
for fee, nm in [(FEE_M, 'maker 0.04%'), (FEE_T, 'taker 0.20%')]:
    be = (SL + fee) / (TP + SL)
    log(f'  {nm:12s}  p* = (SL+FEE)/(TP+SL) = ({SL}+{fee})/{TP+SL} = {be:.1%}')

log('')
log('【一天 15 次的可行性】')
log('=' * 108)
for name, _ in SCEN:
    n = len(RES[name])
    if n == 0:
        continue
    # 全部币合并后，按 3 年 1095 天折算（注意：样本是抽样，此数仅量级）
    log(f'  {name:16s} 抽样命中 {n:>6,} 笔（条件场景已限 2 万），无法直接折日频；看下方"触发密度"')
    break

out = os.path.join(OUT, 'tp2_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
