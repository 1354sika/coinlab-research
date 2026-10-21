# -*- coding: utf-8 -*-
"""gambler_bidir_trade2 · 双向交易期望（修正抽样：先筛条件再抽）

上一版（job-24708-49）致命缺陷：在全部时间点上随机抽 400 个点，
稀有条件（24h跌>=20%、4h涨>=20%）几乎抽不到 → 测试段只剩 7 笔 / 3 笔。

本版：**先按条件筛出全部命中点，再在其中抽样**，保证样本量。

口径不变：
    做多 TP+30% / SL-15% / 48h 超时
    做空 TP+30%(价格跌30%) / SL-15%(价格涨15%) / 48h 超时
    扣费 0.09%，同根内先触及者为准（同根同时触及算止损，保守）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
HOLD = 576
TP = 0.30; SL = 0.15; FEE = 0.0009

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_bidir_trade2 · 双向交易期望（先筛条件再抽样）')
log('=' * 116)

RAWS = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAWS.append((sym, g['close'].values.astype('float64'),
                     g['high'].values.astype('float64'),
                     g['low'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAWS)}   读盘 {time.time()-t0:.0f}s')


def scan(i, c, h, l, long_side):
    n = len(c)
    end = min(i + HOLD, n)
    if end - i < 10:
        return None
    e = c[i]
    for j in range(i + 1, end):
        if long_side:
            if l[j] <= e * (1 - SL):
                return (0, -SL - FEE)
            if h[j] >= e * (1 + TP):
                return (1, TP - FEE)
        else:
            if h[j] >= e * (1 + SL):
                return (0, -SL - FEE)
            if l[j] <= e * (1 - TP):
                return (1, TP - FEE)
    r = c[end - 1] / e - 1.0
    if not long_side:
        r = -r
    return (1 if r > 0 else 0, r - FEE)


# (名字, lag, lo, hi)  —— 半开区间 [lo, hi)
CONDS = [
    ('A 24h跌>=20%',   288, -np.inf, -0.20),
    ('B 24h涨10~20%',  288,  0.10,   0.20),
    ('C 4h涨>=20%',     48,  0.20,   np.inf),
    ('D 24h跌10~20%',  288, -0.20,  -0.10),
    ('E 1h涨10~20%',    12,  0.10,   0.20),
]

RES = {n: {'L': [], 'S': []} for n, *_ in CONDS}
RES['RAND'] = {'L': [], 'S': []}
rng = np.random.default_rng(7)
CAP = 3000        # 每条件每方向采样上限

for (sym, c, h, l) in RAWS:
    n = len(c)
    if n < 288 * 4 + HOLD + 2:
        continue
    t_cut = int(n * 0.70)
    lo_i = 300; hi_i = n - HOLD - 1
    if hi_i <= lo_i:
        continue
    idx_all = np.arange(lo_i, hi_i)
    for (name, lag, lo, hi) in CONDS:
        with np.errstate(invalid='ignore', divide='ignore'):
            r = c / np.roll(c, lag) - 1.0
        m = (r >= lo) & (r < hi)
        hit = idx_all[m[idx_all]]
        if len(hit) == 0:
            continue
        if len(hit) > CAP:
            hit = rng.choice(hit, CAP, replace=False)
        for i in hit:
            a = scan(i, c, h, l, True)
            if a is None:
                continue
            b = scan(i, c, h, l, False)
            key = 'te' if i >= t_cut else 'tr'
            RES[name]['L'].append((key, a[1], a[0]))
            RES[name]['S'].append((key, b[1], b[0]))
    # 随机对照
    for i in rng.choice(idx_all, min(800, len(idx_all)), replace=False):
        a = scan(i, c, h, l, True)
        if a is None:
            continue
        b = scan(i, c, h, l, False)
        key = 'te' if i >= t_cut else 'tr'
        RES['RAND']['L'].append((key, a[1], a[0]))
        RES['RAND']['S'].append((key, b[1], b[0]))
    del c, h, l
    gc.collect()

log(f'扫描完成 {time.time()-t0:.0f}s')

log('')
log('=' * 116)
log('【结果】测试段（后 30%），扣费 0.09%，先触及者为准')
log('=' * 116)
log(f'{"条件":<18} {"方向":<6} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7}')
log('-' * 116)
for name in [n for n, *_ in CONDS] + ['RAND']:
    for side, lab in [('L', '做多'), ('S', '做空')]:
        arr = [x for x in RES[name][side] if x[0] == 'te']
        if len(arr) < 50:
            lo_ = len(arr)
            log(f'{name:<18} {lab:<6} {lo_:>8,}  (样本不足)')
            continue
        rets = np.array([x[1] for x in arr]) * 100
        wins = np.array([x[2] for x in arr])
        log(f'{name:<18} {lab:<6} {len(arr):>8,} {rets.mean():>8.2f}% {np.median(rets):>7.2f}% {100*wins.mean():>6.1f}%')
    log('')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bidir_trade2_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'报告已写 {os.path.join(OUT, "bidir_trade2_summary.txt")}')
log(f'TOTAL {time.time()-t0:.0f}s')
