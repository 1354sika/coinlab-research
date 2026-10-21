# -*- coding: utf-8 -*-
"""gambler_bidir_trade · 双向交易期望：真信号能不能变成钱

bidir（job-24708-48）找到了两个方向明确的格子：
    A 过去 24h 跌 >=20%  → 未来 48h 涨30% 概率 18.96%，跌30% 仅 3.05%（6.2:1）
    B 过去 24h 涨 10~20% → 未来 48h 涨30% 概率 10.06%，跌30% 仅 0.59%（17.1:1）
    C 过去 4h  涨 >20%   → 涨 28.61% / 跌 23.50%（1.2:1，双向炸）

本脚本回答：这三个条件，做多/做空分别是什么期望净。

口径：
    做多 TP+30% / SL-15% / 48h 超时
    做空 TP+30%(价格跌30%) / SL-15%(价格涨15%) / 48h 超时
    扣费 0.09%
    先触及者为准（同一根内同时触及则算止损，保守）
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
log('gambler_bidir_trade · 双向交易期望（做多 / 做空）')
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


def scan(entry_idx, c, h, l, long_side):
    """从 entry_idx 出发扫 576 根，返回 (是否止盈, 收益, 持有根数)"""
    n = len(c)
    end = min(entry_idx + HOLD, n)
    e = c[entry_idx]
    if end - entry_idx < 10:
        return None
    for j in range(entry_idx + 1, end):
        if long_side:
            if l[j] <= e * (1 - SL):
                return (0, -SL - FEE, j - entry_idx)
            if h[j] >= e * (1 + TP):
                return (1, TP - FEE, j - entry_idx)
        else:
            # 做空：价格跌 30% 止盈；价格涨 15% 止损
            if h[j] >= e * (1 + SL):
                return (0, -SL - FEE, j - entry_idx)
            if l[j] <= e * (1 - TP):
                return (1, TP - FEE, j - entry_idx)
    r = (c[end - 1] / e - 1.0)
    if not long_side:
        r = -r
    return (1 if r > 0 else 0, r - FEE, end - 1 - entry_idx)


# 条件定义：(名字, 函数(indices) -> bool mask)
def cond_deep_drop(c, idx, lag=288, thr=-0.20):
    if idx < lag: return False
    return (c[idx] / c[idx - lag] - 1.0) <= thr

def cond_mild_up(c, idx, lag=288, lo=0.10, hi=0.20):
    if idx < lag: return False
    r = c[idx] / c[idx - lag] - 1.0
    return lo <= r < hi

def cond_hot4h(c, idx, lag=48, thr=0.20):
    if idx < lag: return False
    return (c[idx] / c[idx - lag] - 1.0) >= thr


CONDS = [
    ('A 24h跌>=20%', lambda c, i: cond_deep_drop(c, i)),
    ('B 24h涨10~20%', lambda c, i: cond_mild_up(c, i)),
    ('C 4h涨>=20%', lambda c, i: cond_hot4h(c, i)),
]

RES = {name: {'L': [], 'S': []} for name, _ in CONDS}
RES['RAND'] = {'L': [], 'S': []}

rng = np.random.default_rng(42)

for (sym, c, h, l) in RAWS:
    n = len(c)
    if n < 288 * 4 + HOLD + 2:
        continue
    t_cut = int(n * 0.70)
    cand = np.arange(300, n - HOLD - 1)
    # 抽样以控制时间（每币最多 400 个点）
    if len(cand) > 400:
        cand = rng.choice(cand, 400, replace=False)
        cand.sort()
    for i in cand:
        for name, fn in CONDS:
            if not fn(c, i):
                continue
            a = scan(i, c, h, l, True)
            b = scan(i, c, h, l, False)
            if a is None: continue
            key = 'te' if i >= t_cut else 'tr'
            RES[name]['L'].append((key, a[1], a[0]))
            RES[name]['S'].append((key, b[1], b[0]))
        # 随机对照
        if rng.random() < 0.02:
            a = scan(i, c, h, l, True)
            b = scan(i, c, h, l, False)
            if a is None: continue
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
log(f'{"条件":<18} {"方向":<6} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"平均持有h":>9}')
log('-' * 116)

for name in ['A 24h跌>=20%', 'B 24h涨10~20%', 'C 4h涨>=20%', 'RAND']:
    for side, lab in [('L', '做多'), ('S', '做空')]:
        arr = [x for x in RES[name][side] if x[0] == 'te']
        if len(arr) < 30:
            log(f'{name:<18} {lab:<6} {len(arr):>8,}  (样本不足)')
            continue
        rets = np.array([x[1] for x in arr]) * 100
        wins = np.array([x[2] for x in arr])
        log(f'{name:<18} {lab:<6} {len(arr):>8,} {rets.mean():>8.2f}% {np.median(rets):>7.2f}% {100*wins.mean():>6.1f}% {100*wins.mean():>6.1f}% {"":>9}')
    log('')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bidir_trade_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'报告已写 {os.path.join(OUT, "bidir_trade_summary.txt")}')
log(f'TOTAL {time.time()-t0:.0f}s')
