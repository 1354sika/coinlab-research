# -*- coding: utf-8 -*-
"""gambler_regime_count.py — 只统计"局"的数量：不同定义下差多少
目的：把"怎么找局"这件事摆到桌面上——定义越严，样本越少；先看清代价再定。
不训练，只数数。
"""
import os, gc, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_regime_count.txt')
N_COIN_LIMIT = 150

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

DEFS = [
    ('24h内+50% 且 之后回撤30%', 288, 0.50, 0.30),
    ('24h内+50% 且 之后回撤50%', 288, 0.50, 0.50),
    ('24h内+100% 且 之后回撤40%', 288, 1.00, 0.40),
    ('24h内+150% 且 之后回撤40%', 288, 1.50, 0.40),
    ('4h内+50%  且 之后回撤30%', 48, 0.50, 0.30),
    ('4h内+100% 且 之后回撤40%', 48, 1.00, 0.40),
    ('8h内+50%  且 之后回撤30%', 96, 0.50, 0.30),
    ('24h内+50%（不看崩）', 288, 0.50, None),
    ('24h内+100%（不看崩）', 288, 1.00, None),
]

log('=' * 100)
log('gambler_regime_count · 不同"局"定义下的样本量')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32', 'low': 'float32'}
parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=1_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:N_COIN_LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= N_COIN_LIMIT:
                break
            seen = seen + new[:N_COIN_LIMIT - len(seen)]
    parts.append(ch[ch['symbol'].isin(seen)])
df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
df['symbol'] = df['symbol'].astype('category')
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0t:.0f}s')

res = {name: [] for name, *_ in DEFS}
all_days = set()
for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 2000:
        continue
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    ot = g['open_time'].values
    Wl = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= Wl.min(axis=1) + 1e-12
    cand = np.flatnonzero(is_low)
    hmax288 = pd.Series(h).rolling(288, min_periods=1).max().shift(-288).values
    hmax48 = pd.Series(h).rolling(48, min_periods=1).max().shift(-48).values
    hmax96 = pd.Series(h).rolling(96, min_periods=1).max().shift(-96).values
    lmin48 = pd.Series(l).rolling(48, min_periods=1).min().shift(-48).values
    for name, W, G, DD in DEFS:
        hmax = {288: hmax288, 48: hmax48, 96: hmax96}[W]
        last = -10 ** 9
        for t0 in cand:
            if t0 - last < W:
                continue
            H = hmax[t0]
            if not np.isfinite(H) or H / max(l[t0], 1e-12) - 1.0 < G:
                continue
            i1 = t0 + int(np.argmax(h[t0:t0 + W]))
            if DD is not None:
                k2 = min(n - 1, i1 + 48)
                if k2 > i1 and l[i1:k2 + 1].min() > H * (1 - DD):
                    continue
            res[name].append(int(ot[t0]))
            all_days.add(pd.to_datetime(int(ot[t0]), unit='ms').date().isoformat())
            last = i1

log('')
log('=' * 100)
log('【各定义下的局数量】（150 币 / 3 年 5m）')
log('=' * 100)
log(f'{"定义":30s} {"局数":>7s} {"币数":>6s} {"按年分布"}')
for name, *_ in DEFS:
    ts = res[name]
    if not ts:
        log(f'{name:30s} {0:>7d}')
        continue
    yrs = pd.to_datetime(ts, unit='ms').year
    vc = pd.Series(yrs).value_counts().sort_index()
    log(f'{name:30s} {len(ts):>7d} {"":>6s} ' + '  '.join(f'{y}:{v}' for y, v in vc.items()))

log('')
log(f'全样本跨度内"任一定义命中"的天数：{len(all_days)} 天')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
