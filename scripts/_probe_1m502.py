# -*- coding: utf-8 -*-
"""_probe_1m502 · 查 klines_1m_502.csv 的时间跨度与币数"""
import pandas as pd
f = r'D:\中信证券\coinlab\data\klines_1m_502.csv'
mn = {}; mx = {}; cnt = 0
for ch in pd.read_csv(f, usecols=['symbol', 'open_time'], chunksize=2_000_000):
    cnt += len(ch)
    g = ch.groupby('symbol')['open_time'].agg(['min', 'max'])
    for s, r in g.iterrows():
        mn[s] = min(mn.get(s, 10**30), int(r['min']))
        mx[s] = max(mx.get(s, 0), int(r['max']))
print('rows', cnt, 'symbols', len(mn))
print('min', pd.Timestamp(min(mn.values()), unit='ms'))
print('max', pd.Timestamp(max(mx.values()), unit='ms'))
span = sorted((mx[s] - mn[s]) / 86400000 for s in mn)
print('span days: min %.0f  p25 %.0f  med %.0f  p75 %.0f  max %.0f'
      % (span[0], span[len(span)//4], span[len(span)//2], span[3*len(span)//4], span[-1]))
