# -*- coding: utf-8 -*-
"""核实 3 年 5m 数据是否拉完：行数 / 币数 / 时间跨度 / 每币 bar 数"""
import os, collections
import pandas as pd

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'data', 'klines_5m_3y.csv')
print('file =', p)
print('size_MB =', os.path.getsize(p) // 1048576)

d = pd.read_csv(p, usecols=['symbol', 'open_time'])
print('rows =', len(d))
print('symbols =', d.symbol.nunique())
print('min =', pd.to_datetime(d.open_time.min(), unit='ms', utc=True))
print('max =', pd.to_datetime(d.open_time.max(), unit='ms', utc=True))

c = collections.Counter(d.symbol)
v = sorted(c.values())
print('bars/symbol: min=%d median=%d max=%d' % (v[0], v[len(v) // 2], v[-1]))
print('预期满格 = 1095 天 * 288 = %d' % (1095 * 288))

# 覆盖不足的币
bad = [(s, n) for s, n in c.items() if n < 1095 * 288 * 0.9]
print('覆盖 <90%% 的币数 = %d' % len(bad))
for s, n in sorted(bad, key=lambda x: x[1])[:10]:
    print('   %-12s %d' % (s, n))
