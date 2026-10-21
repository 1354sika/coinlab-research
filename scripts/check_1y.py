# -*- coding: utf-8 -*-
"""校验 klines_1m_1y.csv（60 币 x 365 天 1m）完整性"""
import os, datetime as dt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
F = os.path.join(ROOT, "data", "klines_1m_1y.csv")
print('file = %s' % F)
print('size = %.2f GB' % (os.path.getsize(F) / 1e9), flush=True)
d = pd.read_csv(F, usecols=['symbol', 'open_time', 'close'])
print('rows = %d   symbols = %d' % (len(d), d.symbol.nunique()), flush=True)
print('span: %s -> %s' % (
    dt.datetime.utcfromtimestamp(d.open_time.min() / 1000).strftime('%Y-%m-%d'),
    dt.datetime.utcfromtimestamp(d.open_time.max() / 1000).strftime('%Y-%m-%d')))
print('dup(open_time,symbol) = %d' % int(d.duplicated(['symbol', 'open_time']).sum()))
g = d.groupby('symbol').size().sort_values()
print('bars/symbol: min=%d  median=%d  max=%d  (目标 525600)' % (g.min(), g.median(), g.max()))
print('--- 覆盖最少的 8 个币 ---')
print(g.head(8).to_string())
print('--- 覆盖最多的 3 个币 ---')
print(g.tail(3).to_string())
print('--- 前 60 行样本 ---')
print(d.head(3).to_string())
