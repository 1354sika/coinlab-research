# -*- coding: utf-8 -*-
"""探数据：列名、时间范围、币种、每币根数（只读）"""
import pandas as pd, sys, time

P = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
t0 = time.time()
head = pd.read_csv(P, nrows=5)
print('COLUMNS:', list(head.columns))
print(head.to_string())
print()

# 全量只读 symbol/时间列
usecols = ['open_time'] if 'open_time' in head.columns else [head.columns[0]]
symcol = 'symbol' if 'symbol' in head.columns else (head.columns[1] if len(head.columns) > 1 else None)
cols = [symcol, head.columns[0]] if symcol else [head.columns[0]]
df = pd.read_csv(P, usecols=cols)
print('NROWS:', len(df))
print('SYMBOLS:', df[symcol].nunique() if symcol else '?')
print('SYM_SAMPLE:', sorted(df[symcol].unique())[:15] if symcol else '?')
ct = df.groupby(symcol).size() if symcol else None
print('PER_SYM_MIN/MAX:', ct.min() if ct is not None else '?', ct.max() if ct is not None else '?')
tc = head.columns[0]
print('TIME_MIN:', df[tc].min(), 'TIME_MAX:', df[tc].max())
print('ELAPSED %.1fs' % (time.time()-t0))
