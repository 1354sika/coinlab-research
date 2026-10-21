# -*- coding: utf-8 -*-
"""统计现有数据资源：每个文件多少币、多少行、覆盖多长时间。只读。"""
import os, io, sys, datetime as dt
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

D = r'D:\中信证券\coinlab\data'
FILES = [
    'klines_1m_1y.csv', 'klines_1m_502.csv', 'klines_1m_sync.csv',
    'klines_1m_wide.csv', 'klines_1m_long.csv', 'klines_1m_top20_7d.csv',
    'klines_5m_3y.csv', 'klines_5m_502.csv',
    'klines_1h_top20_365d.csv',
    'klines_1d_full.csv', 'klines_1d_top50.csv', 'klines_1d_top502.csv',
]

print('=' * 92)
print('现有数据资源清单（只读统计）')
print('=' * 92)
tot_gb = 0.0
for f in FILES:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f'{f:28s}  不存在')
        continue
    gb = os.path.getsize(p) / 1e9
    tot_gb += gb
    syms, n, t0, t1 = set(), 0, None, None
    try:
        for ch in pd.read_csv(p, usecols=lambda c: c in ('symbol', 'open_time'),
                              chunksize=3_000_000):
            n += len(ch)
            if 'symbol' in ch.columns:
                syms |= set(ch['symbol'].dropna().unique())
            if 'open_time' in ch.columns:
                if t0 is None:
                    t0 = int(ch['open_time'].iloc[0])
                t1 = int(ch['open_time'].iloc[-1])
    except Exception as e:
        print(f'{f:28s}  {gb:6.2f}GB  读失败: {type(e).__name__}: {e}')
        continue
    d0 = dt.datetime.utcfromtimestamp(t0 / 1000).strftime('%Y-%m-%d') if t0 else '?'
    d1 = dt.datetime.utcfromtimestamp(t1 / 1000).strftime('%Y-%m-%d') if t1 else '?'
    span = (t1 - t0) / 86400000.0 if (t0 and t1) else 0
    print(f'{f:28s}  {gb:6.2f}GB  {n:>12,} 行  {len(syms):>4} 币  {d0} ~ {d1}  ({span:.0f} 天)')

print('-' * 92)
print(f'合计 {tot_gb:.1f} GB')

# E 盘下载进度
E = r'E:\binance\klines_1m'
if os.path.isdir(E):
    fs = [x for x in os.listdir(E) if x.endswith('.csv')]
    sz = sum(os.path.getsize(os.path.join(E, x)) for x in fs) / 1e9
    print(f'E 盘 klines_1m: {len(fs)} 个文件, {sz:.1f} GB（目标 502）')
else:
    print(f'E 盘 {E} 尚不存在')

# live 数据
L = os.path.join(D, 'live')
if os.path.isdir(L):
    for day in sorted(os.listdir(L)):
        dp = os.path.join(L, day)
        if not os.path.isdir(dp):
            continue
        items = []
        for x in os.listdir(dp):
            items.append(f'{x}({os.path.getsize(os.path.join(dp, x))/1e6:.0f}MB)')
        print(f'live/{day}: ' + ' '.join(items))
