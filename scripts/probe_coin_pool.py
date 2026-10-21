# -*- coding: utf-8 -*-
"""自查：我"剔除僵尸币"时，是不是把"操盘局"一起扔掉了？

背景：gambler_denoise.py 里我把 304 币清洗成 164 币（剔除 140 个"低活跃"币）。
      但操盘手最可能活动的池子恰恰是**小币/冷门币** —— 这可能是方向性错误。
本脚本检查：pump_hunt_events.csv 里的 820 个"局"，落在"保留币"还是"被剔除币"上。
"""
import os
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'

print('读 symbol/close/quote_volume ...', flush=True)
df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'close', 'quote_volume'],
                 dtype={'symbol': 'category'})
print('行数', len(df), flush=True)
g = df.groupby('symbol', observed=True).agg(
    n=('close', 'size'),
    nz=('quote_volume', lambda s: int((s > 0).sum())),
    qv=('quote_volume', 'sum'),
)
del df
T = int(g['n'].max())
valid_ratio = g['n'] / T                     # 有效行占比（近似"有数据占比"）
zero_ratio = 1.0 - g['nz'] / g['n']          # 零成交占比
keep = (valid_ratio > 0.80) & (zero_ratio < 0.50) & (g['qv'] > 0)
print('总币数 %d ；保留 %d ；剔除 %d' % (len(g), int(keep.sum()), int((~keep).sum())))

ev = pd.read_csv(os.path.join(EV, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)
keep_set = set(g.index[keep].astype(str))
ev['retained'] = ev['symbol'].isin(keep_set)
print('\n事件总数 %d ；涉及币 %d' % (len(ev), ev['symbol'].nunique()))
for thr in sorted(ev['thr'].unique()):
    s = ev[ev['thr'] == thr]
    print('  门槛 +%d%%：事件 %d，落在"保留币"上 %d（%.1f%%），落在"被剔除币"上 %d（%.1f%%）'
          % (int(thr * 100), len(s), int(s['retained'].sum()),
             100 * s['retained'].mean(), int((~s['retained']).sum()),
             100 * (1 - s['retained'].mean())))
print('\n事件涉及币的成交额分位（被剔除的币在事件里占多少）')
evc = ev.groupby('symbol').size().rename('events').to_frame().join(g[['qv']], how='left')
evc['keep'] = evc.index.isin(keep_set)
print('  被剔除币承担的事件数 = %d / %d' % (int(evc.loc[~evc['keep'], 'events'].sum()), len(ev)))
print('  全市场成交额中位（保留币） = %.3e' % g.loc[keep, 'qv'].median())
print('  事件涉及币的成交额中位   = %.3e' % evc['qv'].median())
print('  事件涉及币里，成交额低于"保留币中位"的占 %.1f%%'
      % (100 * (evc['qv'] < g.loc[keep, 'qv'].median()).mean()))
