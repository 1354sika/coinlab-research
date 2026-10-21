# -*- coding: utf-8 -*-
"""gambler_market_env · 市场环境统计

1) 1m 数据（60 币 1 年）按月：中位收益、正收益币占比 —— 判断测试段 2026-07~10 是熊是牛
2) 5m 3 年数据的时间范围 —— 判断有没有牛市段可测
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_market_env · 市场环境：1 年 1m 按月 + 5m 3 年时间范围')
log('=' * 100)

log('\n【一】1m 数据（60 币）按月市场环境')
d = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=['symbol', 'open_time', 'close'])
log(f'  行数 {len(d):,}  币 {d.symbol.nunique()}  耗时 {time.time()-t0:.0f}s')
d['ym'] = pd.to_datetime(d.open_time, unit='ms', utc=True).dt.strftime('%Y-%m')
g = d.groupby(['symbol', 'ym'])['close'].agg(['first', 'last'])
g['ret'] = g['last'] / g['first'] - 1
m = g.groupby('ym')['ret'].agg(['median', 'mean', lambda x: (x > 0).mean(), 'count'])
m.columns = ['中位%', '均值%', '正收益占比%', '币数']
log(f"\n{'月份':<10}{'中位%':>9}{'均值%':>9}{'正收益占比%':>12}{'币数':>6}")
for ym, row in m.iterrows():
    log(f"{ym:<10}{row['中位%']*100:>9.1f}{row['均值%']*100:>9.1f}{row['正收益占比%']*100:>12.1f}{int(row['币数']):>6}")

log('')
log('【二】5m 3 年数据的时间范围（找牛市段）')
try:
    d5 = pd.read_csv(os.path.join(DATA, 'klines_5m_3y.csv'), usecols=['symbol', 'open_time', 'close'])
    log(f'  行数 {len(d5):,}  币 {d5.symbol.nunique()}  耗时 {time.time()-t0:.0f}s')
    d5['ym'] = pd.to_datetime(d5.open_time, unit='ms', utc=True).dt.strftime('%Y-%m')
    g5 = d5.groupby(['symbol', 'ym'])['close'].agg(['first', 'last'])
    g5['ret'] = g5['last'] / g5['first'] - 1
    m5 = g5.groupby('ym')['ret'].agg(['median', 'mean', lambda x: (x > 0).mean()])
    m5.columns = ['中位%', '均值%', '正收益占比%']
    log(f"\n{'月份':<10}{'中位%':>9}{'均值%':>9}{'正收益占比%':>12}")
    for ym, row in m5.iterrows():
        log(f"{ym:<10}{row['中位%']*100:>9.1f}{row['均值%']*100:>9.1f}{row['正收益占比%']*100:>12.1f}")
    log(f'\n  牛市月（正收益占比 >= 70%）:')
    for ym, row in m5.iterrows():
        if row['正收益占比%'] >= 0.7:
            log(f'    {ym}  中位 {row["中位%"]*100:>+.1f}%  正收益占比 {row["正收益占比%"]*100:.0f}%')
except Exception as e:
    log(f'  5m 数据读取失败：{e}')

log('')
log('=' * 100)
log('【判读】')
log('  · 测试段 2026-07~10 若中位为正、正收益占比 >50% ⇒ 测试段不是熊市，是反弹/牛市')
log('  · 5m 3 年里的牛市月 = 可以拿去测「牛市是否有效」的时段')

out = os.path.join(OUT, 'market_env_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
