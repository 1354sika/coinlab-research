# -*- coding: utf-8 -*-
"""gambler_tp2_1m · 用户策略在 1 分钟线上、全 502 币的检验

问题：
  1) 1m 上，投 1 块想赚到 +2%，止盈率是多少？（门槛仍是 34.7%）
  2) 全市场每天有多少次这样的机会？够不够“一天 15 次”？
  3) 换到 1m 粒度，止盈率会不会因为“看得更细”而提高？
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL = 0.02, 0.01
HOLD = 240            # 4 小时 = 240 根 1m
FEE_M, FEE_T = 0.0004, 0.0020
DAYS = 60

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2_1m · 1 分钟线 × 全 502 币：+2% 目标 / -1% 止损 / 4h 超时')
log('=' * 108)

SCEN = [
    ('全部时刻', None),
    ('1m跌>=1%', (1, -0.01)),
    ('5m跌>=2%', (5, -0.02)),
    ('15m跌>=3%', (15, -0.03)),
    ('30m跌>=5%', (30, -0.05)),
]
ACC = {n: dict(n=0, hit_tp=0, hit_sl=0, both=0, hit_any=0) for n, _ in SCEN}
n_sym = 0
t0_read = time.time()

for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64',
                             'close': 'float32', 'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        n = len(c)
        n_sym += 1
        fh = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values
        fl = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values
        ht = fh >= c * (1 + TP)
        hs = fl <= c * (1 - SL)
        any_ = ht | hs
        both = ht & hs
        for name, cond in SCEN:
            if cond is None:
                m = np.ones(n, dtype=bool)
            else:
                lag, thr = cond
                r = np.full(n, np.nan)
                r[lag:] = c[lag:] / c[:-lag] - 1.0
                m = r <= thr
            m = m & np.isfinite(fh) & np.isfinite(fl)
            a = ACC[name]
            a['n'] += int(m.sum())
            a['hit_tp'] += int((ht & m).sum())
            a['hit_sl'] += int((hs & m).sum())
            a['both'] += int((both & m).sum())
            a['hit_any'] += int((any_ & m).sum())
    del ch; gc.collect()
log(f'币 {n_sym}  读盘+统计 {time.time()-t0_read:.0f}s')

log('')
log(f'{"场景":14s}{"入场点数":>12s}{"触TP%":>8s}{"触SL%":>8s}{"双触%":>7s}'
    f'{"止盈率%":>9s}{"净maker%":>10s}{"净taker%":>10s}{"每天机会":>10s}')
log('-' * 108)
for name, _ in SCEN:
    a = ACC[name]
    if a['n'] == 0:
        continue
    n = a['n']
    tp_rate = a['hit_tp'] / n * 100
    sl_rate = a['hit_sl'] / n * 100
    dbl = a['both'] / n * 100
    # 保守：双触算止损
    win = (a['hit_tp'] - a['both']) / max(a['hit_any'], 1) * 100
    # 期望（按触碰口径，粗算）
    p_tp = (a['hit_tp'] - a['both']) / n
    p_sl = (a['hit_sl']) / n
    p_none = 1 - p_tp - p_sl
    gm = p_tp * TP - p_sl * SL + p_none * 0.0
    per_day = n / DAYS
    log(f'{name:14s}{n:>12,}{tp_rate:>8.1f}{sl_rate:>8.1f}{dbl:>7.1f}'
        f'{win:>9.1f}{gm*100-FEE_M*100:>10.3f}{gm*100-FEE_T*100:>10.3f}{per_day:>10,.0f}')

log('')
log('=' * 108)
log('【判读】')
log('=' * 108)
log('  · 止盈率 = (触 TP 且未触 SL) / (触 TP 或触 SL)，双触保守算止损')
log('  · 门槛仍是 34.7%（maker）/ 40%（taker）')
log('  · 每天机会 = 全部币的入场点数 / 60 天；若远小于 15，则“一天 15 次”物理上做不到')

out = os.path.join(OUT, 'tp2_1m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
