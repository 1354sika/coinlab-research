# -*- coding: utf-8 -*-
"""gambler_tp2_grid · 用户策略的赔率网格：TP 与 SL 怎么配才可能为正

用户规则：目标 +2%，跌掉目标的 50%（-1%）止损，一天 15 次。
本脚本回答：这个赔率结构在什么参数下净期望为正？真实止盈率是多少？
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
HOLD = 48
FEE_M, FEE_T = 0.0004, 0.0020

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2_grid · 赔率网格（TP/SL 组合 × 场景）')
log('=' * 108)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}  读盘 {time.time()-t0:.0f}s')

def scan(i, c, h, l, tp, sl):
    n = len(c); end = min(i + HOLD, n)
    if end - i < 3:
        return None
    e = c[i]
    for j in range(i + 1, end):
        hs = l[j] <= e * (1 - sl); ht = h[j] >= e * (1 + tp)
        if hs and ht:
            return ('双触', -sl)
        if hs:
            return ('止损', -sl)
        if ht:
            return ('止盈', tp)
    return ('超时', c[end - 1] / e - 1.0)

rng = np.random.default_rng(23)
SCEN = [
    ('随机', lambda c, i: True),
    ('1h跌>=3%', lambda c, i: i >= 12 and c[i] / c[i - 12] - 1 <= -0.03),
    ('1h跌>=5%', lambda c, i: i >= 12 and c[i] / c[i - 12] - 1 <= -0.05),
    ('1h跌>=8%', lambda c, i: i >= 12 and c[i] / c[i - 12] - 1 <= -0.08),
]
GRID = [(0.01, 0.005), (0.015, 0.0075), (0.02, 0.01), (0.03, 0.015), (0.05, 0.025)]

# 采样点（每币每场景固定同一批点，便于对照）
POINTS = {}
for k, (sym, ot, c, h, l) in enumerate(RAW):
    n = len(c)
    if n < 400:
        continue
    idx = rng.integers(300, n - HOLD - 2, size=400)
    POINTS[sym] = (c, h, l, idx)

log('\n' + '=' * 108)
log(f'{"场景":12s}{"TP/SL":>12s}{"笔数":>8s}{"止盈%":>8s}{"止损%":>8s}{"超时%":>7s}'
    f'{"净maker%":>10s}{"净taker%":>10s}{"平衡胜率%":>10s}')
log('-' * 108)

best = []
for sname, cond in SCEN:
    for tp, sl in GRID:
        res = []
        for sym, (c, h, l, idx) in POINTS.items():
            if sname != '随机':
                for i in idx:
                    try:
                        ok = cond(c, int(i))
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                    r = scan(int(i), c, h, l, tp, sl)
                    if r:
                        res.append(r)
            else:
                for i in idx:
                    r = scan(int(i), c, h, l, tp, sl)
                    if r:
                        res.append(r)
        if not res:
            continue
        kinds = np.array([x[0] for x in res])
        rets = np.array([x[1] for x in res])
        n = len(res)
        tp_r = (kinds == '止盈').mean() * 100
        sl_r = (kinds == '止损').mean() * 100
        to_r = (kinds == '超时').mean() * 100
        gm = rets.mean() * 100
        be = (sl + FEE_M) / (tp + sl) * 100
        log(f'{sname:12s}{f"{tp:.1%}/{sl:.1%}":>12s}{n:>8,}{tp_r:>8.1f}{sl_r:>8.1f}{to_r:>7.1f}'
            f'{gm - FEE_M*100:>10.2f}{gm - FEE_T*100:>10.2f}{be:>10.1f}')
        best.append((gm - FEE_M * 100, sname, tp, sl, n, tp_r, be))

log('\n' + '=' * 108)
log('【净 maker 最高的 5 格】')
log('=' * 108)
for gm, sname, tp, sl, n, tp_r, be in sorted(best, reverse=True)[:5]:
    log(f'  {sname:12s} TP={tp:.1%} SL={sl:.1%}  n={n:>6,}  止盈率 {tp_r:.1f}%  '
        f'平衡胜率 {be:.1f}%  净maker {gm:+.3f}%')

out = os.path.join(OUT, 'tp2_grid_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
