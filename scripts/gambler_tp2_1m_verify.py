# -*- coding: utf-8 -*-
"""gambler_tp2_1m_verify · 验证 cond 找出的正期望子集（含「未触发」样本）

cond 的口径只统计「触发了任意一边」的样本（条件概率），
本脚本把「60 分钟内都没触发」的样本也计入（按 60 分钟收盘价平仓），
给出可交易的净期望。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL = 0.02, 0.01
HOLD = 60
FEE_M, FEE_T = 0.0004, 0.0020
DAYS = 60

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2_1m_verify · 含「未触发」样本的净期望（可交易口径）')
log('=' * 108)

SCEN = [
    ('全部', None),
    ('r15<=-5%', ('r15', -9, -0.05)),
    ('r15<=-5% & volr>=4', ('r15v', -9, -0.05, 4.0)),
    ('r15<=-5% & volr>=1', ('r15v', -9, -0.05, 1.0)),
    ('r15 -5~-3%', ('r15', -0.05, -0.03)),
]
ACC = {n: dict(n=0, tp=0, sl=0, to=0, ret=0.0) for n, _ in SCEN}

t_read = time.time()
n_sym = 0
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64',
                             'close': 'float32', 'high': 'float32', 'low': 'float32',
                             'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 0:
            continue
        n_sym += 1
        cc = c[:m]
        INF = 10 ** 9
        first_tp = np.full(m, INF, dtype=np.int64)
        first_sl = np.full(m, INF, dtype=np.int64)
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]; ll = l[j:j + m]
            nt = (first_tp == INF) & (hh >= cc * (1 + TP))
            ns = (first_sl == INF) & (ll <= cc * (1 - SL))
            first_tp[nt] = j
            first_sl[ns] = j
        r15 = np.full(m, np.nan)
        r15[15:] = cc[15:] / cc[:-15] - 1.0
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        timeout_ret = c[HOLD:HOLD + m] / cc - 1.0

        for name, cond in SCEN:
            if cond is None:
                mm = np.ones(m, dtype=bool)
            elif cond[0] == 'r15':
                mm = (r15 >= cond[1]) & (r15 < cond[2])
            else:
                mm = (r15 >= cond[1]) & (r15 < cond[2]) & (volr >= cond[3])
            mm = mm & np.isfinite(r15) & np.isfinite(volr) & np.isfinite(timeout_ret)
            if not mm.any():
                continue
            tp_m = mm & (first_tp < first_sl)
            sl_m = mm & (first_sl < first_tp)
            to_m = mm & (first_tp == INF) & (first_sl == INF)
            a = ACC[name]
            a['n'] += int(mm.sum())
            a['tp'] += int(tp_m.sum())
            a['sl'] += int(sl_m.sum())
            a['to'] += int(to_m.sum())
            a['ret'] += float((tp_m * TP + sl_m * (-SL) + to_m * timeout_ret).sum())
    del ch; gc.collect()
log(f'币 {n_sym}  扫描 {time.time()-t_read:.0f}s')

log('')
log(f'{"场景":22s}{"入场点":>12s}{"止盈%":>8s}{"止损%":>8s}{"超时%":>7s}'
    f'{"净maker%":>10s}{"净taker%":>10s}{"每天":>10s}')
log('-' * 108)
for name, _ in SCEN:
    a = ACC[name]
    if a['n'] == 0:
        continue
    n = a['n']
    gm = a['ret'] / n * 100
    log(f'{name:22s}{n:>12,}{a["tp"]/n*100:>8.1f}{a["sl"]/n*100:>8.1f}{a["to"]/n*100:>7.1f}'
        f'{gm-FEE_M*100:>10.3f}{gm-FEE_T*100:>10.3f}{n/DAYS:>10,.0f}')

log('')
log('【判读】')
log('  · 这一版把「60 分钟都没触发」按到期收盘价平仓计入，是真正可交易的期望')
log('  · 若某场景净maker > 0 且每天次数 >= 15，才满足用户的「一天 15 次」目标')

out = os.path.join(OUT, 'tp2_1m_verify_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
