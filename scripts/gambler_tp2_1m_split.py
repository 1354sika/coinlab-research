# -*- coding: utf-8 -*-
"""gambler_tp2_1m_split · 时间留出：把 60 天切成前后两半，验证「跌超5%+放量」格是否稳定"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL, HOLD = 0.02, 0.01, 60
FEE_M, FEE_T = 0.0004, 0.0020

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_tp2_1m_split · 时间留出验证（前30天 / 后30天）')
log('=' * 100)

SCEN = [
    ('r15<=-5% & volr>=4', -9, -0.05, 4.0),
    ('r15<=-5% & volr>=1', -9, -0.05, 1.0),
    ('r15<=-5%', -9, -0.05, 0.0),
]
ACC = {}
for nm, *_ in SCEN:
    for half in ('前30天', '后30天'):
        ACC[(nm, half)] = dict(n=0, tp=0, sl=0, to=0, ret=0.0)

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
        ot = g['open_time'].values.astype('int64')
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
        f_tp = np.full(m, INF, dtype=np.int64)
        f_sl = np.full(m, INF, dtype=np.int64)
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]; ll = l[j:j + m]
            nt = (f_tp == INF) & (hh >= cc * (1 + TP))
            ns = (f_sl == INF) & (ll <= cc * (1 - SL))
            f_tp[nt] = j
            f_sl[ns] = j
        r15 = np.full(m, np.nan)
        r15[15:] = cc[15:] / cc[:-15] - 1.0
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tret = c[HOLD:HOLD + m] / cc - 1.0
        half = np.where(ot[:m] < np.median(ot), '前30天', '后30天')

        for nm, lo, hi, vmin in SCEN:
            mm = (r15 >= lo) & (r15 < hi) & (volr >= vmin)
            mm &= np.isfinite(r15) & np.isfinite(volr) & np.isfinite(tret)
            for hh_nm in ('前30天', '后30天'):
                sel = mm & (half == hh_nm)
                if not sel.any():
                    continue
                tp_m = sel & (f_tp < f_sl)
                sl_m = sel & (f_sl < f_tp)
                to_m = sel & (f_tp == INF) & (f_sl == INF)
                a = ACC[(nm, hh_nm)]
                a['n'] += int(sel.sum())
                a['tp'] += int(tp_m.sum())
                a['sl'] += int(sl_m.sum())
                a['to'] += int(to_m.sum())
                a['ret'] += float((tp_m * TP + sl_m * (-SL) + to_m * tret).sum())
    del ch; gc.collect()
log(f'币 {n_sym}  扫描 {time.time()-t_read:.0f}s')

log('')
log(f'{"条件":24s}{"半段":8s}{"入场点":>10s}{"止盈%":>8s}{"止损%":>8s}{"超时%":>7s}{"净maker%":>10s}{"净taker%":>10s}')
log('-' * 100)
for nm, *_ in SCEN:
    for half in ('前30天', '后30天'):
        a = ACC[(nm, half)]
        if a['n'] == 0:
            log(f'{nm:24s}{half:8s}{"无样本":>10s}')
            continue
        n = a['n']
        gm = a['ret'] / n * 100
        log(f'{nm:24s}{half:8s}{n:>10,}{a["tp"]/n*100:>8.1f}{a["sl"]/n*100:>8.1f}'
            f'{a["to"]/n*100:>7.1f}{gm-FEE_M*100:>10.3f}{gm-FEE_T*100:>10.3f}')

log('')
log('【判读】两半同号为正才算稳定；一正一负 = 时段依赖')

out = os.path.join(OUT, 'tp2_1m_split_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
