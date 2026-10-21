# -*- coding: utf-8 -*-
"""gambler_tp2_1m_cond · 全量 502 币：哪些入场条件下「先触 +2%」的概率能超过 34.7% 门槛

在 job-24708-79 的全量扫描基础上，按入场前 15m 涨跌、量比、UTC 小时分档，
统计每档的「先触 TP 率」（触发任意一边的样本中，先触 +2% 的比例）。
只有超过 34.7% 的档才可能为正期望。
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
FEE_M = 0.0004
DAYS = 60

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('gambler_tp2_1m_cond · 全量 502 币：分档找「先触 +2% 率 > 34.7%」的入场条件')
log('=' * 108)

# 分档边界
R15 = [-9, -0.08, -0.05, -0.03, -0.015, -0.005, 0.005, 0.015, 0.03, 0.05, 0.08, 9]
VOLR = [0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]

acc = {}   # key -> [n_trigger, n_tp]

def add(key, mask_trig, mask_tp):
    a = acc.setdefault(key, [0, 0])
    a[0] += int(mask_trig.sum())
    a[1] += int((mask_trig & mask_tp).sum())

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
        first_tp = np.full(m, INF, dtype=np.int64)
        first_sl = np.full(m, INF, dtype=np.int64)
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]; ll = l[j:j + m]
            nt = (first_tp == INF) & (hh >= cc * (1 + TP))
            ns = (first_sl == INF) & (ll <= cc * (1 - SL))
            first_tp[nt] = j
            first_sl[ns] = j
        trig = (first_tp < INF) | (first_sl < INF)
        tp_win = first_tp < first_sl

        r15 = np.full(m, np.nan)
        r15[15:] = cc[15:] / cc[:-15] - 1.0
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        hour = ((ot[:m] // 3600000) % 24).astype(np.int64)

        add('ALL', trig, tp_win)
        # r15 分档
        for k in range(len(R15) - 1):
            mm = (r15 >= R15[k]) & (r15 < R15[k + 1])
            add(('r15', R15[k], R15[k + 1]), trig & mm, tp_win)
        # volr 分档
        for k in range(len(VOLR) - 1):
            mm = (volr >= VOLR[k]) & (volr < VOLR[k + 1])
            add(('volr', VOLR[k], VOLR[k + 1]), trig & mm, tp_win)
        # 小时
        for hh_ in range(24):
            add(('hour', hh_), trig & (hour == hh_), tp_win)
        # 交叉：r15 × volr（粗）
        for k in range(0, len(R15) - 2, 2):
            mm1 = (r15 >= R15[k]) & (r15 < R15[k + 2])
            for v in range(0, len(VOLR) - 1, 2):
                mm2 = (volr >= VOLR[v]) & (volr < VOLR[v + 2])
                add(('x', R15[k], R15[k + 2], VOLR[v], VOLR[v + 2]), trig & mm1 & mm2, tp_win)
    del ch; gc.collect()
log(f'币 {n_sym}  扫描 {time.time()-t_read:.0f}s')

def show(title, keys, fmt):
    log('\n' + '=' * 108)
    log(title)
    log('=' * 108)
    log(f'{"条件":42s}{"触发样本":>12s}{"先触TP":>10s}{"先触TP率":>10s}{"净期望%":>10s}')
    log('-' * 108)
    for k in keys:
        if k not in acc:
            continue
        n_, t_ = acc[k]
        if n_ < 200:
            continue
        rate = t_ / n_
        # 净期望 = rate*TP - (1-rate)*SL - FEE（假设只统计触发样本，未触发按 0 略）
        gm = rate * TP - (1 - rate) * SL - FEE_M
        log(f'{fmt(k):42s}{n_:>12,}{t_:>10,}{rate:>9.1%}{gm*100:>10.3f}')

show('【一】总体与 r15 分档', ['ALL'] + [k for k in acc if isinstance(k, tuple) and k[0] == 'r15'],
     lambda k: 'ALL' if k == 'ALL' else f'r15 ∈ [{k[1]:+.1%}, {k[2]:+.1%})')
show('【二】量比分档', [k for k in acc if isinstance(k, tuple) and k[0] == 'volr'],
     lambda k: f'volr ∈ [{k[1]:.1f}, {k[2]:.0f})')
show('【三】UTC 小时', [k for k in acc if isinstance(k, tuple) and k[0] == 'hour'],
     lambda k: f'{k[1]:02d}:00')

log('\n' + '=' * 108)
log('【四】r15 × volr 交叉：按先触 TP 率排序 top 15（触发样本 >= 500）')
log('=' * 108)
xs = [(k, v) for k, v in acc.items() if isinstance(k, tuple) and k[0] == 'x' and v[0] >= 500]
xs.sort(key=lambda kv: kv[1][1] / kv[1][0], reverse=True)
log(f'{"r15 档":>18s}{"volr 档":>14s}{"触发样本":>12s}{"先触TP":>10s}{"先触TP率":>10s}{"净期望%":>10s}')
log('-' * 108)
for k, (n_, t_) in xs[:15]:
    rate = t_ / n_
    gm = rate * TP - (1 - rate) * SL - FEE_M
    log(f'{f"[{k[1]:+.0%},{k[2]:+.0%})":>18s}{f"[{k[3]:.1f},{k[4]:.0f})":>14s}{n_:>12,}{t_:>10,}{rate:>9.1%}{gm*100:>10.3f}')
log(f'\n交叉格总数 {len(xs)}  其中先触TP率 > 34.7% 的 {sum(1 for k,(n_,t_) in xs if t_/n_ > 0.347)} 个')

out = os.path.join(OUT, 'tp2_1m_cond_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
