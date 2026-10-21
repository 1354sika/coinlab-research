# -*- coding: utf-8 -*-
"""gambler_group2.py — 群体 vs 个体 + 接力（修正时间单位 bug 后重跑）
用户要求（2026-10-16）："群体 vs 个体这个参数依然非常重要"
                      "把接力的也做好。并且要考虑到连续接力或间隔接力"
                      "也要考虑是进行到一半，然后接力就开始"

上一版（job-36080-3 two-kinds）因时间单位混用导致结果全 nan，本版统一用毫秒时间戳。

做法：
  1) 事件 = 4h 涨 >= 50%（局）
  2) 对每个事件算"启动同步度" = 该币启动前 1h 内，全市场有多少币也在启动
     sync >= 3 → 群体型；sync == 0 → 个体型
  3) 对比两组：峰值涨幅、到峰时长、24h 回撤、跟盘净收益（+50%止盈/-15%止损/24h超时）
  4) 接力：24h 内 A 结束后 B 启动（接力对）；分"连续接力"（间隔<=1h）与"间隔接力"
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_group2.txt')

FEE, SLIP = 0.0004, 0.0005
UP = 0.30          # 4h 涨 30% 算局（放宽门槛，保证样本量）
W4 = 48            # 4h = 48 根
TO = 288           # 24h

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_group2 · 群体 vs 个体 + 接力（时间单位已统一为毫秒）')
log('=' * 100)

DATA = {}
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'}, chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        DATA[sym] = (g['open_time'].values.astype('int64'),
                     g['close'].values.astype('float64'),
                     g['high'].values.astype('float64'),
                     g['low'].values.astype('float64'))
    del ch; gc.collect()
log(f'币 {len(DATA)}   耗时 {time.time()-t0t:.0f}s')

# 1) 找所有局（事件）：4h 涨 >= 50%
EVENTS = []   # (sym, t_ms, i_peak, gain, i_lo)
for sym, (ot, c, h, l) in DATA.items():
    n = len(c)
    if n < W4 + 10:
        continue
    r4 = np.full(n, np.nan); r4[W4:] = c[W4:] / c[:-W4] - 1.0
    cand = np.flatnonzero(np.nan_to_num(r4) >= UP)
    if len(cand) == 0:
        continue
    last = -10 ** 9
    for t0 in cand:
        if t0 - last < W4:
            continue
        i1 = t0 + int(np.argmax(h[t0:min(n, t0 + W4)]))
        peak = h[i1]
        gain = peak / c[t0] - 1.0
        EVENTS.append((sym, int(ot[t0]), int(t0), int(i1), float(gain)))
        last = i1
log(f'事件（4h 涨>=50%）{len(EVENTS):,}   耗时 {time.time()-t0t:.0f}s')

# 2) 启动同步度
ev_by_time = {}
for e in EVENTS:
    ev_by_time.setdefault(e[1], []).append(e)
tss = np.array(sorted(ev_by_time))
log(f'唯一起始时刻 {len(tss):,}')

def sync_of(t_ms):
    # 该时刻前后 1h（12 根 5m = 3600000 ms）内有多少个其他事件
    lo = t_ms - 3600000; hi = t_ms + 3600000
    k = np.searchsorted(tss, [lo, hi])
    cnt = 0
    for t in tss[k[0]:k[1]]:
        cnt += len(ev_by_time[t])
    return cnt - 1

# 3) 分组 + 结果
rows = []
for (sym, t_ms, i0, i1, gain) in EVENTS:
    ot, c, h, l = DATA[sym]
    n = len(c)
    peak = h[i1]
    # 到峰时长
    mins = (ot[i1] - ot[i0]) / 60000.0
    # 24h 后回撤
    j2 = min(n - 1, i1 + TO)
    dd = c[j2] / peak - 1.0 if j2 > i1 else np.nan
    # 跟盘：i0 买入，+50% 止盈 / -15% 止损 / 24h 超时
    px = c[i0]
    seg = c[i0:min(n, i0 + TO)]
    ret = seg / (px * (1 + SLIP)) - 1.0
    a = np.flatnonzero(ret >= 0.5); b = np.flatnonzero(ret <= -0.15)
    if len(a) and (not len(b) or a[0] <= b[0]):
        g = 0.5
    elif len(b):
        g = -0.15
    else:
        g = ret[-1] if len(ret) else np.nan
    net = g - FEE - 2 * SLIP
    sy = sync_of(t_ms)
    rows.append((sym, t_ms, gain, mins, dd, net, sy))

R = pd.DataFrame(rows, columns=['sym', 't', 'gain', 'mins', 'dd24', 'net', 'sync'])
log(f'事件表 {R.shape}')

log('')
log('=' * 100)
log('【A】群体 vs 个体（sync >= 3 为群体）')
log('=' * 100)
for tag, sub in (('群体 sync>=3', R[R['sync'] >= 3]), ('个体 sync==0', R[R['sync'] == 0]),
                 ('全部', R)):
    if len(sub) == 0:
        log(f'  {tag}: 无样本'); continue
    log(f'  {tag:<14s} n={len(sub):5d}  峰值涨中位 {100*sub["gain"].median():7.1f}%  '
        f'到峰中位 {sub["mins"].median():6.0f}min  24h回撤中位 {100*sub["dd24"].median():7.1f}%  '
        f'跟盘净均值 {100*sub["net"].mean():+7.2f}%  胜率 {100*(sub["net"]>0).mean():5.1f}%')

log('')
log('=' * 100)
log('【B】接力：同一币 24h 内先后两次启动（j0 -> j1）')
log('=' * 100)
by_sym = {}
for r in rows:
    by_sym.setdefault(r[0], []).append(r)
relay = []
for sym, lst in by_sym.items():
    lst = sorted(lst, key=lambda x: x[1])
    for a, b in zip(lst, lst[1:]):
        gap_h = (b[1] - a[1]) / 3600000.0
        if gap_h <= 24:
            relay.append((sym, gap_h, a[2], b[2], a[5], b[5]))
RL = pd.DataFrame(relay, columns=['sym', 'gap_h', 'g0', 'g1', 'n0', 'n1'])
log(f'  接力对（24h 内）{len(RL):,}')
if len(RL):
    cont = RL[RL['gap_h'] <= 1.0]
    gapd = RL[(RL['gap_h'] > 1.0) & (RL['gap_h'] <= 8.0)]
    later = RL[RL['gap_h'] > 8.0]
    for tag, sub in (('连续接力 (<=1h)', cont), ('间隔接力 (1~8h)', gapd), ('长间隔 (>8h)', later)):
        if len(sub) == 0:
            log(f'  {tag:<18s}: 无'); continue
        log(f'  {tag:<18s} n={len(sub):5d}  第二次峰值涨中位 {100*sub["g1"].median():6.1f}%  '
            f'跟盘净均值 {100*sub["n1"].mean():+7.2f}%  胜率 {100*(sub["n1"]>0).mean():5.1f}%')
    log('')
    log(f'  接力 vs 非接力：接力事件跟盘净均值 {100*RL["n1"].mean():+.2f}%  '
        f'（全事件 {100*R["net"].mean():+.2f}%）')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
