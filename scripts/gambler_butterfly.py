# -*- coding: utf-8 -*-
"""gambler_butterfly.py — 事件概率的跨币关联矩阵（蝴蝶效应）
用户设想（2026-10-16）："事件概率之间的关联也要有。比如这个事情发生了，那其他事情发生的概率有多少？"
                      "一切皆有因果，只是我们还没有发现。"

做法：
  1) 60 币 3 年 5m 对齐到统一时间轴
  2) 二值化"异动"事件：1h 收益 >= +5%（上） / <= -5%（下）
  3) 对每对 (A,B) 与滞后 tau，算：
        P(B异动 @ t+tau | A异动 @ t) / P(B异动)  =  lift
     lift > 1 表示 A 发生后 B 更可能异动（有传导）；lift ≈ 1 表示无关
  4) 输出：每档 tau 的 top 传导对 + 方向性判别（A→B 强还是 B→A 强）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_butterfly.txt')

LAGS = [1, 3, 6, 12, 24, 48, 96]   # 5m 根：5m/15m/30m/1h/2h/4h/8h
THR = 0.05                          # 1h 异动阈值
MAXSYM = 60

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_butterfly · 事件概率的跨币关联矩阵（蝴蝶效应）')
log(f'  异动定义：1h 收益 |ret| >= {THR:.0%}   滞后档：{LAGS}（5m 根）')
log('=' * 100)

frames = {}
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32'},
                      chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        frames[sym] = g.sort_values('open_time').set_index('open_time')['close']
    del ch
    gc.collect()

syms = sorted(frames)[:MAXSYM]
df = pd.DataFrame({s: frames[s] for s in syms}).sort_index()
del frames
gc.collect()
log(f'币数 {len(syms)}   对齐矩阵 {df.shape}   耗时 {time.time()-t0t:.0f}s')

R1 = df.pct_change(12).values          # T x N，1h 收益
T, N = R1.shape
log(f'矩阵 {T:,} 行 x {N} 币   耗时 {time.time()-t0t:.0f}s')

for tag, mask in (('UP', R1 >= THR), ('DN', R1 <= -THR)):
    log('')
    log('=' * 100)
    log(f'【{tag}】事件 = 1h {"涨" if tag=="UP" else "跌"} >= {THR:.0%}')
    log('=' * 100)
    E = mask
    E = np.nan_to_num(E, nan=0).astype('float32')
    base = E.mean(axis=0)                      # 各币基础概率
    log(f'  基础概率中位 {100*np.median(base):.2f}%（>0 的币 {int((base>0).sum())} 个）')

    for lag in LAGS:
        Bm = np.zeros_like(E)
        if lag < T:
            Bm[:-lag] = E[lag:]
        num = (E.T @ Bm).astype('float64')     # N x N
        den = E.sum(axis=0).astype('float64')[:, None]
        P = num / np.maximum(den, 1.0)
        L = P / np.maximum(base[None, :], 1e-9)
        np.fill_diagonal(L, np.nan)
        # top 对
        flat = np.nan_to_num(L, nan=0).ravel()
        idx = np.argsort(flat)[::-1][:8]
        log(f'  --- 滞后 {lag} 根（{lag*5} 分钟）---')
        for k in idx:
            a, b = divmod(int(k), N)
            if flat[k] <= 0:
                continue
            rev = L[b, a]
            log(f'    {syms[a]:>12s} -> {syms[b]:<12s}  lift={flat[k]:6.3f}   '
                f'（反向 {syms[b]}->{syms[a]} = {rev:6.3f}）')
        # 矩阵整体统计
        v = L[np.isfinite(L)]
        log(f'    全矩阵 lift：中位 {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}  '
            f'p99 {np.percentile(v,99):.3f}  最大 {v.max():.2f}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
