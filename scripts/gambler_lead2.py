# -*- coding: utf-8 -*-
"""gambler_lead2.py — 领先信号：大盘（BTC/ETH/市场）动 → 小币跟不跟？
用户要求（2026-10-16）：
  "去试试"（指上一条提的"用 BTC/ETH 异动预测小币异动"）
  "另外信号或许不止一个，而是很多个。只是没有发现"

做法：
  1) 60 币对齐到统一 5m 时间轴
  2) 定义 5 个**领先事件**（不止 BTC 一个）：
       L1 BTC 1h 涨>=3%      L2 BTC 1h 跌>=3%
       L3 ETH 1h 涨>=3%      L4 ETH 1h 跌>=3%
       L5 市场齐动：同刻 >=10 个币 1h 同向动 >=3%
  3) 定义**目标事件**（小币跟随，排除 BTC/ETH/WBETH 本身）：
       未来 tau 内出现 1h 涨>=10%（UP）/ <=-10%（DN）
  4) 对每个 (领先事件, tau) 算：lift、条件概率、精确率、覆盖率
  5) 判读：领先信号的精确率能否达到 85%
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_lead2.txt')

BPD = 288
LAGS = [1, 3, 6, 12, 24]          # 5m/15m/30m/1h/2h
THR_LEAD = 0.03                    # 领先事件阈值：1h 动 3%
THR_TGT = 0.10                     # 目标事件阈值：未来 1h 动 10%

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_lead2 · 领先信号：大盘动 → 小币跟不跟？')
log('=' * 100)

frames = {}
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32'},
                      chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        frames[sym] = g.sort_values('open_time').set_index('open_time')['close']
    del ch; gc.collect()

syms = sorted(frames)
P = pd.DataFrame({s: frames[s] for s in syms}).sort_index()
del frames; gc.collect()
log(f'币 {len(syms)}   矩阵 {P.shape}   耗时 {time.time()-t0t:.0f}s')

R = P.pct_change(12).values                 # T x N，1h 收益
T, N = R.shape
Rn = np.nan_to_num(R, nan=0.0)

MAJOR = {'BTCUSDT', 'ETHUSDT', 'WBETHUSDT'}
tgt_idx = [i for i, s in enumerate(syms) if s not in MAJOR]
log(f'目标币（排除大盘）{len(tgt_idx)} 个')

# 领先事件系列
LEADS = {}
if 'BTCUSDT' in syms:
    b = Rn[:, syms.index('BTCUSDT')]
    LEADS['BTC涨3%'] = b >= THR_LEAD
    LEADS['BTC跌3%'] = b <= -THR_LEAD
if 'ETHUSDT' in syms:
    e = Rn[:, syms.index('ETHUSDT')]
    LEADS['ETH涨3%'] = e >= THR_LEAD
    LEADS['ETH跌3%'] = e <= -THR_LEAD
up_cnt = (Rn >= THR_LEAD)[:, tgt_idx].sum(axis=1)
dn_cnt = (Rn <= -THR_LEAD)[:, tgt_idx].sum(axis=1)
LEADS['市场齐涨(>=10币)'] = up_cnt >= 10
LEADS['市场齐跌(>=10币)'] = dn_cnt >= 10

for k, v in LEADS.items():
    log(f'  领先事件 {k}: 触发 {int(np.nansum(v)):,} 次')

# 目标事件：未来 tau 内出现 1h 涨>=10% / <=-10%
log('')
log('=' * 100)
log('【主结果】领先事件后，小币跟随的概率')
log('=' * 100)
log(f'  度量：条件概率 P(小币动|领先事件) / 无条件 P(小币动) = lift；'
    f'以及"报出后的精确率"（报出的那些币里真的动的比例）')

for lname, lmsk in LEADS.items():
    lidx = np.flatnonzero(lmsk)
    if len(lidx) == 0:
        continue
    log('')
    log(f'--- 领先事件：{lname}（触发 {len(lidx):,} 次）---')
    log(f'{"tau":>7s} | {"目标":>6s} {"报出数":>9s} {"条件概率%":>10s} {"基准%":>8s} {"lift":>8s} {"精确率%":>8s}')
    for lag in LAGS:
        for tag, cond in (('UP', Rn >= THR_TGT), ('DN', Rn <= -THR_TGT)):
            fut = np.zeros_like(cond)
            if lag < T:
                fut[:-lag] = cond[lag:]
            base = cond[:, tgt_idx].mean()
            if base <= 0:
                continue
            # 报出：领先事件时刻，未来 lag 内目标币出现异动
            n_rep = 0; n_hit = 0
            for t in lidx:
                if t + lag >= T:
                    continue
                sub = fut[t:t + lag + 1][:, tgt_idx]
                n_rep += sub.shape[1]
                n_hit += int(sub.any(axis=0).sum())
            if n_rep == 0:
                continue
            pcond = n_hit / n_rep
            log(f'{lag:7d} | {tag:>6s} {n_rep:9,d} {100*pcond:10.2f} {100*base:8.3f} '
                f'{pcond/max(base,1e-9):8.2f} {100*pcond:8.2f}')

log('')
log('=' * 100)
log('【对照】单独用"市场齐跌"预测单币跌>=30%（验证此前 cascade 结论）')
log('=' * 100)
for lname in ('市场齐跌(>=10币)', 'BTC跌3%'):
    if lname not in LEADS:
        continue
    lmsk = LEADS[lname]
    lidx = np.flatnonzero(lmsk)
    if len(lidx) == 0:
        continue
    # 目标：未来 24h 内单币跌 >= 30%
    r24 = P.pct_change(288).values
    cond = np.nan_to_num(r24) <= -0.30
    base = cond[:, tgt_idx].mean()
    n_rep = 0; n_hit = 0
    for t in lidx:
        if t + 288 >= T:
            continue
        sub = cond[t + 1:t + 289][:, tgt_idx]
        n_rep += sub.shape[1]
        n_hit += int(sub.any(axis=0).sum())
    if n_rep:
        p = n_hit / n_rep
        log(f'  {lname}: 报出 {n_rep:,}  条件概率 {100*p:.2f}%  基准 {100*base:.3f}%  '
            f'lift {p/max(base,1e-9):.2f}x')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
