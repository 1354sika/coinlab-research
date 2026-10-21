# -*- coding: utf-8 -*-
"""gambler_multi.py — 多信号组合：把弱信号叠起来，精确率能到多少？
用户洞察（2026-10-16）：
  "信号或许不止一个，而是很多个。只是没有发现"
  "概率的表征是简单的，但最终这个概率的结果却是复杂的，可能又很多概率相互影响组成"

背景：单个领先信号（BTC 5min 内跌 1%）的条件概率只有几个百分点，离 85% 差得远。
本脚本把条件**逐层叠加**，看每一步精确率怎么变——这是"概率相互影响"的直接测量。

做法：
  1) 60 币对齐（close + quote_volume + trades）
  2) 定义条件（全部只用 t 及之前的信息，无前视）：
       C0 全部时刻（基准）
       C1 BTC 在 t 时刻 30min 内跌 >= 1%
       C2 + 该币自身 1h 已跌 >= 3%
       C3 + 该币量比(24根) >= 3
       C4 + 该币笔数密度(24根) >= 2
       C5 + 市场同刻 >= 10 个币 1h 跌 >= 3%
       C6 + 该币波动率跳升（|r1| > 3×σ24）
  3) 目标：该币在未来 tau ∈ {1,3,6,12} 根（5m~1h）内出现 1h 跌幅 >= 10%
  4) 输出：每层条件的 报出数 / 条件概率 / 相对基准的 lift
  5) 判读：要到 85% 精确率，需要叠到第几层？还是永远到不了？
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_multi.txt')

THR_TGT = 0.10
LAGS = [1, 3, 6, 12]
BPD = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_multi · 多信号组合（条件逐层叠加，精确率能到多少？）')
log('=' * 100)

# 读三列，按币分组
cols = {}
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'quote_volume': 'float32', 'trades': 'float32'}, chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        cols[sym] = (g['open_time'].values, g['close'].values.astype('float64'),
                     g['quote_volume'].values.astype('float64'), g['trades'].values.astype('float64'))
    del ch; gc.collect()

syms = sorted(cols)
ot = cols[syms[0]][0]
log(f'币 {len(syms)}   耗时 {time.time()-t0t:.0f}s')

C = pd.DataFrame({s: pd.Series(cols[s][1], index=cols[s][0]) for s in syms}).sort_index()
Q = pd.DataFrame({s: pd.Series(cols[s][2], index=cols[s][0]) for s in syms}).sort_index()
TR = pd.DataFrame({s: pd.Series(cols[s][3], index=cols[s][0]) for s in syms}).sort_index()
del cols; gc.collect()
T, N = C.shape
log(f'矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

R1 = C.pct_change(12).values                       # 1h 收益
Rn = np.nan_to_num(R1, nan=0.0)
MAJOR = {'BTCUSDT', 'ETHUSDT', 'WBETHUSDT'}
tgt = [i for i, s in enumerate(syms) if s not in MAJOR]
log(f'目标币 {len(tgt)} 个')

# 各条件
btc = Rn[:, syms.index('BTCUSDT')] if 'BTCUSDT' in syms else np.zeros(T)
# BTC 短窗跌：30min（6 根）内跌 >= 1%
btc_r6 = np.zeros(T); btc_r6[6:] = btc[6:] - btc[:-6]
C1 = np.repeat((btc_r6 <= -0.01)[:, None], N, axis=1)     # 广播成 (T,N)，与 C2 对齐
C2 = Rn <= -0.03                                    # 自身 1h 跌 >=3%
v24 = Q.rolling(24, min_periods=6).mean().values
vr = np.where(v24 > 0, Q.values / np.maximum(v24, 1e-12), 1.0)
C3 = vr >= 3.0
t24 = TR.rolling(24, min_periods=6).mean().values
trr = np.where(t24 > 0, TR.values / np.maximum(t24, 1e-12), 1.0)
C4 = trr >= 2.0
dn_cnt = (Rn <= -0.03)[:, tgt].sum(axis=1)
C5 = np.repeat((dn_cnt >= 10)[:, None], N, axis=1)
vol24 = pd.DataFrame(np.abs(Rn)).rolling(24, min_periods=6).mean().values
C6 = np.abs(np.nan_to_num(C.pct_change().values, nan=0.0)) > 3.0 * np.maximum(vol24, 1e-4)

CONDS = [('C0 全部（基准）', np.ones_like(C1, dtype=bool)),
         ('C1 BTC 30min 跌>=1%', C1),
         ('C2 +自身1h跌>=3%', C1 & C2),
         ('C3 +量比>=3', C1 & C2 & C3),
         ('C4 +笔数密度>=2', C1 & C2 & C3 & C4),
         ('C5 +市场>=10币跌>=3%', C1 & C2 & C3 & C4 & C5),
         ('C6 +波动跳升3σ', C1 & C2 & C3 & C4 & C5 & C6)]

# 另一种叠法：不打底 BTC，只用自身+市场
ALT = [('A1 自身1h跌>=3%', C2),
       ('A2 +量比>=3', C2 & C3),
       ('A3 +市场齐跌', C2 & C3 & C5),
       ('A4 +笔数密度>=2', C2 & C3 & C5 & C4),
       ('A5 +波动跳升', C2 & C3 & C5 & C4 & C6)]

for lname, ls in (('主链（从 BTC 起叠）', CONDS), ('副链（从自身下跌起叠）', ALT)):
    log('')
    log('=' * 100)
    log(f'【{lname}】')
    log('=' * 100)
    log(f'{"条件":<26s} | ' + ' | '.join(f'{"tau=%d"%l:>26s}' for l in LAGS))
    log(f'{"":<26s} | ' + ' | '.join(f'{"报出  条件概率%  基准%  lift":>26s}' for l in LAGS))
    for tag, msk in ls:
        cells = []
        for lag in LAGS:
            fut = np.zeros((T, N), dtype=bool)
            if lag < T:
                fut[:-lag] = (Rn[lag:] <= -THR_TGT)
            ti = np.array(tgt)
            sub_m = msk[:, ti] & fut[:, ti]        # 该时刻该币：条件满足 且 命中
            rep = msk[:, ti].sum()
            hit = sub_m.sum()
            base = (Rn[:, ti] <= -THR_TGT).mean()
            p = hit / max(rep, 1)
            cells.append((rep, 100*p, 100*base, p/max(base, 1e-9)))
        log(f'{tag:<26s} | ' + ' | '.join(
            f'{c[0]:8,d} {c[1]:8.2f} {c[2]:7.3f} {c[3]:7.2f}' for c in cells))

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 报出数 = 该条件下（币×时刻）的样本数')
log('  · 条件概率 = 满足条件的那些里，真的在未来 tau 内跌>=10% 的比例')
log('  · 目标 85% —— 看哪一层的条件概率最接近')
log('  · 若逐层叠加后条件概率**反而下降**，说明各条件在解释同一个信息（共线），')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
