# -*- coding: utf-8 -*-
"""gambler_multi2.py — 在 C6 之上继续叠新信号，能否从 20.25% 推到 50%+
用户要求（2026-10-16）："去试"（指上一条提议的新信号叠加）

基线：C6 = BTC跌 & 自身跌>=3% & 量比>=3 & 笔数>=2 & 市场齐跌 & 波动跳升3σ
       报出 864，条件概率 20.25%（tau=1）

本脚本在 C6 之上**逐个叠加**新信号，每加一层看：
   报出数（机会变少多少）
   条件概率（精确率升高多少）

新信号：
   S1 连续跳升   —— 近 3 根的 |r1| 连续递增（波动在放大）
   S2 下跌加速   —— 近 3 根跌幅 > 前 3 根跌幅（越跌越快）
   S3 量能刚启动 —— 前 3 根量比 < 1，近 3 根量比 > 3（从静到爆）
   S4 反复被搞   —— 过去 24h 内 1h 跌>=3% 出现 >= 2 次
   S5 新鲜异动   —— 距上次 1h 跌>=3% 在 6 根以内
   S6 深度已跌   —— 自身 1h 已跌 >= 6%
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_multi2.txt')

THR_TGT = 0.10
LAGS = [1, 3, 6, 12]

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_multi2 · 在 C6 之上继续叠新信号')
log('=' * 100)

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
C = pd.DataFrame({s: pd.Series(cols[s][1], index=cols[s][0]) for s in syms}).sort_index()
Q = pd.DataFrame({s: pd.Series(cols[s][2], index=cols[s][0]) for s in syms}).sort_index()
TR = pd.DataFrame({s: pd.Series(cols[s][3], index=cols[s][0]) for s in syms}).sort_index()
del cols; gc.collect()
T, N = C.shape
log(f'矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

R1 = C.pct_change(12).values
Rn = np.nan_to_num(R1, nan=0.0)
r1 = np.nan_to_num(C.pct_change().values, nan=0.0)
MAJOR = {'BTCUSDT', 'ETHUSDT', 'WBETHUSDT'}
tgt = np.array([i for i, s in enumerate(syms) if s not in MAJOR])
log(f'目标币 {len(tgt)} 个')

btc = Rn[:, syms.index('BTCUSDT')]
btc_r6 = np.zeros(T); btc_r6[6:] = btc[6:] - btc[:-6]
C1 = np.repeat((btc_r6 <= -0.01)[:, None], N, axis=1)
C2 = Rn <= -0.03
v24 = Q.rolling(24, min_periods=6).mean().values
vr = np.where(v24 > 0, Q.values / np.maximum(v24, 1e-12), 1.0)
C3 = vr >= 3.0
t24 = TR.rolling(24, min_periods=6).mean().values
trr = np.where(t24 > 0, TR.values / np.maximum(t24, 1e-12), 1.0)
C4 = trr >= 2.0
dn_cnt = (Rn <= -0.03)[:, tgt].sum(axis=1)
C5 = np.repeat((dn_cnt >= 10)[:, None], N, axis=1)
vol24 = pd.DataFrame(np.abs(Rn)).rolling(24, min_periods=6).mean().values
C6 = np.abs(r1) > 3.0 * np.maximum(vol24, 1e-4)
BASE = C1 & C2 & C3 & C4 & C5 & C6

# ---- 新信号 ----
a1 = np.abs(r1)
S1 = np.zeros((T, N), dtype=bool)
S1[2:] = (a1[2:] > a1[1:-1]) & (a1[1:-1] > a1[:-2])
# 下跌加速：近 3 根累计跌幅 > 前 3 根累计跌幅（且在跌）
d3 = np.zeros((T, N)); d3[3:] = C.values[3:] / C.values[:-3] - 1.0
d3p = np.zeros((T, N)); d3p[6:] = C.values[3:-3] / C.values[:-6] - 1.0
S2 = (d3 < 0) & (d3 < d3p)
# 量能刚启动
v3 = Q.rolling(3, min_periods=1).mean().values
v3p = Q.rolling(3, min_periods=1).mean().shift(3).values
vr_prev = np.where(v24 > 0, v3p / np.maximum(v24, 1e-12), 1.0)
S3 = (vr_prev < 1.0) & (vr >= 3.0)
# 反复被搞：24h 内 1h 跌>=3% 出现 >=2 次
dn_evt = (Rn <= -0.03).astype('float32')
S4 = pd.DataFrame(dn_evt).rolling(288, min_periods=24).sum().values >= 2
# 新鲜异动：距上次 1h 跌>=3% 在 6 根内
S5 = np.zeros((T, N), dtype='float32')
S5[dn_evt > 0] = 1
S5 = pd.DataFrame(S5).rolling(6, min_periods=1).max().values > 0
# 深度已跌
S6 = Rn <= -0.06

NEW = [('S1 连续跳升', S1), ('S2 下跌加速', S2), ('S3 量能刚启动', S3),
       ('S4 反复被搞(24h>=2次)', S4), ('S5 新鲜异动(6根内)', S5), ('S6 深度已跌>=6%', S6)]

def ev(M):
    ti = tgt
    out = []
    for lag in LAGS:
        fut = np.zeros((T, N), dtype=bool)
        if lag < T:
            fut[:-lag] = (Rn[lag:] <= -THR_TGT)
        rep = M[:, ti].sum()
        hit = (M[:, ti] & fut[:, ti]).sum()
        out.append((rep, 100 * hit / max(rep, 1)))
    return out

log('')
log('=' * 100)
log('【基线 C6】')
log('=' * 100)
cells = ev(BASE)
log(f'{"C6":<26s} | ' + ' | '.join(f'tau={l}: 报出 {c[0]:6,d} 概率 {c[1]:5.2f}%' for l, c in zip(LAGS, cells)))

log('')
log('=' * 100)
log('【逐个叠加新信号（C6 & Sx）】')
log('=' * 100)
log(f'{"叠加":<30s} | ' + ' | '.join(f'{"tau=%d"%l:>24s}' for l in LAGS))
log(f'{"":<30s} | ' + ' | '.join(f'{"报出   概率%  增益":>24s}' for l in LAGS))
base_cells = ev(BASE)
for tag, S in NEW:
    M = BASE & S
    cells = ev(M)
    row = []
    for (l, c) in zip(LAGS, cells):
        # 增益 = 概率比基线
        bi = LAGS.index(l)
        b = base_cells[bi][1]
        gain = c[1] / max(b, 1e-9)
        row.append(f'{c[0]:8,d} {c[1]:7.2f} {gain:5.2f}x')
    log(f'{tag:<30s} | ' + ' | '.join(f'{x:>24s}' for x in row))

log('')
log('=' * 100)
log('【两两组合（C6 & Si & Sj），只列出概率最高的 8 组】')
log('=' * 100)
import itertools
res = []
for (t1, s1), (t2, s2) in itertools.combinations(NEW, 2):
    M = BASE & s1 & s2
    c = ev(M)
    res.append((c[0][1], c[0][0], t1, t2, c))
res.sort(key=lambda r: -r[0])
log(f'{"组合":<44s} | {"报出":>7s} {"tau=1 概率%":>10s} {"tau=3 概率%":>10s}')
for p, rep, t1, t2, c in res[:8]:
    log(f'{t1+" & "+t2:<44s} | {c[0][0]:7,d} {c[0][1]:10.2f} {c[1][1]:10.2f}')

log('')
log('=' * 100)
log('【全部叠上（C6 & 所有新信号）】')
log('=' * 100)
M = BASE.copy()
for _, S in NEW:
    M = M & S
c = ev(M)
log(f'  报出 {c[0][0]:,}   tau=1 概率 {c[0][1]:.2f}%   tau=3 概率 {c[1][1]:.2f}%')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
