# -*- coding: utf-8 -*-
"""gambler_state.py — 状态机 + 条件概率/期望（"概率博弈"核心）
设计要点（用户 2026-10-16）：窗口不确定（5min~24h），起止由数据定。
做法：不用固定窗口，而用"当期数据"给每根 K 线打状态标签，然后统计
  A 状态转移矩阵  P(未来 k 根后处于 s' | 当前 s)
  B 条件分布      P(未来 24h 内 >=+50% | s)、P(<=-30% | s)、E[未来 24h 最大涨幅 | s]
这直接回答"当前处于什么状态 → 下一步发生什么 → 概率多大 → 期望多少"。
状态定义（全部只用当期及过去数据，无前视）：
  S0 静默   量比<1.5 且 |24h涨幅|<3%
  S1 吸筹   笔数比>=4 且 量比>=3 且 |6根涨幅|<5%      （放量不涨）
  S2 拉升   6 根涨幅>=10% 且 量比>=2
  S3 高位滞涨 距 24h 高点<3% 且 量比>=3 且 6 根涨幅<5% （放量不涨，接近顶部）
  S4 崩塌   6 根涨幅<=-10% 且 量比>=2
  S5 其他
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_state.txt')
N_COIN_LIMIT = 150
buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_state · 状态机 + 条件概率/期望（概率博弈）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
      'low': 'float32', 'quote_volume': 'float32', 'trades': 'float32'}
parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=1_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:N_COIN_LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= N_COIN_LIMIT:
                break
            seen = seen + new[:N_COIN_LIMIT - len(seen)]
    parts.append(ch[ch['symbol'].isin(seen)])
df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
df['symbol'] = df['symbol'].astype('category')
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0:.0f}s')

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values
def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

S_NAMES = ['S0静默', 'S1吸筹', 'S2拉升', 'S3高位滞涨', 'S4崩塌', 'S5其他']
GS, AS, KS = [], [], []      # 状态序列 / 未来24h最大涨幅 / 未来24h最小跌幅

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 2000:
        continue
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64')
    tr = g['trades'].values.astype('float64')
    ot = g['open_time'].values
    ret6 = np.full(n, np.nan); ret6[6:] = c[6:] / c[:-6] - 1.0
    ret24_ = np.full(n, np.nan); ret24_[288:] = c[288:] / c[:-288] - 1.0
    vq = np.where(rmean(q, 48) > 0, q / np.maximum(rmean(q, 48), 1e-12), np.nan)
    vtr = np.where(rmean(tr, 48) > 0, tr / np.maximum(rmean(tr, 48), 1e-12), np.nan)
    hh = rmax(h, 288)
    near_hi = (hh - c) / np.maximum(c, 1e-12) < 0.03

    s = np.full(n, 5, dtype='int8')
    valid = np.isfinite(ret6) & np.isfinite(vq) & np.isfinite(vtr) & np.isfinite(c)
    s[valid & (vq < 1.5) & (np.abs(np.nan_to_num(ret24_, nan=0.0)) < 0.03)] = 0
    s[valid & (vtr >= 4) & (vq >= 3) & (np.abs(ret6) < 0.05)] = 1
    s[valid & (vtr >= 4) & (vq >= 3) & np.isfinite(ret24_) & (ret24_ > 0.05)] = 5  # 放量上涨另计
    s[valid & (ret6 >= 0.10) & (vq >= 2)] = 2
    s[valid & near_hi & (vq >= 3) & (np.abs(ret6) < 0.05)] = 3
    s[valid & (ret6 <= -0.10) & (vq >= 2)] = 4

    fmax = pd.Series(h).rolling(288, min_periods=1).max().shift(-288).values
    fmin = pd.Series(l).rolling(288, min_periods=1).min().shift(-288).values
    gain = fmax / c - 1.0
    drop = 1.0 - fmin / c
    m = valid & np.isfinite(gain) & np.isfinite(drop)
    # 每 3 根取 1，控内存
    idx = np.flatnonzero(m)[::3]
    GS.append(s[idx]); AS.append(gain[idx]); KS.append(drop[idx])

G = np.concatenate(GS); A = np.concatenate(AS); K = np.concatenate(KS)
del GS, AS, KS; gc.collect()
log(f'状态样本 {len(G):,}  耗时 {time.time()-t0:.0f}s')

log('')
log('=' * 100)
log('【A】状态分布（当前时刻）')
log('=' * 100)
tot = len(G)
for i, nm in enumerate(S_NAMES):
    c_ = int((G == i).sum())
    log(f'  {nm:10s}  {c_:>10,}  {100*c_/tot:>6.2f}%')

log('')
log('=' * 100)
log('【B】条件分布：P(未来24h内 >=+50%) / P(<=-30%) / E[未来24h最大涨幅] / E[最大跌幅]')
log('=' * 100)
log(f'{"状态":12s} {"样本":>10s} {"P(+50%)":>9s} {"P(-30%)":>9s} {"E[最大涨]":>10s} {"E[最大跌]":>10s} {"中位涨":>9s} {"中位跌":>9s}')
for i, nm in enumerate(S_NAMES):
    m = G == i
    if m.sum() < 100:
        log(f'{nm:12s} {int(m.sum()):>10,}  (样本过少)')
        continue
    log(f'{nm:12s} {int(m.sum()):>10,} {100*(A[m]>=0.50).mean():>8.2f}% {100*(K[m]>=0.30).mean():>8.2f}% '
        f'{100*A[m].mean():>9.2f}% {100*K[m].mean():>9.2f}% {100*np.median(A[m]):>8.2f}% {100*np.median(K[m]):>8.2f}%')

log('')
log('=' * 100)
log('【C】无条件基准（全样本）')
log('=' * 100)
log(f'  P(+50%)={100*(A>=0.50).mean():.2f}%  P(-30%)={100*(K>=0.30).mean():.2f}%  '
    f'E[最大涨]={100*A.mean():.2f}%  E[最大跌]={100*K.mean():.2f}%')

log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
