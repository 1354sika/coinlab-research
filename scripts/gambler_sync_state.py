# -*- coding: utf-8 -*-
"""gambler_sync_state.py — 状态 × 横截面协同（"币是一起动"的量化）
用户假设（2026-10-16）：这些币是一起动的；懂快慢才是关联性。
做法：
  1. 把 150 币对齐到同一时间轴，给每根 K 线打状态（复用 gambler_state 的定义）
  2. 每个时刻统计：全市场处于 S2(拉升) / S4(崩塌) 的币数 k
  3. 条件概率：P(该币未来24h内 >=+50% | 当下 S2 币数 = k)
  4. 对照：单币状态 vs 全体状态，看"协同"能否提高命中概率
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_sync_state.txt')
N_COIN_LIMIT = 150
buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_sync_state · 状态 × 横截面协同')
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

# 逐币算状态与时序，堆成 (T, N) 矩阵（按 open_time 对齐）
sta = {}     # open_time -> (状态, 未来24h最大涨幅)
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
    valid = np.isfinite(ret6) & np.isfinite(vq) & np.isfinite(vtr)
    s[valid & (vq < 1.5) & (np.abs(np.nan_to_num(ret24_, nan=0.0)) < 0.03)] = 0
    s[valid & (vtr >= 4) & (vq >= 3) & (np.abs(ret6) < 0.05)] = 1
    s[valid & (ret6 >= 0.10) & (vq >= 2)] = 2
    s[valid & near_hi & (vq >= 3) & (np.abs(ret6) < 0.05)] = 3
    s[valid & (ret6 <= -0.10) & (vq >= 2)] = 4
    fmax = pd.Series(h).rolling(288, min_periods=1).max().shift(-288).values
    fmin = pd.Series(l).rolling(288, min_periods=1).min().shift(-288).values
    gain = fmax / c - 1.0
    drop = 1.0 - fmin / c
    m = valid & np.isfinite(gain) & np.isfinite(drop)
    sta[os.path.basename(sym)] = (ot[m], s[m], gain[m], drop[m])

log(f'逐币状态算完 {len(sta)} 币  耗时 {time.time()-t0:.0f}s')

# 对齐：以第一币的时间轴为基准，做并集
allt = None
for k, (ot, s, g, d) in sta.items():
    allt = ot if allt is None else np.union1d(allt, ot)
idx = {t: i for i, t in enumerate(allt)}
T = len(allt)
log(f'时间轴 {T:,} 根')

S = np.full((T, len(sta)), -1, dtype='int8')
G = np.full((T, len(sta)), np.nan, dtype='float32')
DR = np.full((T, len(sta)), np.nan, dtype='float32')
for j, (k, (ot, s, g, d)) in enumerate(sta.items()):
    ii = np.array([idx[t] for t in ot])
    S[ii, j] = s; G[ii, j] = g; DR[ii, j] = d
del sta; gc.collect()

nS2 = (S == 2).sum(axis=1)      # 该时刻处于拉升的币数
nS4 = (S == 4).sum(axis=1)
log(f'各时刻 S2 币数：中位 {np.median(nS2):.0f}  p90 {np.quantile(nS2,0.9):.0f}  max {nS2.max()}')
log(f'各时刻 S4 币数：中位 {np.median(nS4):.0f}  p90 {np.quantile(nS4,0.9):.0f}  max {nS4.max()}')

log('')
log('=' * 100)
log('【A】P(单币未来24h >=+50%)  vs  当下全市场 S2 币数')
log('=' * 100)
log(f'{"全市场S2币数":>12s} {"样本":>12s} {"P(+50%)":>10s} {"相对基准":>10s}')
base = np.nanmean((G >= 0.50)[S >= 0])
log(f'{"无条件基准":>12s} {"-":>12s} {100*base:>9.3f}% {1.0:>9.2f}x')
for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 9999)):
    m = (nS2 >= lo) & (nS2 <= hi)
    mask = m[:, None] & (S >= 0) & np.isfinite(G)
    nn = int(mask.sum())
    if nn < 500:
        log(f'{f"{lo}-{hi}":>12s} {nn:>12,}  (样本过少)')
        continue
    p = (G[mask] >= 0.50).mean()
    log(f'{f"{lo}-{hi}":>12s} {nn:>12,} {100*p:>9.3f}% {p/base:>9.2f}x')

log('')
log('=' * 100)
log('【B】P(单币未来24h <=-30%)  vs  当下全市场 S4 币数')
log('=' * 100)
baseK = np.nanmean((DR >= 0.30)[S >= 0])
log(f'{"全市场S4币数":>12s} {"样本":>12s} {"P(-30%)":>10s} {"相对基准":>10s}')
log(f'{"无条件基准":>12s} {"-":>12s} {100*baseK:>9.3f}% {1.0:>9.2f}x')
for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 9999)):
    m = (nS4 >= lo) & (nS4 <= hi)
    mask = m[:, None] & (S >= 0) & np.isfinite(DR)
    nn = int(mask.sum())
    if nn < 500:
        log(f'{f"{lo}-{hi}":>12s} {nn:>12,}  (样本过少)')
        continue
    p = (DR[mask] >= 0.30).mean()
    log(f'{f"{lo}-{hi}":>12s} {nn:>12,} {100*p:>9.3f}% {p/max(baseK,1e-9):>9.2f}x')

log('')
log('=' * 100)
log('【C】二维：单币状态 × 全市场 S2 币数（>=3 时为"共振"）')
log('=' * 100)
S_NAMES = {0: 'S0静默', 1: 'S1吸筹', 2: 'S2拉升', 3: 'S3高位滞涨', 4: 'S4崩塌', 5: 'S5其他'}
log(f'{"单币状态":12s} {"S2币数<3":>22s} {"S2币数>=3":>22s}')
for k in (0, 1, 2, 3, 4, 5):
    row = f'{S_NAMES[k]:12s}'
    for cond in (nS2 < 3, nS2 >= 3):
        mask = cond[:, None] & (S == k) & np.isfinite(G)
        nn = int(mask.sum())
        row += f'  {nn:>10,} {100*(G[mask]>=0.50).mean():>6.2f}%' if nn >= 200 else f'  {nn:>10,}  (少)'
    log(row)

log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
