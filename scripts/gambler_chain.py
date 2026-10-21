# -*- coding: utf-8 -*-
"""gambler_chain.py — 带时延的事件链（蝴蝶效应：谁先动、多久传到谁）
用户设想（2026-10-16）：蝴蝶效应才是 gambler 的关键 —— 这件事发生了，其他事发生的概率是多少。
上一版 cascade 只算了"同一时刻"的成对关联；这一版加**时延 τ**：
  因事件 A 发生 → 隔 τ 根 K 线 → 果事件 B 发生
扫描 τ ∈ {1,3,6,12,24,48}（5min~4h），找出"传导最强"的滞后。
重点回答：谁先动（BTC？大币？市场？），多久传到单币。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_chain.txt')
LIMIT = 303
H = 288
TAUS = (1, 3, 6, 12, 24, 48)

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_chain · 带时延的事件链（τ = 5min ~ 4h）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'quote_volume']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32', 'quote_volume': 'float32'}
parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=3_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= LIMIT:
                break
            seen = seen + new[:LIMIT - len(seen)]
    parts.append(ch[ch['symbol'].isin(seen)])
df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0t:.0f}s')

piv = {}
for col in ('close', 'high', 'quote_volume'):
    piv[col] = df.pivot_table(index='open_time', columns='symbol', values=col, observed=True)
C = piv['close'].values.astype('float32')
Hg = piv['high'].values.astype('float32')
Q = piv['quote_volume'].values.astype('float32')
cols_ = list(piv['close'].columns)
del df, piv; gc.collect()
T, N = C.shape
log(f'矩阵 {T:,} × {N}  耗时 {time.time()-t0t:.0f}s')

r1 = np.full_like(C, np.nan); r1[12:] = C[12:] / C[:-12] - 1.0
r6 = np.full_like(C, np.nan); r6[6:] = C[6:] / C[:-6] - 1.0
hmax = pd.DataFrame(Hg).rolling(H, min_periods=1).max().shift(-H).values
fup = hmax / C - 1.0
lmin = pd.DataFrame(C).rolling(H, min_periods=1).min().shift(-H).values
fdn = 1.0 - lmin / C
qm = pd.DataFrame(Q).rolling(48, min_periods=12).mean().values
vratio = np.where(qm > 0, Q / np.maximum(qm, 1e-12), np.nan)
log(f'特征算完 耗时 {time.time()-t0t:.0f}s')

# 参考基准：未来 24h 涨>=50% / 跌>=30% 的无条件概率
base_up = np.nanmean(fup >= 0.50)
base_dn = np.nanmean(fdn >= 0.30)
log(f'基准：涨>=50% {100*base_up:.3f}%   跌>=30% {100*base_dn:.3f}%')

# 因事件（跨币/市场级）
btc_i = 0
btc_r1 = r1[:, btc_i]
up_cnt = np.nansum(r1 >= 0.05, axis=1)
dn_cnt = np.nansum(r1 <= -0.05, axis=1)
VC = np.nan_to_num(vratio, nan=0.0)

SRC_EV = {
    'BTC 1h 跌>=2%':        np.repeat((btc_r1 <= -0.02)[:, None], N, axis=1),
    'BTC 1h 涨>=2%':        np.repeat((btc_r1 >= 0.02)[:, None], N, axis=1),
    '市场 >=10币齐跌>=5%':   np.repeat((dn_cnt >= 10)[:, None], N, axis=1),
    '市场 >=10币齐涨>=5%':   np.repeat((up_cnt >= 10)[:, None], N, axis=1),
    '同币量比>=10':          (VC >= 10),
    '同币 1h 涨>=5%':        (r1 >= 0.05),
    '同币 1h 跌>=5%':        (r1 <= -0.05),
}

# 果事件
R = {
    '单币未来24h涨>=50%': (fup >= 0.50),
    '单币未来24h跌>=30%': (fdn >= 0.30),
}

log('')
log('=' * 100)
log('【A】带时延的传导：因事件发生在 t，果事件在 t+τ 的窗口内发生')
log('  （lift = P(果在 t..t+τ 内 | 因在 t) / P(果在 τ 窗内)）')
log('=' * 100)
for en, Em in SRC_EV.items():
    log('')
    log(f'因：{en}')
    hdr = f'    {"τ":>6s}'
    for rn in R:
        hdr += f' {rn:>22s}'
    log(hdr)
    for tau in TAUS:
        # 果：t..t+tau 内是否出现
        hit_up = np.zeros_like(fup, dtype=bool)
        hit_dn = np.zeros_like(fdn, dtype=bool)
        for k in range(0, tau + 1):
            if k >= T:
                continue
            hu = np.zeros_like(fup, dtype=bool); hd = np.zeros_like(fdn, dtype=bool)
            hu[:T - k] = fup[k:] >= 0.50
            hd[:T - k] = fdn[k:] >= 0.30
            hit_up |= hu; hit_dn |= hd
        line = f'    {tau*5:>4d}m'
        for rn, Rm in (('up', hit_up), ('dn', hit_dn)):
            both = Em & Rm; na = int(np.nansum(Em)); nb = int(np.nansum(both))
            nbase = int(np.nansum(Rm))
            if na < 100 or nbase == 0:
                line += f' {"-":>22s}'; continue
            pr_a = nb / na; pr = nbase / Rm.size
            line += f' {pr_a/pr:>14.2f}x({100*pr_a:>5.2f}%)'
        log(line)

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
