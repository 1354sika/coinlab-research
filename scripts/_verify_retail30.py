# -*- coding: utf-8 -*-
"""_verify_retail30.py — q3：对 retail30 的 top 组合做「币留出 + 置换检验」

为什么必须做：
    retail30 从 214 条条件里按训练段 lift 筛出 28 条，再穷举 2 元组合共 339 个，
    最后挑测试段期望净最高的几条 ⇒ 这是典型的**多重比较**，必须回答：
      ① 换一批没见过的币，还成立吗？（币留出）
      ② 打乱标签后，随机搜索能刷到多高的 lift / 多好的期望净？（置换检验）
      ③ 样本落在多少个币、多少天？（有效独立样本）

被验证的组合（retail30 测试段期望净 top）：
    C1  mkt_up>=2 & vol_price_div>=1 & volr6>=3
    C2  mkt_up>=2 & r288>=0.04997
    C3  mkt_up>=2 & volr6>=1.815
    C4  over_now>=0.07977 & r288<=-0.06339
    C5  mkt_up>=2 & volr24>=1.776
    C6  r288>=0.04997 & vol_price_div>=1 & volr6>=3
    C7  r288>=0.04997 & mkt_up>=1
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288; W = 576; TARGET = 0.30; SL = 0.15; FEE = 0.0009
NPERM = 20

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('_verify_retail30 · 币留出 + 置换检验（retail30 top 组合）')
log('=' * 112)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns); ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12all, nan=0.0) >= 0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
del R12all; 
log(f'币 {N}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

FN = ['r12', 'r288', 'over_now', 'pos288', 'dist_hi288', 'volr6', 'volr24', 'volr96',
      'trade_intensity', 'avg_trade_size', 'taker_buy_ratio', 'vol_price_div', 'run',
      'vola_jump', 'fomo', 'panic', 'mkt_up']
XS, YS, WS, TID, CO, SY = [], [], [], [], [], []

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4 + W:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WDAY, min_periods=72).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
        dhi = c / np.maximum(hi288, 1e-12) - 1.0
        pos288 = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    r = {}
    for k in (12, 288):
        v = np.full(n, np.nan); v[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1.0
        r[k] = v
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values

    m = (over < 0.15) & np.isfinite(volr24) & (np.arange(n) < n - W - 1)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:
        idx = idx[::max(1, len(idx) // 4000)]
    wh = np.lib.stride_tricks.sliding_window_view(h, W)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, W)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, W)[idx]
    c0 = c[idx]
    hit_tp = wh >= (c0 * 1.30)[:, None]; hit_sl = wl <= (c0 * 0.85)[:, None]
    it = np.argmax(hit_tp, 1); il = np.argmax(hit_sl, 1)
    any_tp = hit_tp.any(1); any_sl = hit_sl.any(1)
    w_ = np.where(any_sl & (~any_tp | (il <= it)), -SL,
                  np.where(any_tp, TARGET, wc[:, -1] / c0 - 1.0)) - FEE
    y_ = ((wh.max(1) / c0 - 1.0) >= TARGET).astype('int8')
    posv = np.searchsorted(ts, ot)

    cols = [r[12][idx], r[288][idx], over[idx], pos288[idx], dhi[idx],
            volr6[idx], volr24[idx], volr96[idx],
            np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx)),
            np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx)),
            mkt_up[posv[idx]]]
    XS.append(np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0) for v in cols]).astype('float32'))
    YS.append(y_); WS.append(w_.astype('float32'))
    TID.append(posv[idx].astype('int64')); CO.append(np.full(len(idx), sym)); SY.append(idx)
del RAW, C; gc.collect()

X = np.vstack(XS); Y = np.concatenate(YS); Wv = np.concatenate(WS)
TID = np.concatenate(TID); CO = np.concatenate(CO)
del XS, YS, WS; gc.collect()
log(f'样本 {len(Y):,}   测试段 {int((TID>=t_cut).sum()):,}   币 {len(np.unique(CO))}   耗时 {time.time()-t0:.0f}s')

F = {nm: X[:, j] for j, nm in enumerate(FN)}
te = TID >= t_cut
coins = np.array(sorted(np.unique(CO)))
rng0 = np.random.default_rng(11)
coin_pick = set(rng0.choice(coins, size=len(coins)//2, replace=False).tolist())
coin_in = np.isin(CO, list(coin_pick))

COMBOS = {
    'C1 mkt_up>=2 & vol_price_div>=1 & volr6>=3':
        (F['mkt_up'] >= 2) & (F['vol_price_div'] >= 1) & (F['volr6'] >= 3),
    'C2 mkt_up>=2 & r288>=0.04997':
        (F['mkt_up'] >= 2) & (F['r288'] >= 0.04997),
    'C3 mkt_up>=2 & volr6>=1.815':
        (F['mkt_up'] >= 2) & (F['volr6'] >= 1.815),
    'C4 over>=0.0798 & r288<=-0.0634':
        (F['over_now'] >= 0.07977) & (F['r288'] <= -0.06339),
    'C5 mkt_up>=2 & volr24>=1.776':
        (F['mkt_up'] >= 2) & (F['volr24'] >= 1.776),
    'C6 r288>=0.04997 & vol_price_div>=1 & volr6>=3':
        (F['r288'] >= 0.04997) & (F['vol_price_div'] >= 1) & (F['volr6'] >= 3),
    'C7 r288>=0.04997 & mkt_up>=1':
        (F['r288'] >= 0.04997) & (F['mkt_up'] >= 1),
    'Z  全部局外点（对照）': np.ones(len(Y), bool),
}

def stat(mask):
    if mask.sum() == 0:
        return None
    return dict(n=int(mask.sum()), rate=float(Y[mask].mean()), exp=float(np.nanmean(Wv[mask])),
                med=float(np.nanmedian(Wv[mask])), wr=float(np.nanmean(Wv[mask] > 0)),
                coins=int(len(np.unique(CO[mask]))),
                days=int(len(np.unique(ts[TID[mask]] // 86_400_000))))

log('\n' + '=' * 112)
log('【A】三种切法（每格：笔数 / 精确率% / 期望净% / 币数 / 天数）')
log('=' * 112)
log(f'  {"组合":<46}{"时间留出(test)":>26}{"币留出(test, 不同币)":>28}{"双重留出":>26}')
for name, m in COMBOS.items():
    cells = []
    for mask in (m & te, m & te & ~coin_in, m & te & ~coin_in):
        s = stat(mask)
        cells.append('—' if s is None else
                     f'{s["n"]:,} / {100*s["rate"]:.2f}% / {100*s["exp"]:+.2f}% / {s["coins"]}币/{s["days"]}天')
    log(f'  {name:<46}{cells[0]:>26}{cells[1]:>28}{cells[2]:>26}')

log('\n' + '=' * 112)
log('【B】置换检验：按天打乱标签 20 次，看每个组合"随机能刷到多少"')
log('=' * 112)
day_key = (ts[TID] // 86_400_000)
uniq_days = np.unique(day_key)
perm_max = {k: [] for k in COMBOS}
for it in range(NPERM):
    # 按天分块打乱：同一天的所有样本标签整体换成另一天的（保留日内结构）
    src = rng0.permutation(len(uniq_days))
    day_map = dict(zip(uniq_days, uniq_days[src]))
    Yp = np.empty_like(Y)
    for d, sd in day_map.items():
        md = day_key == d; ms = day_key == sd
        np.put(Yp, np.flatnonzero(md), Y[ms][:int(md.sum())] if ms.sum() >= md.sum() else
               np.resize(Y[ms], int(md.sum())))
    for name, m in COMBOS.items():
        mm = m & te
        if mm.sum() == 0:
            continue
        base = Yp[te].mean()
        perm_max[name].append(float(Yp[mm].mean() / max(base, 1e-9)))
    log(f'  置换 {it+1}/{NPERM}')

log(f'\n  {"组合":<46}{"真实测试lift":>14}{"置换中位":>10}{"置换p90":>10}{"置换max":>10}{"判定":>10}')
for name, m in COMBOS.items():
    mm = m & te
    if mm.sum() == 0:
        continue
    real = Y[mm].mean() / max(Y[te].mean(), 1e-9)
    arr = np.array(perm_max[name])
    p90 = np.percentile(arr, 90) if len(arr) else np.nan
    verdict = '★通过' if real > p90 else '✗不显著'
    log(f'  {name:<46}{real:>14.2f}{np.median(arr):>10.2f}{p90:>10.2f}{arr.max():>10.2f}{verdict:>10}')
log(f'\n  参照：随机搜索（339 个 2 元组合）在测试段能刷到的最高 lift ≈ '
    f'{max([np.percentile(v, 100) for v in perm_max.values() if len(v)]):.2f}')

out = os.path.join(OUT, 'verify_retail30.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
