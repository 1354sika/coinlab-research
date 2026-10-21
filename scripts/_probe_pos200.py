# -*- coding: utf-8 -*-
"""_probe_pos200.py — 验证 pos_trade 里「已涨 200%+ → 期望净 +57%」是真信号还是假象

疑点（必须先证伪再用）：
    job-24708-17 测试段：位置 200%+ 的 469 笔，R3 期望净 +57.08%、胜率 87%
    而 job-24708-14 stage 同口径下 200%+ 的 P(再拿30%) 只有 0.00%（34 样本）
    两者样本量差 14 倍 ⇒ pos_trade 没加「自高点回落<50%」约束，
    很可能把「已从高点崩掉一半以上的残局」也算进来了（崩后反弹 ≠ 局中段）。

本探针回答三件事：
    ① 200%+ 档里，「未崩完（dist_hi288 > -50%）」vs「已崩过（<= -50%）」各占多少、各多少期望净
    ② 每个 位置×状态 格子里，样本落在多少个不同币、多少个不同日期（有效独立样本远小于总笔数）
    ③ 时间分布（按年）—— 若是单一行情段的产物，则不可外推
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288; W = 576; FEE = 0.0009

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 108)
log('_probe_pos200 · 验证「已涨 200%+ 期望净 +57%」')
log('=' * 108)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
log(f'币 {N}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

OV, DHI, TID, SY, R3, R1, PK = [], [], [], [], [], [], []

for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < WDAY * 4 + W:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WDAY, min_periods=72).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
        dhi = c / np.maximum(hi288, 1e-12) - 1.0
    ok = np.isfinite(over) & (over >= 1.0) & (np.arange(n) < n - W - 1)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    wh = np.lib.stride_tricks.sliding_window_view(h, W)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, W)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, W)[idx]
    c0 = c[idx]
    # R1 +30/-15/48h
    tp1 = c0 * 1.30; sl1 = c0 * 0.85
    ht = (wh >= tp1[:, None]).any(1); hl = (wl <= sl1[:, None]).any(1)
    it = np.argmax(wh >= tp1[:, None], axis=1); il = np.argmax(wl <= sl1[:, None], axis=1)
    v1 = np.where(hl & (~ht | (il <= it)), -0.15, np.where(ht, 0.30, wc[:, -1] / c0 - 1.0)) - FEE
    # R3 移动止损 回落30%
    cummax = np.maximum.accumulate(wh, axis=1)
    dd = 1.0 - wc / np.maximum(cummax, 1e-12)
    trig = dd >= 0.30
    it3 = np.minimum(np.argmax(trig, axis=1), W - 1)
    h3 = trig.any(1)
    px3 = np.take_along_axis(wl, it3[:, None], axis=1)[:, 0]
    v3 = np.where(h3, px3 / c0 - 1.0, wc[:, -1] / c0 - 1.0) - FEE

    OV.append(over[idx].astype('float32')); DHI.append(dhi[idx].astype('float32'))
    TID.append(np.searchsorted(ts, ot)[idx].astype('int64'))
    SY.append(np.full(len(idx), sym)); R1.append(v1.astype('float32')); R3.append(v3.astype('float32'))
    PK.append((wh.max(1) / c0 - 1.0).astype('float32'))
    del wh, wl, wc
del RAW, C; gc.collect()

OV = np.concatenate(OV); DHI = np.concatenate(DHI); TID = np.concatenate(TID)
SY = np.concatenate(SY); R1 = np.concatenate(R1); R3 = np.concatenate(R3); PK = np.concatenate(PK)
te = TID >= t_cut
log(f'200%+ 样本 {len(OV):,}（测试段 {int(te.sum()):,}）   耗时 {time.time()-t0:.0f}s')

log('\n【① 未崩完 vs 已崩过（自 288 根高点回落 >50%）】测试段')
log(f'  {"状态":<22}{"笔数":>8}{"币数":>7}{"天数":>7}{"R1期望%":>10}{"R3期望%":>10}{"R3中位%":>10}{"峰值中位%":>11}')
for lbl, m in (('未崩完 dhi>-50%', DHI > -0.50), ('已崩过 dhi<=-50%', DHI <= -0.50), ('全部', np.ones_like(te, bool))):
    mm = te & m
    if mm.sum() == 0:
        continue
    days = np.unique((ts[TID[mm]] // 86_400_000))
    log(f'  {lbl:<22}{int(mm.sum()):>8}{len(np.unique(SY[mm])):>7}{len(days):>7}'
        f'{100*np.nanmean(R1[mm]):>+10.2f}{100*np.nanmean(R3[mm]):>+10.2f}'
        f'{100*np.nanmedian(R3[mm]):>10.2f}{100*np.nanmedian(PK[mm]):>11.1f}')

log('\n【② 按年】测试段 全部 200%+（看是否单一行情产物）')
yr = pd.to_datetime(ts[TID[te]], unit='ms').year
for y in sorted(np.unique(yr)):
    m = yr == y
    log(f'  {y}  笔{int(m.sum()):>7,}  币{len(np.unique(SY[te][m])):>4}  '
        f'R1 {100*np.nanmean(R1[te][m]):>+8.2f}%  R3 {100*np.nanmean(R3[te][m]):>+8.2f}%')

log('\n【③ 按 位置档 × 状态】测试段，R3（移动止损）')
log(f'  {"位置档":<10}{"状态":<18}{"笔数":>8}{"币数":>7}{"天数":>7}{"R3期望%":>10}{"中位%":>9}{"胜率%":>8}')
for a, b, t in ((0.15, 0.40, '15~40%'), (0.40, 1.00, '40~100%'), (1.00, 2.00, '100~200%'), (2.00, 99., '200%+')):
    for lbl, m in (('未崩完', DHI > -0.50), ('已崩过', DHI <= -0.50)):
        mm = te & (OV >= a) & (OV < b) & m
        if mm.sum() < 30:
            continue
        days = np.unique((ts[TID[mm]] // 86_400_000))
        v = R3[mm]
        log(f'  {t:<10}{lbl:<18}{int(mm.sum()):>8}{len(np.unique(SY[mm])):>7}{len(days):>7}'
            f'{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}{100*np.nanmean(v > 0):>8.1f}')

out = os.path.join(OUT, 'probe_pos200.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
