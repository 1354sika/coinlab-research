# -*- coding: utf-8 -*-
"""_probe_syncdn.py — 深挖 A 阶段第一名：市场齐跌(syncDn) 的蝴蝶效应

A 阶段（核心池 50 条件，79,954 条全穷举）测试段排序 top-25 几乎全部含
    `syncDn`（全市场齐跌币数） 与 `hour12~18`
第一名：hour12~18 & over>=15% & pos>=0.5 & syncDn>=3
        训练 lift 5.8x → 测试 lift 22.0x，测试真局率 23.81%（基准 1.08%）

本脚本回答三件事（全部无前视）：
  ① syncDn 的**边际贡献**：控制住 over/hour 后，syncDn 从 0 到 3+ 真局率怎么变？
  ② 二维表：syncDn 档 × 单币 over 档 → 真局率 / lift / n
  ③ **能不能赚钱**：syncDn>=3 触发 → 买 → +50%止盈/−15%止损/24h超时（扣 0.09%）
     与随机、与"不用 syncDn（只看 over>=15%）"对照

产物：D:\中信证券\coinlab\evidence\butterfly\syncdn_probe.txt
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
OUTDIR = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY, H48 = 288, 576
FEE = 0.0009

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.append(s)

log('=' * 100)
log('_probe_syncdn · 市场齐跌（syncDn）的蝴蝶效应深挖')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
                             'low': 'float32', 'quote_volume': 'float32', 'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
Cw = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = Cw.index.values; T, N = Cw.shape
R12 = Cw.pct_change(12).values
up3 = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
dn3 = (np.nan_to_num(R12, nan=0.0) <= -0.05).sum(axis=1)
sync_dn_now = pd.Series(dn3).rolling(24, min_periods=1).max().values     # 过去 2h 内齐跌币数最大值
sync_up_now = pd.Series(up3).rolling(24, min_periods=1).max().values
btc = Cw['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
log(f'币 {len(syms)}  矩阵 {Cw.shape}  耗时 {time.time()-t0t:.0f}s')
del Cw; gc.collect()

rows = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r288 = np.full(n, np.nan); r288[288:] = c[288:] / c[:-288] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    mt = trs.rolling(24, min_periods=6).mean().values
    trr24 = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    for t in np.flatnonzero(np.isfinite(f_up)):
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    ii = pd.Index(ot)
    sd = pd.Series(sync_dn_now, index=ts).reindex(ii).values
    su = pd.Series(sync_up_now, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24

    # 交易模拟：t 时刻入场，+50% 止盈 / -15% 止损 / 24h(288根) 超时
    rets = np.full(n, np.nan)
    for t in np.flatnonzero(np.isfinite(f_up)):
        if not (np.isfinite(over[t]) and 0.03 <= over[t] <= 0.40 and np.isfinite(volr24[t]) and volr24[t] >= 1):
            continue
        p0 = c[t]; e = min(n - 1, t + WDAY)
        seg = c[t+1:e+1]
        if not len(seg): continue
        tp = p0 * 1.50; sl = p0 * 0.85
        hit = np.flatnonzero((seg >= tp) | (seg <= sl))
        if len(hit):
            k = hit[0]; rets[t] = (tp / p0 - 1) if seg[k] >= tp else (sl / p0 - 1)
        else:
            rets[t] = seg[-1] / p0 - 1
        rets[t] -= FEE

    sel = np.isfinite(over) & (over >= 0.03) & (over <= 0.40) & np.isfinite(volr24) & (volr24 >= 1) \
          & np.isfinite(f_up) & np.isfinite(dd) & np.isfinite(sd) & np.isfinite(rets)
    idx = np.flatnonzero(sel)[::3]
    if len(idx) < 200: continue
    rows.append(pd.DataFrame({
        't': ot[idx], 'sym': sym, 'over': over[idx], 'r288': r288[idx],
        'volr24': volr24[idx], 'trr24': trr24[idx], 'hour': hh[idx],
        'syncDn': sd[idx], 'syncUp': su[idx], 'ret': rets[idx],
        'true': ((f_up[idx] >= 0.50) & (dd[idx] >= 0.30)).astype('int8')}))
    del hs, ls, qs, trs; gc.collect()

df = pd.concat(rows, ignore_index=True)
n_all = len(df); Y = df['true'].values.astype('int8')
cut_t = np.quantile(df['t'].values, 0.70)
TR = df['t'].values <= cut_t; TE = ~TR
btr, bte = Y[TR].mean(), Y[TE].mean()
log(f'候选样本 {n_all:,}  基准真局率 全 {100*Y.mean():.3f}% | 训练 {100*btr:.4f}% | 测试 {100*bte:.4f}%')
log(f'训练 {int(TR.sum()):,}  测试 {int(TE.sum()):,}   耗时 {time.time()-t0t:.0f}s')

log('\n' + '=' * 100)
log('【① syncDn 的边际贡献：控制 over 在 15~40%（都是已大涨的币）后，syncDn 从低到高）')
log('=' * 100)
sub = df[(df['over'] >= 0.15) & (df['over'] <= 0.40)].copy()
for lo, hi, nm in [(0, 1, 'syncDn=0'), (1, 3, 'syncDn 1~2'), (3, 6, 'syncDn 3~5'), (6, 999, 'syncDn>=6')]:
    m = (sub['syncDn'] >= lo) & (sub['syncDn'] < hi)
    a = sub[m]
    if len(a) < 50: log(f'  {nm:<14} n={len(a):>6,} 样本太少'); continue
    tr = a[a['t'] <= cut_t]; te = a[a['t'] > cut_t]
    log(f'  {nm:<14} n={len(a):>7,} 真局率 {100*a["true"].mean():>6.2f}%  '
        f'| 训练 {100*tr["true"].mean() if len(tr) else np.nan:>6.2f}%(n={len(tr):>6,})  '
        f'| 测试 {100*te["true"].mean() if len(te) else np.nan:>6.2f}%(n={len(te):>6,})  '
        f'| 期望净 {100*a["ret"].mean():>6.2f}%')

log('\n' + '=' * 100)
log('【② 二维：syncDn 档 × over 档（测试段真局率 / lift）')
log('=' * 100)
obins = [(0.03, 0.08), (0.08, 0.15), (0.15, 0.25), (0.25, 0.40)]
dbins = [(0, 1), (1, 3), (3, 6), (6, 999)]
hdr = '  syncDn\\over    ' + '  '.join(f'{a:.0%}~{b:.0%}'.rjust(16) for a, b in obins)
log(hdr)
for dlo, dhi in dbins:
    line = f'  {dlo:>2}~{dhi if dhi<999 else "+":<3}        '
    for olo, ohi in obins:
        m = (df['syncDn'] >= dlo) & (df['syncDn'] < dhi) & (df['over'] >= olo) & (df['over'] < ohi) & TE
        a = df[m]
        if len(a) < 30:
            line += f'{"—":>16}  '; continue
        line += f'{100*a["true"].mean():>6.2f}%({len(a):>5,})  '
    log(line)

log('\n' + '=' * 100)
log('【③ 能不能赚钱：三种入场对照（+50%止盈/-15%止损/24h超时，扣0.09%）')
log('=' * 100)
def sim(mask, nm):
    a = df[mask]
    r = a['ret'].values
    if len(a) < 30: log(f'  {nm:<44} n={len(a):>6,} 样本太少'); return
    te = a['t'].values > cut_t
    log(f'  {nm:<44} n={len(a):>7,} 期望净 {100*np.nanmean(r):>6.2f}%  中位 {100*np.nanmedian(r):>6.2f}%  '
        f'胜率 {100*np.nanmean(r > 0):>5.1f}%  ≥+50% {100*np.nanmean(r >= 0.49):>4.1f}%  '
        f'| 测试段 n={int(te.sum()):>6,} 期望净 {100*np.nanmean(r[te]):>6.2f}%')

sim(df['over'] >= 0.15, 'A 只看 over>=15%（不用 syncDn）')
sim((df['over'] >= 0.15) & (df['syncDn'] >= 3), 'B over>=15% 且 syncDn>=3  ★第一名特征')
sim((df['over'] >= 0.15) & (df['syncDn'] >= 3) & (df['hour'] >= 12) & (df['hour'] < 18), 'C 再加 hour12~18')
sim((df['over'] >= 0.15) & (df['syncDn'] >= 3) & (df['hour'] >= 12) & (df['hour'] < 18)
    & (df['r288'] >= 0.30), 'D 再加 r288>=30%')
sim((df['over'] >= 0.15) & (df['syncUp'] >= 3), 'E 对照：改看 syncUp>=3（市场齐涨）')
sim((df['over'] >= 0.15) & (df['syncDn'] < 1), 'F 对照：syncDn=0')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'syncdn_probe.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {os.path.join(OUTDIR, "syncdn_probe.txt")}')
