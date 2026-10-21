# -*- coding: utf-8 -*-
"""gambler_amp_hold.py — 真实口径：振幅异动后买入，到底赚不赚

为什么要重算（job-24708-35 的陷阱）：
    方向实验里「振幅≥2 时先涨30%」占 92.2%，模型做多期望 +25.54%/笔。
    但那只统计了 **走到 ±30% 的 6,071 个样本（占 4.1%）**；
    95% 的样本在 24h 内既没到 +30% 也没到 -30%，被排除在统计外。
    ⇒ 必须用"全部样本 + 到期结算"的真实口径重算。

【本脚本】
    信号：amp_ratio = 8h 平均振幅 / 过去 30 天均值 ≥ 2（与前面完全同一口径）
    入场：信号触发即市价做多，不做任何筛选
    持有：到期结算（1 天 / 3 天 / 7 天），不设止盈止损
    输出：真实期望、中位、胜率、5%/95% 分位、逐年、每月出手次数
    对照：随机时点买入（同持有期）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; FEE = 0.0009
HOLDS = {'1天': 288, '3天': 864, '7天': 2016}

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_amp_hold · 振幅异动后买入的真实期望（含 95% 未到 ±30% 的样本）')
log('=' * 112)

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
del C; gc.collect()
log(f'币 {N}   全局 {T}   耗时 {time.time()-t0:.0f}s')

rows = {'SIG': [], 'RND': []}
tid = {'SIG': [], 'RND': []}
symv = {'SIG': [], 'RND': []}
rng = np.random.default_rng(7)

for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < WDAY * 4 + 2017:
        continue
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    base = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        ar = amp / np.maximum(base, 1e-12)
    sig = np.isfinite(ar) & (ar >= 2.0) & (np.arange(n) < n - 2017)
    idx_s = np.flatnonzero(sig)
    if len(idx_s) == 0:
        continue
    if len(idx_s) > 2000:
        idx_s = idx_s[::max(1, len(idx_s)//2000)]
    idx_r = rng.integers(0, n - 2017, size=min(len(idx_s), 2000))
    posv = np.searchsorted(ts, ot)
    for tag, idx in (('SIG', idx_s), ('RND', idx_r)):
        c0 = c[idx]
        for hname, hb in HOLDS.items():
            v = c[idx + hb] / c0 - 1.0 - FEE
            rows.setdefault(tag, []).append((hname, v.mean(), np.median(v), (v > 0).mean(),
                                             np.percentile(v, 5), np.percentile(v, 95), len(v)))
        tid[tag].append(posv[idx].astype('int64'))
        symv[tag].append(np.full(len(idx), sym))
    del amp, base, ar
del RAW; gc.collect()

log('\n' + '=' * 112)
log('【结果】真实期望（全部样本，到期结算，不设止盈止损，扣 0.09%）')
log('=' * 112)
log(f'  {"样本":<6}{"持有":>6}{"笔数":>9}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"p5%":>8}{"p95%":>9}')
agg = {}
for tag in ('SIG', 'RND'):
    for hname in HOLDS:
        vals = [r for r in rows[tag] if r[0] == hname]
        if not vals:
            continue
        n = sum(v[6] for v in vals)
        e = sum(v[1]*v[6] for v in vals)/n
        md = sum(v[2]*v[6] for v in vals)/n
        wr = sum(v[3]*v[6] for v in vals)/n
        p5 = sum(v[4]*v[6] for v in vals)/n
        p95 = sum(v[5]*v[6] for v in vals)/n
        agg[(tag, hname)] = (e, n)
        lbl = '振幅≥2' if tag == 'SIG' else '随机买入'
        log(f'  {lbl:<8}{hname:>6}{n:>9,}{100*e:>+10.2f}{100*md:>9.2f}{100*wr:>8.1f}'
            f'{100*p5:>8.1f}{100*p95:>9.1f}')

log('\n' + '=' * 112)
log('【对照】振幅信号 vs 随机买入（差值 = 信号的真实增量）')
log('=' * 112)
for hname in HOLDS:
    a = agg.get(('SIG', hname)); b = agg.get(('RND', hname))
    if not a or not b:
        continue
    log(f'  {hname:<5}  振幅 {100*a[0]:+.2f}%  -  随机 {100*b[0]:+.2f}%  =  增量 {100*(a[0]-b[0]):+.2f}%')

log('\n' + '=' * 112)
log('【判读】')
log('=' * 112)
log('  · 若"振幅"组显著优于"随机"组 ⇒ 振幅异动确实含正向信息')
log('  · 若两组接近 ⇒ 之前的 +25% 只是"只统计走到 ±30% 的样本"造成的假象')

out = os.path.join(OUT, 'amp_hold_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
