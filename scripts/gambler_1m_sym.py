# -*- coding: utf-8 -*-
"""gambler_1m_sym · 1m 粒度：高波动时刻，涨和跌对称吗？

upside_1m 发现：vstd 单特征 AUC=0.8646（远高于 r1/r5/r15），
说明模型抓的是「波动会变大」，不是「方向会向上」。
本脚本补这一刀：按 vstd 分档，对照 涨≥2% / 跌≥2% / 涨≥5% / 跌≥5% 的概率，
以及「先涨后跌回去」的比例（摸到 +2% 但随后跌破 -1%）。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 20

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_1m_sym · 高波动时刻：涨和跌对称吗？')
log('=' * 100)

VS, FMX, FMN, TS = [], [], [], []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        r = np.full(m, np.nan); r[1:] = cc[1:] / cc[:-1] - 1
        vstd = pd.Series(r).rolling(15, min_periods=5).std().values
        fmx = fmax / cc - 1.0
        fmn = fmin / cc - 1.0
        ok = np.isfinite(vstd) & np.isfinite(fmx) & np.isfinite(fmn)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        VS.append(vstd[idx].astype('float32'))
        FMX.append(fmx[idx].astype('float32'))
        FMN.append(fmn[idx].astype('float32'))
        TS.append(ot[idx])
    del ch; gc.collect()

v = np.concatenate(VS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN); t = np.concatenate(TS)
log(f'样本 {len(v):,}')

cut = np.median(t); trm = t <= cut; tem = t > cut
qs = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
edges = np.quantile(v[trm], qs)

log('\n【一】按 vstd（15 分钟波动率）分档，涨跌对称性（测试段）')
log(f'  {"vstd档":<26s}{"n":>9s}{"涨>=2%":>9s}{"跌>=2%":>9s}{"涨>=5%":>9s}{"跌>=5%":>9s}{"摸到+2%又跌回-1%":>18s}')
for i in range(len(qs)-1):
    lo, hi = edges[i], edges[i+1]
    sel = (v[tem] >= lo) & (v[tem] < hi) if i < len(qs)-2 else (v[tem] >= lo)
    nn = sel.sum()
    if nn == 0: continue
    fx = fmx[tem][sel]; fn = fmn[tem][sel]
    both = ((fx >= 0.02) & (fn <= -0.01)).mean()
    log(f'  [{lo:.5f},{hi:.5f})  {nn:>9,}{(fx>=0.02).mean():>9.1%}{(fn<=-0.02).mean():>9.1%}'
        f'{(fx>=0.05).mean():>9.1%}{(fn<=-0.05).mean():>9.1%}{both:>18.1%}')

log('\n【二】最高档（vstd>=p99）对比：涨>=2% vs 跌>=2%')
sel = v[tem] >= edges[-2]
fx = fmx[tem][sel]; fn = fmn[tem][sel]
log(f'  n={sel.sum():,}')
log(f'  涨>=2% 概率 {(fx>=0.02).mean():.1%}   跌>=2% 概率 {(fn<=-0.02).mean():.1%}')
log(f'  涨>=5% 概率 {(fx>=0.05).mean():.1%}   跌>=5% 概率 {(fn<=-0.05).mean():.1%}')
log(f'  摸到+2% 后 60 分钟内又跌破 -1% 的比例 {( (fx>=0.02)&(fn<=-0.01) ).mean():.1%}')
log(f'  涨幅中位 {np.median(fx):.2%}   跌幅中位 {np.median(fn):.2%}')

log('\n【判读】')
log('  · 若 跌>=2% 概率 ≈ 涨>=2% 概率 ⇒ 模型抓的是「波动」不是「方向」')
log('  · 若 摸到+2%又跌回-1% 占摸到+2% 的一半以上 ⇒ 止盈 2% 会被回落打掉')

out = os.path.join(OUT, 'sym_1m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
