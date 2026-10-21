# -*- coding: utf-8 -*-
"""gambler_evolve_slip_tp · 对已验证组合做两件事
② 滑点测试：TP=2%/SL=1% 下，滑点 0/0.1/0.2/0.3/0.5% 各剩多少
① 止盈调大：TP∈{2,3,5,8,10}% × SL∈{1,1.5,2.5,4,5}%，看赢率/净期望

组合：r15>+2% & volr<0.5 & vstd>p90 & tbr>0.55 做空
口径：60 分钟窗口；双触保守算止损；滑点按双边（开+平）计
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 50; FEE = 0.0004

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_evolve_slip_tp · ②滑点 + ①止盈调大')
log('=' * 100)

FS, FMX, FMN, TS, SS = [], [], [], [], []
sym_id = {}
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
        r240, r1440 = ret(240), ret(1440)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour])
        fmax_rel = fmax / cc - 1.0
        fmin_rel = fmin / cc - 1.0
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmax_rel) & np.isfinite(fmin_rel)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        FS.append(F[idx].astype('float32'))
        FMX.append(fmax_rel[idx].astype('float32'))
        FMN.append(fmin_rel[idx].astype('float32'))
        TS.append(ot[idx]); SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(FS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN)
t = np.concatenate(TS); s = np.concatenate(SS)
del FS, FMX, FMN, TS, SS; gc.collect()
r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour = [X[:,i] for i in range(11)]
log(f'样本 {len(X):,}  币 {len(sym_id)}')

cut = np.median(t); trm = t <= cut; tem = t > cut
P90 = float(np.nanpercentile(vstd[trm], 90))
COMBO = (r15 > 0.02) & (volr < 0.5) & (vstd > P90) & (tbr > 0.55)

def eval_tp(tp, sl, slip, mask):
    # 做空：止盈=价格跌 tp（赚 tp），止损=价格涨 sl（亏 sl）；双触保守算止损
    w = (fmn <= -tp) & (fmx < sl) & mask
    l = (fmx >= sl) & mask
    n = w.sum() + l.sum()
    if n == 0: return 0, 0.0, 0.0
    wr = w.sum() / n
    net = (w.sum()*tp - l.sum()*sl) / n - FEE - 2*slip
    return n, wr, net

log('\n' + '=' * 100)
log('【A】测试段组合 + 原口径 TP=2%/SL=1% 的滑点敏感性（滑点=双边总计）')
log('=' * 100)
m = tem & COMBO
n0, wr0, _ = eval_tp(0.02, 0.01, 0.0, m)
log(f'  组合样本 n={n0}  赢率 {wr0:.1%}')
log(f'  {"双边滑点":>10s}{"净期望":>12s}')
for slip in [0.0, 0.001, 0.002, 0.003, 0.005, 0.01]:
    _, _, net = eval_tp(0.02, 0.01, slip/2, m)   # 单边=slip/2 → 双边=slip
    log(f'  {slip:>10.1%}{net:>+12.3%}')

log('\n' + '=' * 100)
log('【B】止盈调大：TP × SL 网格（测试段组合，滑点 0）')
log('=' * 100)
log(f'  {"TP":>6s}{"SL":>6s}{"n":>8s}{"赢率":>8s}{"净期望":>10s}')
for tp in [0.02, 0.03, 0.05, 0.08, 0.10]:
    for sl in [0.01, 0.015, 0.025, 0.04, 0.05]:
        n, wr, net = eval_tp(tp, sl, 0.0, m)
        if n == 0: continue
        log(f'  {tp:>6.1%}{sl:>6.1%}{n:>8}{wr:>8.1%}{net:>+10.3%}')

log('\n' + '=' * 100)
log('【C】止盈调大 + 滑点 0.3%（双边）的组合')
log('=' * 100)
log(f'  {"TP":>6s}{"SL":>6s}{"n":>8s}{"赢率":>8s}{"净期望":>10s}')
for tp in [0.02, 0.03, 0.05, 0.08, 0.10]:
    for sl in [0.01, 0.015, 0.025, 0.04, 0.05]:
        n, wr, net = eval_tp(tp, sl, 0.0015, m)
        if n == 0: continue
        if net > -0.01:
            log(f'  {tp:>6.1%}{sl:>6.1%}{n:>8}{wr:>8.1%}{net:>+10.3%}')

log('\n' + '=' * 100)
log('【D】判读')
log('=' * 100)
log('  · 滑点 0.3% 双边后仍 >0 的格 = 抗滑点可行')
log('  · TP 调大后赢率下降、单笔变大；找「滑点后仍显著为正」的格')

out = os.path.join(OUT, 'slip_tp_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
