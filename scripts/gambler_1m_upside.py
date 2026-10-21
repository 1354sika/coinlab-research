# -*- coding: utf-8 -*-
"""gambler_1m_upside · 1m 粒度：未来 60 分钟「涨」的概率与幅度

回答三件事：
  1. 任意一分钟，未来 60 分钟涨 2%/5%/10% 的基础概率是多少
  2. 模型能把「要涨」的概率提到多少（AUC / 精确率）
  3. 模型报出来之后，实际能涨多少（涨幅分布）
口径：1m K 线，502 币，60 天；采样 STEP=20；时间留出（前 70% 训练 / 后 30% 测试）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
log('gambler_1m_upside · 1m 粒度：未来 60 分钟「涨」的概率与幅度')
log('=' * 100)

FS, FMX, FMN, TS, SS = [], [], [], [], []
sym_id = {}
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','quote_volume','volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','volume':'float32',
                             'trades':'float32','taker_base':'float32'},
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
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,volr,tbr,trr,vstd,vstd1440,hour])
        fmx = fmax / cc - 1.0
        fmn = fmin / cc - 1.0
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmx) & np.isfinite(fmn)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        FS.append(F[idx].astype('float32'))
        FMX.append(fmx[idx].astype('float32'))
        FMN.append(fmn[idx].astype('float32'))
        TS.append(ot[idx]); SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(FS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN)
t = np.concatenate(TS); s = np.concatenate(SS)
del FS, FMX, FMN, TS, SS; gc.collect()
log(f'样本 {len(X):,}  币 {len(sym_id)}')
log(f'时间范围 {pd.to_datetime(t.min(), unit="ms")} ~ {pd.to_datetime(t.max(), unit="ms")}')

log('\n【一】基础概率（不挑时机，任意一分钟）')
for th in [0.02, 0.05, 0.10]:
    log(f'  未来 60 分钟最高涨 >= {th:.0%} 的概率: {(fmx >= th).mean():.3%}')

cut = np.median(t); trm = t <= cut; tem = t > cut
log(f'\n训练段 {trm.sum():,}  测试段 {tem.sum():,}')

log('\n【二】模型：预测「未来 60 分钟涨 >= 5%」')
Y = (fmx >= 0.05).astype(np.int8)
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                   min_samples_leaf=50, random_state=7)
m.fit(X[trm], Y[trm])
p = m.predict_proba(X[tem])[:, 1]
auc = roc_auc_score(Y[tem], p)
log(f'  AUC = {auc:.4f}  基准概率 {Y[tem].mean():.3%}')

log('\n【三】按概率分档：报出时的真实命中率 + 涨幅分布')
qs = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
edges = np.quantile(p, qs)
log(f'  {"档":<16s}{"n":>8s}{"命中率(涨5%)":>14s}{"涨>=2%":>10s}{"涨>=10%":>10s}{"涨幅P50":>10s}{"涨幅P90":>10s}')
for i in range(len(qs)-1):
    lo, hi = edges[i], edges[i+1]
    sel = (p >= lo) & (p < hi) if i < len(qs)-2 else (p >= lo)
    nn = sel.sum()
    if nn == 0: continue
    log(f'  [{lo:.3f},{hi:.3f})  {nn:>8,}{Y[tem][sel].mean():>14.1%}'
        f'{(fmx[tem][sel]>=0.02).mean():>10.1%}{(fmx[tem][sel]>=0.10).mean():>10.1%}'
        f'{np.median(fmx[tem][sel]):>10.1%}{np.quantile(fmx[tem][sel],0.9):>10.1%}')

log('\n【四】按阈值报出（可交易视角）')
for th in [0.5, 0.7, 0.9]:
    sel = p >= th
    nn = sel.sum()
    if nn == 0: continue
    log(f'  P>={th}: 报出 {nn:>7,} ({sel.mean():.2%})  命中(涨5%) {Y[tem][sel].mean():>6.1%}  '
        f'涨>=2% {(fmx[tem][sel]>=0.02).mean():>6.1%}  涨>=10% {(fmx[tem][sel]>=0.10).mean():>6.1%}  '
        f'涨幅中位 {np.median(fmx[tem][sel]):>7.1%}')

log('\n【五】单特征 AUC 对照（HistGradientBoosting 无 feature_importances_）')
for i, nm in enumerate(['r1','r5','r15','r60','volr','tbr','trr','vstd','vstd1440','hour']):
    try:
        a = roc_auc_score(Y[tem], X[tem][:, i])
        log(f'  {nm:8s} 单特征AUC={a:.4f}' + ('  ← 反向（越涨越不涨）' if a < 0.5 else ''))
    except Exception as e:
        log(f'  {nm:8s} 计算失败 {e}')

log('\n【判读】')
log('  · 若 AUC < 0.60 ⇒ 1m 粒度下也预测不了「要涨」')
log('  · 若高档命中率仍 < 20% ⇒ 模型报出也不值得下注')
log('  · 「涨>=2%」那列是能做多的口径：TP 2%，赢率需 > 34.7%（maker）')

out = os.path.join(OUT, 'upside_1m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
