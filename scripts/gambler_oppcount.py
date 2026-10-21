# -*- coding: utf-8 -*-
"""gambler_oppcount · 精确数「每天几次」：把同一币 60 分钟内的连续触发合并成 1 个独立机会

回答两个问题：
  Q1 每天出现的机会中，用 P>=0.9 排除后还剩几次？
  Q2 剩下的有效信息就是涨的吗？（用「涨>=2%」事件口径，与 downpredict_robust 一致）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
THR_DN = 0.01
THR_UP = 0.02
STEP = 100

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_oppcount · 独立机会数（同币 60 分钟内合并）+ 排除后剩几次 + 剩的是涨吗')
log('=' * 100)

XS, YD, YU, TS, SS = [], [], [], [], []
sym_id = {}
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'volume',
                               'quote_volume', 'trades', 'taker_base'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'volume': 'float32',
                             'quote_volume': 'float32', 'trades': 'float32', 'taker_base': 'float32'},
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
        if m <= 200:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        r1 = np.full(m, np.nan); r1[1:] = cc[1:] / cc[:-1] - 1
        r5 = np.full(m, np.nan); r5[5:] = cc[5:] / cc[:-5] - 1
        r15 = np.full(m, np.nan); r15[15:] = cc[15:] / cc[:-15] - 1
        r60 = np.full(m, np.nan); r60[60:] = cc[60:] / cc[:-60] - 1
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1, r5, r15, r60, volr, tbr, trr, vstd, hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin) & np.isfinite(fmax)
        idx = np.arange(200, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YD.append((fmin[idx] <= cc[idx] * (1 - THR_DN)).astype('int8'))
        YU.append((fmax[idx] >= cc[idx] * (1 + THR_UP)).astype('int8'))
        TS.append(ot[idx])
        SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(XS); yd = np.concatenate(YD); yu = np.concatenate(YU)
t = np.concatenate(TS); s = np.concatenate(SS)
del XS, YD, YU, TS, SS; gc.collect()
log(f'币 {len(sym_id)}  样本 {len(X):,}  读盘 {time.time()-t0:.0f}s')

from sklearn.ensemble import GradientBoostingClassifier
cut = np.median(t)
tr = t <= cut; te = t > cut
mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=7, subsample=0.8)
mdl.fit(X[tr], yd[tr])
p = mdl.predict_proba(X[te])[:, 1]
pt = t[te]; ps = s[te]; pyu = yu[te]

days = (pt.max() - pt.min()) / 86400000.0
log(f'测试段 {days:.0f} 天  样本 {len(pt):,}')

def episodes(mask):
    """同币、时间间隔 <= HOLD 分钟 的连续触发合并为 1 个 episode"""
    tt = pt[mask]; ss = ps[mask]
    order = np.lexsort((tt, ss))
    tt, ss = tt[order], ss[order]
    if len(tt) == 0:
        return 0, []
    eps = 1
    cur_s, cur_t = ss[0], tt[0]
    lens = []
    cur_len = 1
    for i in range(1, len(tt)):
        if ss[i] == cur_s and (tt[i] - cur_t) <= HOLD * 60000:
            cur_t = tt[i]; cur_len += 1
        else:
            lens.append(cur_len)
            eps += 1; cur_s, cur_t, cur_len = ss[i], tt[i], 1
    lens.append(cur_len)
    return eps, lens

hi = p >= 0.9
ep_hi, lens_hi = episodes(hi)
log(f'\n【Q1】独立机会数（同币 60 分钟内合并，STEP={STEP} 采样）')
log(f'  P>=0.9 触发点 {hi.sum():,} 个 → 合并成 {ep_hi:,} 个独立机会（平均每次连续 {hi.sum()/max(ep_hi,1):.1f} 个触发点）')
log(f'  采样口径：{ep_hi/days:.2f} 次/天 → 全量折算 ×{STEP} ≈ {ep_hi*STEP/days:,.0f} 次/天（币·独立机会）')

# 全部触发点也合并，作为分母
ep_all, _ = episodes(np.ones(len(pt), bool))
log(f'  全部触发点 {len(pt):,} → 合并成 {ep_all:,} 个独立机会 → 全量 {ep_all*STEP/days:,.0f} 次/天')
log(f'  排除 P>=0.9 后剩：{(ep_all-ep_hi):,} 个独立机会（{(ep_all-ep_hi)/max(ep_all,1):.1%}）→ 全量 {(ep_all-ep_hi)*STEP/days:,.0f} 次/天')

log(f'\n【Q2】剩下的有效信息就是涨的吗？（事件口径：未来 60 分钟最高价 >= +2%）')
log(f'{"口径":30s}{"样本":>10s}{"涨>=2% 比例":>14s}')
log('-' * 100)
log(f'{"整体（全部触发点）":30s}{len(pyu):>10,}{pyu.mean():>14.1%}')
for thr_hi in [0.5, 0.7, 0.8, 0.9]:
    keep = p < thr_hi
    log(f'{f"排除 P>={thr_hi:.1f} 后":30s}{keep.sum():>10,}{pyu[keep].mean():>14.1%}')
log('  （若排除后涨率变化 < 1pp ⇒ 剩下的不是「就是涨的」）')

log(f'\n【附】P>=0.9 那批自己的涨率：{pyu[hi].mean():.1%}（对照整体 {pyu.mean():.1%}）')
log(f'  即：模型说「要跌」的时刻，60 分内涨 2% 的比例 = {pyu[hi].mean():.1%}，并不比整体低多少')
log(f'  这解释了为什么排除法无效——同一批高波动时刻，跌和涨都会发生')

out = os.path.join(OUT, 'oppcount_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
