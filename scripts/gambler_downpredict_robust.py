# -*- coding: utf-8 -*-
"""gambler_downpredict_robust · 看跌能力的稳健性验证 + 回答两个具体问题

1) 时间留出再切两半（测试段后 30 天 → 前 15 天 / 后 15 天）：
   P>=0.9 档的跌破率在两半是否一致（不是一次运气）
2) P>=0.9 样本落在多少个币上（泛化性，别只看 3 个币）
3) 每天机会数：采样口径 + 按 STEP 折算全量
4) 排除 P>=0.9 后，"60 分内涨 >= 2%" 的比例 vs 整体 —— 回答"剩下的是涨的吗"
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
log('gambler_downpredict_robust · 看跌能力稳健性：时间两半 / 币分布 / 每天机会数 / 排除后是涨吗')
log('=' * 100)

t_read = time.time()
XS, YD, YU, TS, SS = [], [], [], [], []
n_sym = 0
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
        n = len(c)
        m = n - HOLD
        if m <= 200:
            continue
        n_sym += 1
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        yd = (fmin <= cc * (1 - THR_DN)).astype('int8')
        yu = (fmax >= cc * (1 + THR_UP)).astype('int8')
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
        YD.append(yd[idx]); YU.append(yu[idx])
        TS.append(ot[:m][idx])
        SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(XS); yd = np.concatenate(YD); yu = np.concatenate(YU)
t = np.concatenate(TS); s = np.concatenate(SS)
del XS, YD, YU, TS, SS; gc.collect()
log(f'币 {n_sym}  样本 {len(X):,}  跌标签正例 {yd.mean():.1%}  涨2%标签正例 {yu.mean():.1%}  读盘 {time.time()-t_read:.0f}s')
log(f'时间跨度 {pd.to_datetime(t.min(), unit="ms")} ~ {pd.to_datetime(t.max(), unit="ms")}')

cut = np.median(t)
tr = t <= cut
te = t > cut

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=7, subsample=0.8)
t_fit = time.time()
mdl.fit(X[tr], yd[tr])
log(f'训练耗时 {time.time()-t_fit:.0f}s')

p = mdl.predict_proba(X[te])[:, 1]
auc = roc_auc_score(yd[te], p)
log(f'\n【测试段 AUC = {auc:.3f}】')

# ---- 1) 测试段再切两半 ----
te_t = t[te]
med2 = np.median(te_t)
h1 = te_t <= med2
h2 = te_t > med2
log('\n【一】时间两半稳健性（测试段再切前后两半）')
log(f'{"半段":8s}{"样本":>10s}{"P>=0.9 样本":>12s}{"实际跌破率":>12s}{"AUC":>8s}')
log('-' * 100)
for nm, hm in [('前半', h1), ('后半', h2)]:
    pm = p[hm]; ym = yd[te][hm]
    hi = pm >= 0.9
    a = roc_auc_score(ym, pm) if len(np.unique(ym)) > 1 else float('nan')
    log(f'{nm:8s}{hm.sum():>10,}{hi.sum():>12,}{ym[hi].mean() if hi.sum() else float("nan"):>12.1%}{a:>8.3f}')

# ---- 2) P>=0.9 样本的币分布 ----
hi_all = p >= 0.9
sym_hi = s[te][hi_all]
n_coins_hi = len(np.unique(sym_hi))
cnt = pd.Series(sym_hi).value_counts()
log(f'\n【二】P>=0.9 样本的币分布（泛化性）')
log(f'  P>=0.9 样本 {hi_all.sum():,} 笔，落在 {n_coins_hi} 个币上')
log(f'  单币最多 {cnt.iloc[0]} 笔，占 {cnt.iloc[0]/hi_all.sum():.1%}；前 5 币占 {cnt.iloc[:5].sum()/hi_all.sum():.1%}')

# ---- 3) 每天机会数 ----
days = (t.max() - t.min()) / 86400000.0
log(f'\n【三】每天机会数（采样口径 → 全量折算，STEP={STEP}）')
log(f'  时间跨度 {days:.0f} 天')
log(f'  采样 P>=0.9：{hi_all.sum():,} 笔 / {days:.0f} 天 = {hi_all.sum()/days:.2f} 次/天（采样）')
log(f'  全量折算 ≈ ×{STEP} → {hi_all.sum()*STEP/days:,.0f} 次/天（币·分钟）')
log(f'  采样总样本：{len(X):,} / {days:.0f} 天 = {len(X)/days:,.0f} 次/天（采样）→ 全量 {len(X)*STEP/days:,.0f}')

# ---- 4) 排除后是涨吗 ----
log(f'\n【四】排除 P>=0.9 后，剩余样本「60 分内涨 >= 2%」的比例')
log(f'{"口径":26s}{"样本":>10s}{"涨>=2% 比例":>14s}')
log('-' * 100)
y_all = yu[te]
base_up = y_all.mean()
log(f'{"整体（不排除）":26s}{len(y_all):>10,}{base_up:>14.1%}')
for thr_hi in [0.5, 0.7, 0.8, 0.9]:
    keep = p < thr_hi
    log(f'{f"排除 P>={thr_hi:.1f}":26s}{keep.sum():>10,}{y_all[keep].mean():>14.1%}')
log('  （若排除后只升 <1pp ⇒ 剩下的不是「就是涨的」，排掉的和留下的是同一批高波动时刻）')

log('\n【判读】')
log('  · 时间两半若同号且接近 ⇒ 看跌能力稳定，不是单段运气')
log('  · 排除后涨率几乎不变 ⇒ 排除法对"找涨"无效（与 downshort 的做多止盈率结论一致）')

out = os.path.join(OUT, 'downpredict_robust_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
