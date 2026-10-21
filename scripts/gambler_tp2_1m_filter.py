# -*- coding: utf-8 -*-
"""gambler_tp2_1m_filter · 检验用户的「排除法」：能否预测「这次跌完还会继续跌」

样本：15 分钟内已跌 ≥5% 的入场点（此前测出这组止盈率最高，43.5%）
问题：在这些点里，能不能用可观测特征区分「先反弹到 +2%」和「继续跌到 -1%」？
    能区分 ⇒ 排除法成立，把「会继续跌的」剔掉，止盈率还能提高
    不能区分 ⇒ 不是机制不够好，是这份数据里没有这个信息
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
TP, SL, HOLD = 0.02, 0.01, 60

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_tp2_1m_filter · 排除法检验：跌完会不会继续跌')
log('=' * 100)

XS, YS, TS = [], [], []
t_read = time.time()
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64',
                             'close': 'float32', 'high': 'float32', 'low': 'float32',
                             'quote_volume': 'float32'},
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
        n = len(c); m = n - HOLD
        if m <= 60:
            continue
        cc = c[:m]
        INF = 10 ** 9
        f_tp = np.full(m, INF, dtype=np.int64)
        f_sl = np.full(m, INF, dtype=np.int64)
        for j in range(1, HOLD + 1):
            hh = h[j:j + m]; ll = l[j:j + m]
            nt = (f_tp == INF) & (hh >= cc * (1 + TP))
            ns = (f_sl == INF) & (ll <= cc * (1 - SL))
            f_tp[nt] = j
            f_sl[ns] = j
        r1 = np.full(m, np.nan); r1[1:] = cc[1:] / cc[:-1] - 1
        r5 = np.full(m, np.nan); r5[5:] = cc[5:] / cc[:-5] - 1
        r15 = np.full(m, np.nan); r15[15:] = cc[15:] / cc[:-15] - 1
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        sel = (r15 <= -0.05) & np.isfinite(r1) & np.isfinite(r5) & np.isfinite(volr) & np.isfinite(r15)
        if not sel.any():
            continue
        F = np.column_stack([volr[sel], r1[sel], r5[sel], r15[sel], hour[sel]])
        y = (f_tp[sel] < f_sl[sel]).astype('int8')
        XS.append(F); YS.append(y); TS.append(ot[:m][sel])
    del ch; gc.collect()
X = np.vstack(XS); y = np.concatenate(YS); t = np.concatenate(TS)
log(f'样本 {len(X):,}  正例（先触+2%）{y.mean():.1%}  扫描 {time.time()-t_read:.0f}s')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cut = np.median(t)
tr = t <= cut
te = t > cut
log(f'训练 {tr.sum():,}（正例 {y[tr].mean():.1%}）  测试 {te.sum():,}（正例 {y[te].mean():.1%}）')

for name, mdl in [('逻辑回归', LogisticRegression(max_iter=1000)),
                  ('GBDT', GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=7))]:
    mdl.fit(X[tr], y[tr])
    p = mdl.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    log(f'\n【{name}】测试段 AUC = {auc:.3f}')
    for lo, hi in [(0, .30), (.30, .40), (.40, .50), (.50, .60), (.60, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        log(f'  P∈[{lo:.2f},{hi:.2f})  n={m.sum():>6,}  实际先触+2% {y[te][m].mean():>5.1%}')

log('\n【判读】')
log('  · AUC > 0.60 ⇒ 排除法可行：预测「会继续跌」的剔掉，剩下的止盈率更高')
log('  · AUC 0.50~0.55 ⇒ 这份 1m K 线数据里没有区分信息，需要订单簿/逐笔')
log('  · 特征：量比、1m 跌幅、5m 涨幅、15m 涨幅、UTC 小时')

out = os.path.join(OUT, 'tp2_1m_filter_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
