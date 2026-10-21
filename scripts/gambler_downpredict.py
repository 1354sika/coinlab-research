# -*- coding: utf-8 -*-
"""gambler_downpredict · 「看跌能力」第一版：用 1m 全量特征预测「60 分钟内跌破 -1%」

穷尽可用的 K 线内信息：多尺度涨跌、量比、主动买占比、笔数密度、波动率、小时。
标签：未来 60 分钟最低价 <= 入场价 * (1 - 1%)
时间留出：前 30 天训练 / 后 30 天测试。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
THR = 0.01
STEP = 100          # 抽样步长

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_downpredict · 看跌能力：1m 全量特征 → 未来 60 分钟跌破 -1%')
log('=' * 100)

t_read = time.time()
XS, YS, TS = [], [], []
n_sym = 0
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
        # 未来 60 分钟最低价（不含当前）
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        y = (fmin <= cc * (1 - THR)).astype('int8')
        # 特征
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
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(200, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YS.append(y[idx])
        TS.append(ot[:m][idx])
    del ch; gc.collect()

X = np.vstack(XS); y = np.concatenate(YS); t = np.concatenate(TS)
del XS, YS, TS; gc.collect()
log(f'币 {n_sym}  样本 {len(X):,}  正例（60分钟内跌破-1%）{y.mean():.1%}  读盘 {time.time()-t_read:.0f}s')

cut = np.median(t)
tr = t <= cut
te = t > cut
log(f'训练 {tr.sum():,}（正例 {y[tr].mean():.1%}）  测试 {te.sum():,}（正例 {y[te].mean():.1%}）')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATS = ['r1', 'r5', 'r15', 'r60', 'volr', 'tbr', 'trr', 'vstd', 'hour']
mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=7, subsample=0.8)
t_fit = time.time()
mdl.fit(X[tr], y[tr])
log(f'训练耗时 {time.time()-t_fit:.0f}s')
p = mdl.predict_proba(X[te])[:, 1]
auc = roc_auc_score(y[te], p)
log(f'\n【测试段 AUC = {auc:.3f}】  （0.5=瞎猜，0.65+=可用，0.75+=很强）')

log('')
log(f'{"预测跌概率档":16s}{"样本":>10s}{"实际跌破率":>12s}{"相对基准":>12s}')
log('-' * 100)
base = y[te].mean()
for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (p >= lo) & (p < hi)
    if m.sum() == 0:
        continue
    log(f'[{lo:.1f},{hi:.1f})         {m.sum():>10,}{y[te][m].mean():>12.1%}{y[te][m].mean()/base:>11.2f}x')

log('\n【特征重要性】')
for f, v in sorted(zip(FEATS, mdl.feature_importances_), key=lambda x: -x[1]):
    log(f'  {f:6s} {v:.3f}')

log('\n【判读】')
log('  · 若最高档（P>=0.9）实际跌破率 > 80% ⇒ 可以拿它做排除法')
log('  · 若分档差异 < 10pp ⇒ K 线内信息不够，需要订单簿/逐笔')

out = os.path.join(OUT, 'downpredict_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
