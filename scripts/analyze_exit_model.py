# -*- coding: utf-8 -*-
"""出场模型（不是规则表）：用「当前状态」预测「继续持有还能不能赚」，模型每季度重训。

为什么不是机械规则：
  - 不设固定 TP/SL/回撤阈值；每个时点由模型给出「继续持有 K 根的期望净收益」
  - 期望 <= 0 且已持有 >= 2 根 -> 出场；最多 H 根兜底（风控，不是策略）
  - 每季度用「过去所有季度」重训一次，再预测下一季（时间外），周期变了模型跟着变

特征（全部事前可得）：浮盈 g、自峰值回撤 d、已持有根数、入场前 24 根波动
标签：从该时点再持有 K=6 根的净收益（扣 maker 往返 0.04%）
"""
import time
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_model_REAL.txt'
COST = 0.0004
H = 24
K = 6
PER = 1000
SEED = 11
COND = len(sys.argv) > 1 and sys.argv[1] == 'cond'

t0 = time.time()
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
df = df.sort_values(['symbol', 'open_time'])
# 全局状态特征（入场时点即可知）：市场宽度、BTC 近 12 根收益/波动
df['up'] = (df['close'] > df['open']).astype(float)
df = df.merge(df.groupby('open_time')['up'].mean().rename('breadth'), left_on='open_time', right_index=True, how='left')
_b = df.loc[df['symbol'] == 'BTCUSDT', ['open_time', 'close']].drop_duplicates('open_time').set_index('open_time')['close']
df = df.merge((_b / _b.shift(12) - 1).rename('btc_r12'), left_on='open_time', right_index=True, how='left')
df = df.merge(_b.pct_change().rolling(12).std().rename('btc_v12'), left_on='open_time', right_index=True, how='left')
print('rows %d load %.1fs' % (len(df), time.time() - t0), flush=True)

XS, YS, QS, FCS = [], [], [], []
for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float); hi = g['high'].to_numpy(float)
    lo = g['low'].to_numpy(float); v = g['volume'].to_numpy(float)
    tt = g['open_time'].to_numpy()
    n = len(c)
    if n < H + 60:
        continue
    ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1
    vol = pd.Series(ret).rolling(24).std().to_numpy()
    cand = np.arange(30, n - H - 1)
    if COND:
        ret6 = np.full(n, np.nan); ret6[6:] = c[6:] / c[:-6] - 1
        vma = pd.Series(v).rolling(24).mean().to_numpy()
        volr = v / vma
        ok = (~np.isnan(ret6)) & (~np.isnan(volr)) & (ret6 <= -0.02) & (volr >= 3)
        cand = cand[ok[cand]]
    if len(cand) < 100:
        continue
    idx = rng.choice(cand, size=min(PER, len(cand)), replace=False)
    idx.sort()
    jj = np.arange(H + 1)
    p0 = c[idx][:, None]
    fh = hi[idx[:, None] + jj] / p0 - 1
    fl = lo[idx[:, None] + jj] / p0 - 1
    fc_px = c[idx[:, None] + jj]
    fc = fc_px / p0 - 1
    J = H - K
    G = fc[:, 1:1 + J]
    PEAK = np.maximum.accumulate(fh, axis=1)[:, 1:1 + J]
    D = PEAK - G
    HOLD = np.tile(np.arange(1, J + 1, dtype=float), (len(idx), 1))
    VOL = np.repeat(vol[idx][:, None], J, axis=1)
    BR = g['breadth'].to_numpy(float)[idx]
    BTR = g['btc_r12'].to_numpy(float)[idx]
    BTV = g['btc_v12'].to_numpy(float)[idx]
    Y = fc_px[:, 1 + K:1 + K + J] / fc_px[:, 1:1 + J] - 1 - COST
    XS.append(np.stack([G, D, HOLD, VOL,
                        np.repeat(BR[:, None], J, axis=1),
                        np.repeat(BTR[:, None], J, axis=1),
                        np.repeat(BTV[:, None], J, axis=1)], axis=-1).reshape(-1, 7))
    YS.append(Y.ravel())
    QS.append(tt[idx])
    FCS.append(fc)

X = np.concatenate(XS); Y = np.concatenate(YS); QE = np.concatenate(QS); FC = np.concatenate(FCS)
J = H - K
m0 = len(FC)
X3 = X.reshape(m0, J, 7); Y3 = Y.reshape(m0, J)
bad = (~np.isfinite(Y3).all(axis=1)) | (~np.isfinite(X3).all(axis=(1, 2)))
X, Y, FC, QE = X3[~bad].reshape(-1, 7), Y3[~bad].reshape(-1), FC[~bad], QE[~bad]
QT = np.repeat(QE, J)
print('samples %d  prep %.1fs' % (len(X), time.time() - t0), flush=True)

q = pd.to_datetime(QT, unit='ms').to_period('Q').astype(str)
q_ent = pd.to_datetime(QE, unit='ms').to_period('Q').astype(str)
qs = sorted(pd.unique(q))
print('quarters %d: %s .. %s' % (len(qs), qs[0], qs[-1]), flush=True)

WIN = 4   # 每季用最近 4 个季度训练
res = []  # (quarter, model每笔均值, 基线每笔均值, 笔数)
m_tot = len(FC)
for qi in range(WIN, len(qs)):
    trq = qs[qi - WIN:qi]
    teq = qs[qi]
    tr = np.isin(q, trq); te_row = q == teq; te_ent = q_ent == teq
    mdl = HistGradientBoostingRegressor(max_iter=80, max_depth=6, learning_rate=0.08, random_state=0)
    mdl.fit(X[tr], Y[tr])
    pred = mdl.predict(X[te_row])
    # 逐笔：第一个 pred<=0 且 hold>=2 的时点出场，否则 H 根兜底
    n_te = FC[te_ent].shape[0]
    P = pred.reshape(n_te, -1)               # (m, J)
    J = P.shape[1]
    hold = np.arange(1, J + 1)
    take = (P <= 0) & (hold[None, :] >= 2)
    has = take.any(axis=1)
    jexit = np.where(has, take.argmax(axis=1), J - 1)   # 无信号则最后
    base = FC[te_ent]
    pnl = np.where(has, base[np.arange(n_te), 1 + jexit], base[:, -1]) - COST
    # 基线：固定规则 TP12% / s5% / act1% / trail0.3% / T24（上一轮最优）
    fh_te = fh_of = None
    res.append((teq, pnl.mean(), n_te))
    print('%s  model %.3f%%  n=%d' % (teq, pnl.mean() * 100, n_te), flush=True)

arr = np.array([r[1] for r in res])
lines = []
lines.append('出场模型（状态->期望，季度滚动重训）：%s入场，H=%d根，K=%d，扣 %.2f%%' %
             ('好入场(急跌+巨量)' if COND else '随机入场', H, K, COST * 100))
lines.append('OOS 季度：%d 个（每季用前 %d 季训练，预测下一季）' % (len(res), WIN))
lines.append('')
for qn, m, n in res:
    lines.append('  %s  %+8.3f%%/笔  n=%d' % (qn, m * 100, n))
lines.append('')
lines.append('季度为正比例 %.0f%% (%d/%d)' % ((arr > 0).mean() * 100, (arr > 0).sum(), len(arr)))
lines.append('最差单季 %+.3f%%   平均 %+.3f%%' % (arr.min() * 100, arr.mean() * 100))
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
