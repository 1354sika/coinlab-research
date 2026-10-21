# -*- coding: utf-8 -*-
"""方案 A：状态价值学习（不是规则表，也不是我挑的语义特征）

人只定两件事：给什么原始输入、奖励怎么算。其余（看什么、什么时候走、阈值多少）全由模型学。

输入（每个时点）：最近 W=12 根原始 K 线的
    - 逐根收益（12 维）
    - 逐根量比（12 维）
    - 当前浮盈 g（1 维）
标签：从该时点之后到路径末的最大浮盈（Q_hold，离线 value target）
决策：模型预测的 Q_hold <= 当前浮盈 -> 走（比较由价值决定，人不设阈值）
进化：每季度用过去 4 个季度重训（周期变了模型跟着变）

事后检验（你第 3 条）：每笔的「事后最优浮盈」为基线，算
    - miss：走早了多少（实际出场价 < 事后最优，且之后还有更高点）
    - 给出 regret = 最优 - 实际
"""
import time
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_q_REAL.txt'
COST = 0.0004
H = 24
W = 12
PER = 800
SEED = 13
COND = len(sys.argv) > 1 and sys.argv[1] == 'cond'

t0 = time.time()
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d load %.1fs' % (len(df), time.time() - t0), flush=True)

XS, YS, GS, FC, QE = [], [], [], [], []
for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float); hi = g['high'].to_numpy(float)
    v = g['volume'].to_numpy(float); tt = g['open_time'].to_numpy()
    n = len(c)
    if n < H + W + 10:
        continue
    ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1
    vma = pd.Series(v).rolling(24).mean().to_numpy()
    volr = v / vma
    cand = np.arange(W + 2, n - H - 1)
    if COND:
        ret6 = np.full(n, np.nan); ret6[6:] = c[6:] / c[:-6] - 1
        ok = (~np.isnan(ret6)) & (~np.isnan(volr)) & (ret6 <= -0.02) & (volr >= 3)
        cand = cand[ok[cand]]
    if len(cand) < 100:
        continue
    idx = rng.choice(cand, size=min(PER, len(cand)), replace=False)
    idx.sort()
    jj = np.arange(H + 1)
    p0 = c[idx][:, None]
    fc = c[idx[:, None] + jj] / p0 - 1                  # (m,H+1) 相对入场浮盈
    pos = idx[:, None] + np.arange(1, H + 1)[None, :]    # (m,H) 每个候选出场时点
    WR = np.stack([ret[pos - w] for w in range(W, 0, -1)], axis=-1)   # (m,H,W)
    WV = np.stack([volr[pos - w] for w in range(W, 0, -1)], axis=-1)
    G = fc[:, 1:1 + H]                                   # (m,H) 当前浮盈
    suf = np.maximum.accumulate(fc[:, ::-1], axis=1)[:, ::-1]
    Q = np.concatenate([suf[:, 2:], fc[:, -1:]], axis=1)  # (m,H) 从 j 之后的最大浮盈
    X = np.concatenate([WR.reshape(-1, W), WV.reshape(-1, W), G.reshape(-1, 1)], axis=1)
    XS.append(X.astype(np.float32))
    YS.append(Q.ravel().astype(np.float32))
    GS.append(G.ravel().astype(np.float32))
    FC.append(fc)
    QE.append(tt[idx])

X = np.concatenate(XS); Y = np.concatenate(YS); GV = np.concatenate(GS)
FC = np.concatenate(FC); QE = np.concatenate(QE)
print('samples %d  paths %d  prep %.1fs' % (len(X), len(FC), time.time() - t0), flush=True)

# 过滤坏值（按笔整体过滤）
J = H
m0 = len(FC)
X3 = X.reshape(m0, J, X.shape[1]); Y3 = Y.reshape(m0, J); G3 = GV.reshape(m0, J)
bad = (~np.isfinite(X3).all(axis=(1, 2))) | (~np.isfinite(Y3).all(axis=1))
X, Y, GV = X3[~bad].reshape(-1, X.shape[1]), Y3[~bad].reshape(-1), G3[~bad].reshape(-1)
FC, QE = FC[~bad], QE[~bad]

QT = np.repeat(QE, J)
q = pd.to_datetime(QT, unit='ms').to_period('Q').astype(str)
q_ent = pd.to_datetime(QE, unit='ms').to_period('Q').astype(str)
qs = sorted(pd.unique(q_ent))
print('quarters %d: %s .. %s' % (len(qs), qs[0], qs[-1]), flush=True)

WIN = 4
res = []
for qi in range(WIN, len(qs)):
    trq = qs[qi - WIN:qi]
    teq = qs[qi]
    tr = np.isin(q, trq); te_row = q == teq; te_ent = q_ent == teq
    mdl = HistGradientBoostingRegressor(max_iter=100, max_depth=6, learning_rate=0.07, random_state=0)
    mdl.fit(X[tr], Y[tr])
    pred = mdl.predict(X[te_row])
    n_te = int(te_ent.sum())
    P = pred.reshape(n_te, J)
    take = P <= GV[te_row].reshape(n_te, J)          # 预测的继续价值 <= 当前浮盈 -> 走
    hold = np.arange(1, J + 1)
    take &= hold[None, :] >= 1
    has = take.any(axis=1)
    jex = np.where(has, take.argmax(axis=1), J - 1)
    base = FC[te_ent]
    pnl = base[np.arange(n_te), jex] - COST
    best = np.max(base, axis=1) - COST
    regret = (best - pnl).mean()
    exit_px = base[np.arange(n_te), jex]
    after = np.full(n_te, -9.0)
    for j in range(1, H + 1):
        mm = j > jex
        after[mm] = np.maximum(after[mm], base[mm, j])
    miss_rate = float((after > exit_px).mean())
    miss_amt = float((after - exit_px)[after > exit_px].mean()) if (after > exit_px).any() else 0.0
    capture = float(pnl.sum() / best.sum()) if best.sum() > 0 else 0.0
    res.append((teq, pnl.mean(), regret, miss_amt, miss_rate, capture, n_te))
    print('%s  model %.3f%%  regret %.3f%%  n=%d' % (teq, pnl.mean() * 100, regret * 100, n_te), flush=True)

arr = np.array([r[1] for r in res])
lines = []
lines.append('方案A 状态价值学习：%s入场，原始窗口 W=%d + 浮盈，H=%d 根，每季重训' %
             ('好入场(急跌+巨量)' if COND else '随机入场', W, H))
lines.append('样本 %d 条路径 / %d 个时点，OOS 季度 %d 个（前 %d 季训练）' % (len(FC), len(X), len(res), WIN))
lines.append('')
lines.append('%-8s %10s %10s %10s %9s %8s %7s' % ('季度', '每笔净', 'regret', '错过额', '错过比例', '捕获率', '笔数'))
for qn, p, r, m_, mr, cap, n in res:
    lines.append('%-8s %9.3f%% %9.3f%% %9.3f%% %8.1f%% %7.1f%% %7d' % (qn, p * 100, r * 100, m_ * 100, mr * 100, cap * 100, n))
lines.append('')
lines.append('季度为正比例 %.0f%% (%d/%d)' % ((arr > 0).mean() * 100, (arr > 0).sum(), len(arr)))
lines.append('最差单季 %+.3f%%   平均 %+.3f%%' % (arr.min() * 100, arr.mean() * 100))
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
