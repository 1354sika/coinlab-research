# -*- coding: utf-8 -*-
"""方案 A2：目标从「预测未来最大浮盈」改为「预测未来还会不会再创新高」。
    分类器直接学你关心的那件事：出场之后还有没有更高点。

输入（每个时点）：最近 W=24 根原始 K 线（逐根收益 24 维 + 逐根量比 24 维）
                 + 当前浮盈 + 已持有根数
标签：未来路径里是否出现「比当前浮盈高出 COST 以上」的点（1/0）
决策：P(新高) < θ -> 走；θ 由训练段网格搜索最大化净收益决定（数据定，不是人拍）
进化：每季度用过去 4 个季度重训
检验：每笔净收益、季度为正比例、捕获率、错过比例/错过额
"""
import time
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_q2_REAL.txt'
COST = 0.0004
H = 96
W = 24
PER = 300
SEED = 17
COND = len(sys.argv) > 1 and sys.argv[1] == 'cond'

t0 = time.time()
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d load %.1fs' % (len(df), time.time() - t0), flush=True)

XS, YS, FS, FC, QE = [], [], [], [], []
for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float); v = g['volume'].to_numpy(float)
    tt = g['open_time'].to_numpy(); n = len(c)
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
    fc = c[idx[:, None] + jj] / p0 - 1
    pos = idx[:, None] + np.arange(1, H + 1)[None, :]
    WR = np.stack([ret[pos - w] for w in range(W, 0, -1)], axis=-1)
    WV = np.stack([volr[pos - w] for w in range(W, 0, -1)], axis=-1)
    G = fc[:, 1:1 + H]
    HOLD = np.tile(np.arange(1, H + 1, dtype=float), (len(idx), 1))
    X = np.concatenate([WR.reshape(-1, W), WV.reshape(-1, W),
                        G.reshape(-1, 1), HOLD.reshape(-1, 1)], axis=1)
    suf = np.maximum.accumulate(fc[:, ::-1], axis=1)[:, ::-1]
    Q = np.concatenate([suf[:, 2:], fc[:, -1:]], axis=1)      # (m,H) 未来最大浮盈
    Y = (Q > G + COST).astype(np.float32)
    XS.append(X.astype(np.float32)); YS.append(Y.ravel())
    FS.append(G.ravel().astype(np.float32))
    FC.append(fc); QE.append(tt[idx])

X = np.concatenate(XS); Y = np.concatenate(YS); FS = np.concatenate(FS)
FC = np.concatenate(FC); QE = np.concatenate(QE)
J = H
m0 = len(FC)
X3 = X.reshape(m0, J, X.shape[1]); Y3 = Y.reshape(m0, J)
bad = ~np.isfinite(X3).all(axis=(1, 2))
X, Y = X3[~bad].reshape(-1, X.shape[1]), Y3[~bad].reshape(-1)
FS = FS.reshape(m0, J)[~bad].reshape(-1)
FC, QE = FC[~bad], QE[~bad]
print('samples %d  paths %d  prep %.1fs' % (len(X), len(FC), time.time() - t0), flush=True)

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
    mdl = HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.08, random_state=0)
    mdl.fit(X[tr], Y[tr].astype(int))
    ptr = mdl.predict_proba(X[tr])[:, 1]
    pte = mdl.predict_proba(X[te_row])[:, 1]
    n_tr = int(tr.sum() // J); n_te = int(te_ent.sum())
    Ptr = ptr.reshape(n_tr, J); Pte = pte.reshape(n_te, J)
    base_tr = FC[np.isin(q_ent, trq)]
    Gtr = FS[tr].reshape(n_tr, J)
    # θ 由训练段决定：最大化每笔净收益
    best_theta, best_pnl = 0.5, -9.0
    for th in np.arange(0.10, 0.91, 0.05):
        take = Ptr < th
        has = take.any(axis=1)
        jex = np.where(has, take.argmax(axis=1), J - 1)
        pnl = base_tr[np.arange(n_tr), 1 + jex] - COST
        if has.mean() < 0.05:
            continue
        if pnl.mean() > best_pnl:
            best_pnl, best_theta = pnl.mean(), th
    take = Pte < best_theta
    has = take.any(axis=1)
    jex = np.where(has, take.argmax(axis=1), J - 1)
    base = FC[te_ent]
    pnl = base[np.arange(n_te), 1 + jex] - COST
    best = np.max(base, axis=1) - COST
    exit_px = base[np.arange(n_te), 1 + jex]
    after = np.full(n_te, -9.0)
    for j in range(1, H + 1):
        mm = j > (1 + jex)
        after[mm] = np.maximum(after[mm], base[mm, j])
    miss_rate = float((after > exit_px).mean())
    miss_amt = float((after - exit_px)[after > exit_px].mean()) if (after > exit_px).any() else 0.0
    capture = float(pnl.sum() / best.sum()) if best.sum() > 0 else 0.0
    res.append((teq, pnl.mean(), float(best_theta), miss_amt, miss_rate, capture, n_te))
    print('%s  net %.3f%%  theta %.2f  miss %.1f%%  n=%d' %
          (teq, pnl.mean() * 100, best_theta, miss_rate * 100, n_te), flush=True)

arr = np.array([r[1] for r in res])
lines = []
lines.append('方案A2 新高分类：%s入场，W=%d 原始窗口，H=%d，每季重训' %
             ('好入场(急跌+巨量)' if COND else '随机入场', W, H))
lines.append('样本 %d 路径 / %d 时点，OOS %d 季' % (len(FC), len(X), len(res)))
lines.append('')
lines.append('%-8s %9s %7s %9s %9s %8s %7s' % ('季度', '每笔净', 'theta', '错过额', '错过比例', '捕获率', '笔数'))
for qn, p, th, m_, mr, cap, n in res:
    lines.append('%-8s %8.3f%% %7.2f %8.3f%% %8.1f%% %7.1f%% %7d' % (qn, p * 100, th, m_ * 100, mr * 100, cap * 100, n))
lines.append('')
lines.append('季度为正比例 %.0f%% (%d/%d)' % ((arr > 0).mean() * 100, (arr > 0).sum(), len(arr)))
lines.append('最差单季 %+.3f%%   平均 %+.3f%%' % (arr.min() * 100, arr.mean() * 100))
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
