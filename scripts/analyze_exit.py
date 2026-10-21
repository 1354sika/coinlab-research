# -*- coding: utf-8 -*-
"""出场研究：浮盈 g 之后继续持有 k 根的期望净收益（扣 maker 往返成本）。
回答的问题：涨到多少之后，再拿着就不划算了。
口径：
  - 数据 5m，60 币，3 年；每 6 根取一个入场点（避免同段重叠）
  - 浮盈 g = 当前 close / 入场 close - 1（在入场后 0..24 根内取）
  - 未来收益 = 再过 k 根的 close 变化，扣 COST（maker 双边 0.04%）
  - 分档统计：期望、样本数、胜率；并给时间四分段的同向性
用法：python -X utf8 analyze_exit.py
"""
import time
import numpy as np
import pandas as pd

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_study_REAL.txt'
COST = 0.0004          # maker 单边 0.02% × 2
K = 24                 # 前向最多 24 根（2 小时）
KS = (1, 3, 6, 12, 24)
BINS = [-1.0, -0.03, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.0]
LAB = ['<-3%', '-3~-1%', '-1~-0.5%', '-0.5~0', '0~0.5%', '0.5~1%', '1~2%', '2~3%',
       '3~5%', '5~8%', '8~12%', '12~20%', '>20%']

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  load %.1fs' % (len(df), time.time() - t0), flush=True)

nb = len(LAB)
S = np.zeros((nb, len(KS))); N = np.zeros_like(S); W = np.zeros_like(S)
Q = 4
qS = np.zeros((Q, nb, len(KS))); qN = np.zeros_like(qS)
# 时间四分段的边界（按第一个币的时间轴切）
tmin, tmax = df['open_time'].min(), df['open_time'].max()
edges = np.linspace(tmin, tmax, Q + 1)

for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float)
    tt = g['open_time'].to_numpy()
    n = len(c)
    if n < K + 20:
        continue
    idx = np.arange(0, n - K - 1, 6)
    if len(idx) < 200:
        continue
    p0 = c[idx][:, None]
    fut = np.stack([c[idx + j] for j in range(K + 1)], axis=1)   # (m,K+1)
    gmat = fut / p0 - 1
    qb = np.clip(np.searchsorted(edges, tt[idx], side='right') - 1, 0, Q - 1)
    for ki, k in enumerate(KS):
        gp = gmat[:, :K + 1 - k]
        fwd = fut[:, k:] / fut[:, :K + 1 - k] - 1 - COST
        b = np.digitize(gp.ravel(), BINS) - 1
        ok = (b >= 0) & (b < nb)
        f = fwd.ravel()[ok]; bb = b[ok]
        np.add.at(S[:, ki], bb, f)
        np.add.at(N[:, ki], bb, 1.0)
        np.add.at(W[:, ki], bb, (f > 0).astype(float))
        # 分季度（时间四分位）
        qexp = np.repeat(qb, gp.shape[1])[ok]
        flat = qexp * nb + bb
        np.add.at(qS[:, :, ki].reshape(-1), flat, f)
        np.add.at(qN[:, :, ki].reshape(-1), flat, 1.0)

lines = []
lines.append('出场研究：浮盈 g 之后继续持有 k 根的期望净收益（已扣 %.2f%% 往返）' % (COST * 100))
lines.append('样本：5m / 60 币 / 每 6 根一个入场点 / 前向最多 %d 根' % K)
lines.append('')
head = '%-10s' % '浮盈档' + ''.join(['%14s' % ('再持%d根' % k) for k in KS]) + '%12s' % '样本数'
lines.append(head)
lines.append('-' * len(head))
for bi in range(nb):
    if N[bi, 0] < 500:
        continue
    cells = []
    for ki in range(len(KS)):
        n = N[bi, ki]
        cells.append('%13.3f%%' % (S[bi, ki] / n * 100) if n else '%14s' % '-')
    lines.append('%-10s' % LAB[bi] + ''.join(cells) + '%12d' % int(N[bi, 0]))
lines.append('')
lines.append('【读法】每格是该档浮盈下、再持有 k 根的净期望（%），负值=再拿就亏。')
lines.append('')
lines.append('=== 时间四分段同向性（再持 6 根） ===')
lines.append('%-10s' % '浮盈档' + ''.join(['%12s' % ('段%d' % (i + 1)) for i in range(Q)]))
for bi in range(nb):
    if N[bi, 2] < 500:
        continue
    cells = []
    for qi in range(Q):
        n = qN[qi, bi, 2]
        cells.append('%11.3f%%' % (qS[qi, bi, 2] / n * 100) if n > 100 else '%12s' % '-')
    lines.append('%-10s' % LAB[bi] + ''.join(cells))
lines.append('')
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
