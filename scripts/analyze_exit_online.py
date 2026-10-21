# -*- coding: utf-8 -*-
"""在线 Q-learning 出场策略：模型自己在交易中学习，人只给状态和奖励。

与前面几版的区别：
  - 没有任何人工阈值（没有 TP/SL/θ），决策 = argmax Q(状态, 走/留)
  - 模型持续进化：每笔交易结束后立刻更新 Q 表（不是每季离线重训）
  - 状态分箱边界由训练段数据的分位数决定，不是人拍数值

状态：浮盈档(8) × 自峰值回撤档(5) × 已持有档(6) × 波动档(3) = 720
动作：0=留，1=走
奖励：走 = 当前浮盈 − 成本；留 = 下一步浮盈增量（最终由 Q 传播）
检验：每季 每笔净 / 捕获率 / 错过比例 / 错过额
"""
import time
import numpy as np
import pandas as pd

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_online_REAL.txt'
COST = 0.0004
H = 24
PER = 600
SEED = 23
ALPHA = 0.15
GAMMA = 0.9
COND = False

t0 = time.time()
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d load %.1fs' % (len(df), time.time() - t0), flush=True)

paths = []   # (fc[H+1], vol[H], t0)
for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float); v = g['volume'].to_numpy(float)
    tt = g['open_time'].to_numpy(); n = len(c)
    if n < H + 40:
        continue
    ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1
    vol = pd.Series(ret).rolling(24).std().to_numpy()
    cand = np.arange(30, n - H - 1)
    idx = rng.choice(cand, size=min(PER, len(cand)), replace=False); idx.sort()
    jj = np.arange(H + 1)
    p0 = c[idx][:, None]
    fc = c[idx[:, None] + jj] / p0 - 1
    vv = np.repeat(vol[idx][:, None], H, axis=1)
    for a, b, c2, tv in zip(fc, vv, idx, tt[idx]):
        paths.append((a, b, tv))
print('paths %d prep %.1fs' % (len(paths), time.time() - t0), flush=True)
paths.sort(key=lambda x: x[2])   # 按时间排序（进化要用时序）

# 训练段（前 70%）定义分箱边界
cut = int(len(paths) * 0.7)
g_all = np.concatenate([p[0][1:] for p in paths[:cut]])
d_all = np.concatenate([np.maximum.accumulate(p[0])[1:] - p[0][1:] for p in paths[:cut]])
v_all = np.concatenate([p[1][:, -1] for p in paths[:cut]])
g_edges = np.quantile(g_all, np.linspace(0, 1, 9)[1:-1])
d_edges = np.quantile(d_all, np.linspace(0, 1, 6)[1:-1])
v_edges = np.quantile(v_all, np.linspace(0, 1, 4)[1:-1])


def state(g, peak, d, hold, vol):
    a = np.searchsorted(g_edges, g)
    b = np.searchsorted(d_edges, d)
    c = min(int(np.searchsorted([3, 6, 12, 18, 23], hold)), 5)
    e = np.searchsorted(v_edges, vol)
    return ((a * 5 + b) * 6 + c) * 3 + e


NS = 8 * 5 * 6 * 3
Q = np.zeros((NS, 2))
res = []
for pi, (fc, vv, tv) in enumerate(paths):
    peak = np.maximum.accumulate(fc)
    # 逐时点
    for j in range(1, H + 1):
        g = fc[j]; pk = peak[j]; d = pk - g; v0 = vv[j - 1, -1]
        s = state(g, pk, d, j, v0)
        if j == H:
            # 强制平仓
            r = g - COST
            Q[s, 1] += ALPHA * (r - Q[s, 1])
            if pi >= cut:
                res.append((tv, g - COST, -1, j))
            break
        s2 = state(fc[j + 1], peak[j + 1], peak[j + 1] - fc[j + 1], j + 1, v0)
        nxt = max(Q[s2, 0], Q[s2, 1])
        r_hold = (fc[j + 1] - g) + GAMMA * nxt
        Q[s, 0] += ALPHA * (r_hold - Q[s, 0])
        r_close = g - COST
        Q[s, 1] += ALPHA * (r_close - Q[s, 1])
        if pi >= cut and Q[s, 1] > Q[s, 0]:
            res.append((tv, g - COST, -1, j))
            break
    else:
        continue

dfr = pd.DataFrame(res, columns=['t', 'pnl', 'x', 'j'])
dfr['q'] = pd.to_datetime(dfr['t'], unit='ms').to_period('Q').astype(str)
agg = dfr.groupby('q')['pnl'].agg(['mean', 'count'])
lines = []
lines.append('在线 Q-learning 出场：%d 状态，α=%.2f γ=%.2f，人只给状态与奖励' % (NS, ALPHA, GAMMA))
lines.append('进化：每笔交易后立即更新 Q（前 70%% 时间只学不交易，后 30%% 边交易边学）')
lines.append('')
lines.append('%-8s %12s %8s' % ('季度', '每笔净', '笔数'))
for qn, row in agg.iterrows():
    lines.append('%-8s %11.3f%% %8d' % (qn, row['mean'] * 100, int(row['count'])))
arr = agg['mean'].values
lines.append('')
lines.append('季度为正 %.0f%% (%d/%d)，平均 %+.3f%%，最差 %+.3f%%' %
             ((arr > 0).mean() * 100, (arr > 0).sum(), len(arr), arr.mean() * 100, arr.min() * 100))
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
