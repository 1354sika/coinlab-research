# -*- coding: utf-8 -*-
"""诊断：为什么「有信号·跌型」样本均值 +3.50%，而组合回测却是 -24.7%？
两者都用 365 天、同期、同一信号定义，必须解释清楚，否则结论不可信。
逐层拆：
  L1 样本简单平均（未去重，分钟x币）
  L2 事件级去重后，每笔只看「入场到 60 分钟后」的 close-to-close 收益 —— 简单平均
  L3 同上，但按「时间等权」聚合（每笔占 1/同时持仓数）—— 模拟组合的资金约束
  L4 组合口径：逐分钟 pos*R 累加（复利净值）
另按持仓期长度扫描（5/15/30/60/120 分钟），看是不是「持有期口径」造成的差异
另：按币、按月看 L1 的分布（是否被少数币/少数月主导）
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

DQ, VOLR, DEP, FEE_M = 0.90, 10.0, 0.05, 0.0002
t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
QV = qv.values.astype(np.float64)
qv60 = qv.rolling(60).mean().values
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
t = px.index.values
cols = list(px.columns)
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
SIG = (streak_ret < -DEP) & (volr > VOLR) & sel
say('信号（分钟x币，未去重）= %d' % int(SIG.sum()))

for H in (5, 15, 30, 60, 120, 240):
    FWD = np.full((T, N), np.nan)
    FWD[:-H] = P[H:] / P[:-H] - 1.0
    y = FWD[SIG]; y = y[np.isfinite(y)]
    if len(y) < 30:
        continue
    # L2 事件级去重
    opens = []
    for j in range(N):
        ii = np.flatnonzero(SIG[:, j])
        if len(ii) == 0:
            continue
        last = -1
        for k in ii:
            if k <= last:
                continue
            if np.isfinite(FWD[k, j]):
                opens.append((k, j))
            last = k + H - 1
    yo = np.array([FWD[k, j] for k, j in opens])
    yo = yo[np.isfinite(yo)]
    # L4 组合（时间等权 + 复利）
    pos = np.zeros((T, N), np.float32)
    for k, j in opens:
        pos[k:k + H, j] = 1.0
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (pos[:-1] > 0).sum(axis=1).astype(float)
    eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    d = np.abs(np.diff(pos, axis=0, prepend=0))
    fs = (d > 1e-6).sum(axis=1) * FEE_M
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = eq - fse
    nav = np.cumprod(1 + net)
    say('\n--- 持有 %d 分钟 ---' % H)
    say('  L1 样本简单平均(未去重)  n=%5d  均值=%+.4f%%' % (len(y), 100 * y.mean()))
    say('  L2 事件级去重·简单平均    n=%5d  均值=%+.4f%%  中位=%+.4f%%  净(扣maker)=%+.4f%%' % (
        len(yo), 100 * yo.mean(), 100 * np.median(yo), 100 * (yo.mean() - FEE_M)))
    say('  L4 组合·时间等权·复利     净值=%+.2f%%  (max同时持仓=%d)' % (
        100 * (nav[-1] - 1), int(act.max())))

say('\n=== 按币看（持有 60 分钟，L1 未去重均值）—— 是否被少数币主导 ===')
H = 60
FWD = np.full((T, N), np.nan)
FWD[:-H] = P[H:] / P[:-H] - 1.0
rowsout = []
for j in range(N):
    v = FWD[:, j][SIG[:, j]]
    v = v[np.isfinite(v)]
    if len(v) >= 3:
        rowsout.append((cols[j], len(v), 100 * v.mean()))
rowsout.sort(key=lambda x: x[2])
say('  最低 5 个币: ' + ', '.join('%s n=%d %+.2f%%' % r for r in rowsout[:5]))
say('  最高 5 个币: ' + ', '.join('%s n=%d %+.2f%%' % r for r in rowsout[-5:]))

say('\n=== 按事件月份看（持有 60 分钟，L2 去重后）===')
opens = []
for j in range(N):
    ii = np.flatnonzero(SIG[:, j])
    if len(ii) == 0:
        continue
    last = -1
    for k in ii:
        if k <= last:
            continue
        if np.isfinite(FWD[k, j]):
            opens.append((k, j))
        last = k + H - 1
dfe = pd.DataFrame([{'t': t[k], 'sym': cols[j], 'y': FWD[k, j]} for k, j in opens])
dfe['mo'] = pd.to_datetime(dfe.t, unit='ms', utc=True).dt.strftime('%Y-%m')
g = dfe.groupby('mo').y.agg(['count', 'mean'])
say('  %-9s %6s %11s' % ('月份', 'n', '均值%'))
for m, r in g.iterrows():
    say('  %-9s %6d %10.4f%%' % (m, r['count'], 100 * r['mean']))
say('  → 正收益月 %d/%d' % ((g['mean'] > 0).sum(), len(g)))

with open(os.path.join(EV, 'diag_gap.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
