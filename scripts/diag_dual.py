# -*- coding: utf-8 -*-
"""诊断：为什么事件级统计说「连续跌>5% -> 未来60分钟 +1.66%」，
而组合回测却亏到 -98%？逐层拆开看。
诊断项：
  D1 未去重信号样本的 FWD60 均值（= analyze_streak 的口径）
  D2 去重后（同币 60 分钟内只开一次）开仓时点的 FWD60 均值
  D3 按 streak_ret 深度分档（-5~-7 / -7~-10 / -10~-15 / <-15）分别看
      -> 验证假说：「首次触及 -5% 时最弱，跌得更深才强」
  D4 模拟「允许换仓」（新信号更强时替换旧仓）与「加仓」两种规则
  D5 用固定窗口 R15<-5% 做对照（排除连续性算法本身的问题）
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, THR = 60, 0.05

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
FWD60 = np.full((T, N), np.nan)
FWD60[:-60] = P[60:] / P[:-60] - 1.0
R15 = (px / px.shift(15) - 1.0).values
d90 = float(np.nanpercentile(disp_v, 90))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)

SIG = (streak_ret < -THR) & sel
say('\n[原始信号] 分钟x币 = %d' % SIG.sum())

# D1 未去重
y1 = FWD60[SIG]; y1 = y1[np.isfinite(y1)]
say('D1 未去重样本: n=%d  FWD60 均值=%+.4f%%  中位=%+.4f%%  P(涨)=%.2f%%' % (
    len(y1), 100*y1.mean(), 100*np.median(y1), 100*(y1 > 0).mean()))

# D2 去重（同币 60 分钟内只取首个）
opens = []
for j in range(N):
    ii = np.flatnonzero(SIG[:, j])
    if len(ii) == 0:
        continue
    last = -1
    for k in ii:
        if k <= last:
            continue
        if np.isfinite(FWD60[k, j]):
            opens.append((k, j))
        last = k + HOLD - 1
opens = np.array(opens, int)
y2 = FWD60[opens[:, 0], opens[:, 1]]; y2 = y2[np.isfinite(y2)]
say('D2 去重后开仓: n=%d  FWD60 均值=%+.4f%%  中位=%+.4f%%  P(涨)=%.2f%%' % (
    len(y2), 100*y2.mean(), 100*np.median(y2), 100*(y2 > 0).mean()))

# 未去重 vs 去重 差在哪：把去重时被跳过的样本单列
mask_open = np.zeros((T, N), bool)
for k, j in opens:
    mask_open[k, j] = True
skipped = SIG & ~mask_open
y3 = FWD60[skipped]; y3 = y3[np.isfinite(y3)]
say('D2b 被跳过的样本（持仓期内重复触发）: n=%d  FWD60 均值=%+.4f%%' % (
    len(y3), 100*y3.mean() if len(y3) else np.nan))

# D3 按深度分档
say('\nD3 按「连续段累计跌幅」分档（用去重前全样本）')
sr = streak_ret[SIG]
yy = FWD60[SIG]
ok = np.isfinite(sr) & np.isfinite(yy)
sr, yy = sr[ok], yy[ok]
for lo, hi in [(-0.07, -0.05), (-0.10, -0.07), (-0.15, -0.10), (-0.25, -0.15), (-1, -0.25)]:
    m = (sr >= lo) & (sr < hi)
    if m.sum() < 30:
        continue
    say('  [%.0f%% .. %.0f%%] n=%6d  FWD60 均值=%+.4f%%  P(涨)=%.1f%%' % (
        lo*100, hi*100, m.sum(), 100*yy[m].mean(), 100*(yy[m] > 0).mean()))

say('\nD3b 同一分档，但只看【去重后开仓】的样本')
sr_o = streak_ret[opens[:, 0], opens[:, 1]]
yy_o = FWD60[opens[:, 0], opens[:, 1]]
ok = np.isfinite(sr_o) & np.isfinite(yy_o)
sr_o, yy_o = sr_o[ok], yy_o[ok]
for lo, hi in [(-0.07, -0.05), (-0.10, -0.07), (-0.15, -0.10), (-0.25, -0.15), (-1, -0.25)]:
    m = (sr_o >= lo) & (sr_o < hi)
    if m.sum() < 20:
        continue
    say('  [%.0f%% .. %.0f%%] n=%6d  FWD60 均值=%+.4f%%  P(涨)=%.1f%%' % (
        lo*100, hi*100, m.sum(), 100*yy_o[m].mean(), 100*(yy_o[m] > 0).mean()))

# D4 换仓规则：持仓期内若出现更强信号（streak_ret 更低），则替换
pos = np.zeros((T, N), np.float32)
cur = np.full(N, -1, int)      # 当前持仓的入场行
curdepth = np.zeros(N)
for k in range(T):
    act = np.flatnonzero(SIG[k])
    for j in act:
        if cur[j] >= 0 and k - cur[j] >= HOLD:
            cur[j] = -1
        if cur[j] < 0 or streak_ret[k, j] < curdepth[j]:
            cur[j] = k
            curdepth[j] = streak_ret[k, j]
            pos[k:min(k + HOLD, T), j] = 1
    for j in np.flatnonzero(cur >= 0):
        if k - cur[j] >= HOLD:
            cur[j] = -1
say('\nD4 换仓规则（更强信号替换）: 持仓分钟=%d' % int((pos > 0).sum()))
act = (pos > 0).sum(axis=1).astype(float)
R_next = np.vstack([R[1:], np.full((1, N), np.nan)])
pnl = np.nansum(pos[:-1] * R_next[:-1], axis=1)
eq = np.where(act[:-1] > 0, pnl / np.maximum(act[:-1], 1), 0.0)
say('  等权毛收益合计=%+.2f%%   净值=%+.2f%%' % (100*eq.sum(), 100*(np.cumprod(1 + eq)[-1] - 1)))

# D5 固定窗口对照
SIG15 = (R15 < -0.05) & sel
y5 = FWD60[SIG15]; y5 = y5[np.isfinite(y5)]
say('\nD5 对照 固定窗口 R15<-5%% & 分化: n=%d  FWD60 均值=%+.4f%%  P(涨)=%.2f%%' % (
    len(y5), 100*y5.mean(), 100*(y5 > 0).mean()))

with open(os.path.join(EV, 'diag_dual.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
