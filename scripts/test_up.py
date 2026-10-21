# -*- coding: utf-8 -*-
"""四格对称检验：我们到底是在「预测跌」还是在「预测涨」？
信号原语（共用已验证的最硬门槛 volr>8）：
  A 连续段急跌 |streak_ret|>=5% 方向=跌
  B 连续段急涨 |streak_ret|>=5% 方向=涨
四个格：A做多 / A做空 / B做多 / B做空
输出每格的：开仓数、毛收益、净(maker/taker)、夏普、回撤、逐期正收益期数、方向命中率
目的：回答用户「不能只做预测跌的机器」「能不能反过来预测涨」。
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

HOLD, DQ, VOLR = 60, 0.90, 8.0
FEE_T, FEE_M = 0.0010, 0.0002
K = 6
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
QV = qv.values.astype(np.float64)
qv60 = qv.rolling(60).mean().values
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
FWD60 = np.full((T, N), np.nan)
FWD60[:-60] = P[60:] / P[:-60] - 1.0
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
edges = np.linspace(0, T, K + 1).astype(int)

SIG_DOWN = (streak_ret < -0.05) & (volr > VOLR) & sel   # 急跌
SIG_UP = (streak_ret > +0.05) & (volr > VOLR) & sel     # 急涨


def hit_rate(sig, sign):
    y = FWD60[sig]; y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan, 0
    return (np.sign(y) == np.sign(sign)).mean(), len(y)


say('\n=== 方向命中率（未去重样本）：P(未来60分钟方向 == 下注方向) ===')
for nm, sig, sign in (('急跌->做多', SIG_DOWN, +1), ('急跌->做空', SIG_DOWN, -1),
                      ('急涨->做多', SIG_UP, +1), ('急涨->做空', SIG_UP, -1)):
    h, n = hit_rate(sig, sign)
    say('  %-12s n=%6d  命中率=%.2f%%' % (nm, n, 100 * h))


def build_pos(sig, sign):
    pos = np.zeros((T, N), np.float32)
    n = 0
    for j in range(N):
        ii = np.flatnonzero(sig[:, j])
        if len(ii) == 0:
            continue
        last = -1
        for k in ii:
            if k <= last:
                continue
            pos[k:k + HOLD, j] = sign
            n += 1
            last = k + HOLD - 1
    return pos, n


def run(sig, sign, tag):
    pos, n = build_pos(sig, sign)
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (np.abs(pos[:-1]) > 0).sum(axis=1).astype(np.float64)
    eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    d = np.abs(np.diff(pos, axis=0, prepend=0))
    fs = (d > 1e-6).sum(axis=1) * FEE_M
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = eq - fse
    gross = eq
    nav = np.cumprod(1 + net)
    pk = np.maximum.accumulate(nav)
    sh = net.mean() / net.std() * np.sqrt(525600) if net.std() > 0 else np.nan
    posi = 0
    for p in range(K):
        a, b = edges[p], edges[p + 1]
        seg = net[a:min(b, T - 1)]
        if len(seg) and np.prod(1 + seg) - 1 > 0:
            posi += 1
    # taker 版
    fs2 = (d > 1e-6).sum(axis=1) * FEE_T
    fse2 = np.where(act > 0, fs2[:-1] / np.maximum(act, 1), 0.0)
    nav2 = np.cumprod(1 + (eq - fse2))
    say('  %-14s 开仓=%5d  毛=%+9.1f%%  净maker=%+9.1f%%  净taker=%+9.1f%%  夏普=%5.2f  回撤=%6.1f%%  正期=%d/6' % (
        tag, n, 100 * gross.sum(), 100 * (nav[-1] - 1), 100 * (nav2[-1] - 1),
        sh, 100 * ((nav - pk) / pk).min(), posi))


say('\n=== 四格组合回测（各自单独跑，等权，无前视）===')
run(SIG_DOWN, +1, '急跌→做多')
run(SIG_DOWN, -1, '急跌→做空')
run(SIG_UP, +1, '急涨→做多')
run(SIG_UP, -1, '急涨→做空')

say('\n=== 两个「顺势」方向单独看（预测涨/预测跌的正统含义）===')
say('  注：「顺势做多」=跌之后继续跌？不，顺势应为：跌→做空、涨→做多')
run(SIG_DOWN, -1, '顺势:跌→做空')
run(SIG_UP, +1, '顺势:涨→做多')
run(SIG_DOWN, +1, '逆势:跌→做多')
run(SIG_UP, -1, '逆势:涨→做空')

with open(os.path.join(EV, 'test_up.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
