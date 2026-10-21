# -*- coding: utf-8 -*-
"""防过拟合：三个门槛的单调性扫描（V3 = disp>P90 & |streak_ret|>=5% & volr>5 & 段长>=3）
判据：若收益随门槛提高而【单调变化】，说明不是从网格里碰巧挑出来的点；
      若只有某个特定值最优、两侧都掉回去，则是过拟合的强烈信号。
同时给出各档的样本数与逐期正收益期数。
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

HOLD, DQ, FEE_M = 60, 0.90, 0.0002
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
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
absr = np.abs(streak_ret)
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
edges = np.linspace(0, T, K + 1).astype(int)


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


def run(sigL, sigS):
    posL, nL = build_pos(sigL, +1)
    posS, nS = build_pos(sigS, -1)
    pos = posL + posS
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (np.abs(pos[:-1]) > 0).sum(axis=1).astype(np.float64)
    eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    d = np.abs(np.diff(pos, axis=0, prepend=0))
    fs = (d > 1e-6).sum(axis=1) * FEE_M
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = eq - fse
    nav = np.cumprod(1 + net)
    pk = np.maximum.accumulate(nav)
    sh = net.mean() / net.std() * np.sqrt(525600) if net.std() > 0 else np.nan
    posi = 0
    for p in range(K):
        a, b = edges[p], edges[p + 1]
        seg = net[a:min(b, T - 1)]
        if len(seg) and np.prod(1 + seg) - 1 > 0:
            posi += 1
    return dict(nL=nL, nS=nS, tot=nav[-1] - 1, sharpe=sh,
                mdd=((nav - pk) / pk).min(), posi=posi)


base = dict(v=5.0, l=3, d=0.05)

say('\n=== 扫描 A：量比阈值 volr（其余固定：段长>=3, 深度>=5%）===')
say('  %-8s %7s %7s %11s %8s %9s %8s' % ('volr>', 'n多', 'n空', '净maker%', '夏普', '回撤%', '正期/6'))
for v in (2, 3, 5, 8, 10, 15, 20):
    SL = (streak_ret < -base['d']) & (volr > v) & (seglen >= base['l']) & sel
    SS = (streak_ret > +base['d']) & (volr > v) & (seglen >= base['l']) & sel
    r = run(SL, SS)
    say('  %-8s %7d %7d %10.1f%% %8.2f %8.1f%% %8s' % (
        v, r['nL'], r['nS'], 100 * r['tot'], r['sharpe'], 100 * r['mdd'], '%d' % r['posi']))

say('\n=== 扫描 B：段长阈值 seglen（其余固定：volr>5, 深度>=5%）===')
say('  %-8s %7s %7s %11s %8s %9s %8s' % ('段长>=', 'n多', 'n空', '净maker%', '夏普', '回撤%', '正期/6'))
for l in (1, 2, 3, 5, 8, 12, 20):
    SL = (streak_ret < -base['d']) & (volr > base['v']) & (seglen >= l) & sel
    SS = (streak_ret > +base['d']) & (volr > base['v']) & (seglen >= l) & sel
    r = run(SL, SS)
    say('  %-8s %7d %7d %10.1f%% %8.2f %8.1f%% %8s' % (
        l, r['nL'], r['nS'], 100 * r['tot'], r['sharpe'], 100 * r['mdd'], '%d' % r['posi']))

say('\n=== 扫描 C：深度阈值 |streak_ret|（其余固定：volr>5, 段长>=3）===')
say('  %-8s %7s %7s %11s %8s %9s %8s' % ('深度>=', 'n多', 'n空', '净maker%', '夏普', '回撤%', '正期/6'))
for d in (0.03, 0.04, 0.05, 0.07, 0.10, 0.15):
    SL = (streak_ret < -d) & (volr > base['v']) & (seglen >= base['l']) & sel
    SS = (streak_ret > +d) & (volr > base['v']) & (seglen >= base['l']) & sel
    r = run(SL, SS)
    say('  %-8s %7d %7d %10.1f%% %8.2f %8.1f%% %8s' % (
        '%.0f%%' % (d * 100), r['nL'], r['nS'], 100 * r['tot'], r['sharpe'],
        100 * r['mdd'], '%d' % r['posi']))

say('\n=== 二维：volr x 段长（净maker%，深度固定 5%）===')
vs, ls = (3, 5, 10, 20), (1, 3, 5, 10)
say('  %-8s' % 'volr\\段长' + ''.join('%16s' % ('>=%d' % l) for l in ls))
for v in vs:
    row = '  %-8s' % ('>%d' % v)
    for l in ls:
        SL = (streak_ret < -base['d']) & (volr > v) & (seglen >= l) & sel
        SS = (streak_ret > +base['d']) & (volr > v) & (seglen >= l) & sel
        r = run(SL, SS)
        row += '%16s' % ('%+.0f%% (n%d)' % (100 * r['tot'], r['nL'] + r['nS']))
    say(row)

with open(os.path.join(EV, 'mono_test.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
