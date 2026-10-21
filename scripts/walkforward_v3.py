# -*- coding: utf-8 -*-
"""V3 的逐期稳定性 —— 回答「+282% 是不是从网格里挑出来的一次性结果」
V3 = disp>P90 & |streak_ret|>=5% & volr>5 & 段长>=3（做多；做空镜像）
持 60 分钟，无前视；把 60 天切成 6 期各自算净值、收益、回撤。
另测：把 V3 的三个门槛逐个去掉，看收益怎么变（判断哪个门槛是真贡献）。
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

HOLD, THR, DQ = 60, 0.05, 0.90
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
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)


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


def run(SL, SS, tag, lo=0, hi=T):
    posL, nL = build_pos(SL, +1)
    posS, nS = build_pos(SS, -1)
    pos = (posL + posS)[lo:hi]
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])[lo:hi]
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (np.abs(pos[:-1]) > 0).sum(axis=1).astype(np.float64)
    pnl_eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    d = np.abs(np.diff(pos, axis=0, prepend=0))
    fs = (d > 1e-6).sum(axis=1) * FEE_M
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = pnl_eq - fse
    nav = np.cumprod(1 + net)
    pk = np.maximum.accumulate(nav)
    return dict(tag=tag, nL=nL, nS=nS, tot=nav[-1] - 1,
                sharpe=net.mean() / net.std() * np.sqrt(525600) if net.std() > 0 else np.nan,
                mdd=((nav - pk) / pk).min(), win=(net > 0).mean())


SL3 = (streak_ret < -THR) & (volr > 5) & (seglen >= 3) & sel
SS3 = (streak_ret > +THR) & (volr > 5) & (seglen >= 3) & sel

say('\n=== V3 逐期（60 天切 6 期，maker 0.02%/边，无前视）===')
edges = np.linspace(0, T, K + 1).astype(int)
per = []
for p in range(K):
    r = run(SL3, SS3, 'p%d' % (p + 1), edges[p], edges[p + 1])
    d0 = pd.to_datetime(t[edges[p]], unit='ms', utc=True).strftime('%m-%d')
    d1 = pd.to_datetime(t[min(edges[p + 1], T - 1)], unit='ms', utc=True).strftime('%m-%d')
    say('  期%d %s~%s  开仓 多%d/空%d  净=%+.2f%%  夏普=%.2f  回撤=%.1f%%' % (
        p + 1, d0, d1, r['nL'], r['nS'], 100 * r['tot'], r['sharpe'], 100 * r['mdd']))
    per.append(r['tot'])
per = np.array(per)
say('  → 正收益期 %d/%d   中位 %+.2f%%   均值 %+.2f%%' % (
    (per > 0).sum(), K, 100 * np.median(per), 100 * per.mean()))

say('\n=== 门槛消融（全期，maker）===')
variants = [
    ('V3 全部门槛 volr>5 & 段>=3', (streak_ret < -THR) & (volr > 5) & (seglen >= 3) & sel,
     (streak_ret > +THR) & (volr > 5) & (seglen >= 3) & sel),
    ('去掉 volr（段>=3）', (streak_ret < -THR) & (seglen >= 3) & sel,
     (streak_ret > +THR) & (seglen >= 3) & sel),
    ('去掉 段长（volr>5）', (streak_ret < -THR) & (volr > 5) & sel,
     (streak_ret > +THR) & (volr > 5) & sel),
    ('只留 disp+幅度', (streak_ret < -THR) & sel, (streak_ret > +THR) & sel),
    ('volr>10 & 段>=3', (streak_ret < -THR) & (volr > 10) & (seglen >= 3) & sel,
     (streak_ret > +THR) & (volr > 10) & (seglen >= 3) & sel),
    ('volr>5 & 段>=3 & 深度>=8%', (streak_ret < -0.08) & (volr > 5) & (seglen >= 3) & sel,
     (streak_ret > +0.08) & (volr > 5) & (seglen >= 3) & sel),
]
for tag, SL, SS in variants:
    r = run(SL, SS, tag)
    say('  %-32s 多%d/空%d  净=%+.1f%%  夏普=%.2f  回撤=%.1f%%' % (
        tag, r['nL'], r['nS'], 100 * r['tot'], r['sharpe'], 100 * r['mdd']))

with open(os.path.join(EV, 'walkforward_v3.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
