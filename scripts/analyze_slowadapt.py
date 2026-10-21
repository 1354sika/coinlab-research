# -*- coding: utf-8 -*-
"""进化实验：无前视的阈值策略对比（回答「市场在变，机制能适应/进化吗」）
策略：
  S0 固定（首期校准）：用期1的 disp P90，之后永不改
  S1 快自适应：用上一期的 disp P90
  S2 慢自适应：用过去 30 天（3 期）的 disp P90
  S3 事后最优（上界）：用当期自己的 P90（有前视，仅作上界）
信号：分化 + 急跌5% + 放量2倍 -> 做多，持 60 分钟，不设止损；全部事件级。
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)
FEE_M = 0.0004
K = 6

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float32',
                        'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

r = px.pct_change()
disp_v = r.std(axis=1).values
R15 = (px / px.shift(15) - 1.0).values
VR = (qv / qv.rolling(60).mean()).values
FWD = (px.shift(-60) / px - 1.0).values
C = px.values
t = px.index.values
edges = np.linspace(t[0], t[-1], K + 1)

p90 = [float(np.nanpercentile(disp_v[(t >= edges[p]) & (t < edges[p + 1]) & np.isfinite(disp_v)], 90))
       for p in range(K)]
say('各期 disp P90: ' + ' '.join('%.4f%%' % (x * 100) for x in p90))


def events(a, b, thr):
    mrow = (t >= a) & (t < b)
    cond = (R15 < -0.05) & (VR > 2.0) & mrow[:, None] & (disp_v[:, None] > thr)
    out = []
    for j in range(cond.shape[1]):
        ii = np.flatnonzero(cond[:, j])
        if len(ii) == 0:
            continue
        for s in np.r_[0, np.flatnonzero(np.diff(ii) > 5) + 1]:
            out.append((ii[s], j))
    return out


def perf(ev):
    y = np.array([FWD[k, j] for k, j in ev]); y = y[np.isfinite(y)]
    if len(y) < 20:
        return (len(y), np.nan, np.nan, np.nan)
    return (len(y), 100 * y.mean(), 100 * (y.mean() - FEE_M), 100 * (y > 0).mean())


say('\n=== 各策略逐期表现（持有60分钟，净收益已扣 maker 0.04%）===')
say('  %-22s %6s %7s %10s %10s %8s' % ('策略', '期', 'n', '毛%', '净maker%', '胜率%'))
rows = []
for p in range(K):
    a, b = edges[p], edges[p + 1]
    strategies = {'S0 固定(首期校准)': p90[0],
                  'S1 快自适应(上期)': p90[p - 1] if p >= 1 else None,
                  'S2 慢自适应(过去30天)': (float(np.nanpercentile(
                      disp_v[(t >= edges[max(0, p - 3)]) & (t < edges[p]) & np.isfinite(disp_v)], 90))
                      if p >= 3 else None),
                  'S3 事后最优(当期,前视)': p90[p]}
    for nm, thr in strategies.items():
        if thr is None:
            continue
        n, g, nt, wr = perf(events(a, b, thr))
        say('  %-22s %6d %7d %9.3f %10.3f %8.1f' % (nm, p + 1, n, g, nt, wr))
        rows.append(dict(strategy=nm, period=p + 1, n=n, gross=g, net=nt, win=wr, thr=thr))
    say('')

R = pd.DataFrame(rows)
say('\n=== 汇总：各策略在可用期数上的平均净收益 ===')
for nm, d in R.groupby('strategy'):
    say('  %-22s 期数=%d  平均净=%+.3f%%  正收益期=%d/%d  事件合计=%d' % (
        nm, len(d), d.net.mean(), (d.net > 0).sum(), len(d), d.n.sum()))

say('\n=== 关键对比（仅取三策略都有的期：4/5/6）===')
common = R[R.period >= 4]
say('  %-22s 平均净%%   合计事件' % '策略')
for nm, d in common.groupby('strategy'):
    say('  %-22s %+.3f    %d' % (nm, d.net.mean(), d.n.sum()))

fig, ax = plt.subplots(figsize=(11, 5.5))
for nm, d in R.groupby('strategy'):
    ax.plot(d.period, d.net, 'o-', label=nm)
ax.axhline(0, color='gray', lw=.8)
ax.set_xlabel('period'); ax.set_ylabel('net return after maker fee (%)')
ax.set_title('Fixed vs fast-adaptive vs slow-adaptive threshold (no look-ahead)')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
p_ = os.path.join(EV, 'slowadapt.png')
plt.savefig(p_, dpi=125)
say('\nsaved: %s' % p_)
R.to_csv(os.path.join(EV, 'slowadapt.csv'), index=False, encoding='utf-8')

with open(os.path.join(EV, 'slowadapt.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
