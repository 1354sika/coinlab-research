# -*- coding: utf-8 -*-
"""验证爆发点信号（先验证，不做算子）
(A) 参数网格：跌幅阈值 x 量比阈值 x 分化档 -> 事件级 毛/净/胜率/n
(B) 持有期扫描：1/3/5/10/15/30/60 分钟
(C) 做空侧对称性：R15>+thr & 量比>vr & 分化 -> 做空
(D) 逐周/逐日稳定性
全部事件级：同一 symbol 间隔<=5 分钟的触发合并为一次，取首次触发分钟。
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

HOLDS = [1, 3, 5, 10, 15, 30, 60]
FEE_T, FEE_M = 0.0020, 0.0004

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time()-t0))

r = px.pct_change()
disp = r.std(axis=1)
disp_v = disp.values
R15 = (px / px.shift(15) - 1.0).values
VR = (qv / qv.rolling(60).mean()).values
FWD = {k: (px.shift(-k) / px - 1.0).values for k in HOLDS}
cols = list(px.columns); times = px.index.values

# 宽条件做事件合并（R15 < -3% 或 > +3%）
WIDE = 0.03
for side in ('down', 'up'):
    say('\n' + '=' * 70)
    say('【%s 侧】宽条件 %s R15 %s %.0f%%' % (
        '做多(急跌)' if side == 'down' else '做空(急涨)',
        '', '<' if side == 'down' else '>', WIDE * 100))
    cond = (R15 < -WIDE) if side == 'down' else (R15 > WIDE)
    rows = []
    for j in range(cond.shape[1]):
        ii = np.flatnonzero(cond[:, j])
        if len(ii) == 0:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(ii) > 5) + 1]
        for s in starts:
            k = ii[s]
            d = {'t': times[k], 'sym': cols[j], 'r15': R15[k, j],
                 'vr': VR[k, j], 'disp': disp_v[k]}
            for h in HOLDS:
                d['f%d' % h] = FWD[h][k, j]
            rows.append(d)
    ev = pd.DataFrame(rows)
    ev = ev[np.isfinite(ev.r15) & np.isfinite(ev.vr)]
    say('事件数 %d' % len(ev))
    sign = -1.0 if side == 'up' else 1.0     # 做空取反
    for h in HOLDS:
        ev['y%d' % h] = sign * ev['f%d' % h]

    dq = np.nanquantile(disp_v[np.isfinite(disp_v)], [0.85, 0.90, 0.95, 0.99])

    say('\n(A) 参数网格（持有5分钟，事件级）  跌幅阈 x 量比阈 x 分化档')
    say('  %-6s %-6s %-8s %8s %10s %10s %10s %8s' % (
        'R15阈', '量比阈', '分化档', 'n', '毛均值%', '净taker%', '净maker%', '胜率%'))
    for rthr in (0.03, 0.04, 0.05, 0.07, 0.10):
        for vthr in (1.5, 2.0, 3.0, 5.0):
            for dname, dmin in [('P85', dq[0]), ('P90', dq[1]), ('P95', dq[2]), ('P99', dq[3])]:
                m = ((ev.r15 < -rthr) if side == 'down' else (ev.r15 > rthr)) & \
                    (ev.vr > vthr) & (ev.disp > dmin)
                d = ev[m]
                if len(d) < 80:
                    continue
                y = d.y5.values; y = y[np.isfinite(y)]
                if len(y) < 80:
                    continue
                say('  %-6.0f%% %-6.1f %-8s %8d %9.3f %10.3f %10.3f %8.1f' % (
                    rthr*100, vthr, dname, len(y), 100*y.mean(),
                    100*(y.mean()-FEE_T), 100*(y.mean()-FEE_M), 100*(y > 0).mean()))

    # 最优组合
    best = None
    for rthr in (0.03, 0.04, 0.05, 0.07, 0.10):
        for vthr in (1.5, 2.0, 3.0, 5.0):
            for dmin in (dq[0], dq[1], dq[2], dq[3]):
                m = ((ev.r15 < -rthr) if side == 'down' else (ev.r15 > rthr)) & \
                    (ev.vr > vthr) & (ev.disp > dmin)
                d = ev[m]; y = d.y5.values; y = y[np.isfinite(y)]
                if len(y) < 150:
                    continue
                net = y.mean() - FEE_M
                if best is None or net > best[0]:
                    best = (net, rthr, vthr, dmin, len(y), y.mean(), y.mean()-FEE_T)
    if best:
        say('\n  最优（按 maker 净收益，n>=150）: 跌幅阈%.0f%% 量比%.1f 分化>%.4f%%  '
            'n=%d 毛=%.3f%% 净taker=%.3f%% 净maker=%.3f%%' % (
                best[1]*100, best[2], best[3]*100, best[4],
                100*best[5], 100*best[6], 100*best[0]))
        brthr, bvthr, bdmin = best[1], best[2], best[3]
    else:
        brthr, bvthr, bdmin = 0.05, 2.0, dq[1]

    mbest = ((ev.r15 < -brthr) if side == 'down' else (ev.r15 > brthr)) & \
            (ev.vr > bvthr) & (ev.disp > bdmin)
    db = ev[mbest]

    say('\n(B) 持有期扫描（用最优条件，事件级 n=%d）' % len(db))
    say('  %-6s %10s %10s %10s %8s' % ('持有', '毛均值%', '净taker%', '净maker%', '胜率%'))
    for h in HOLDS:
        y = db['y%d' % h].values; y = y[np.isfinite(y)]
        if len(y) < 50:
            continue
        say('  %-6s %9.3f %10.3f %10.3f %8.1f' % (
            '%dm' % h, 100*y.mean(), 100*(y.mean()-FEE_T),
            100*(y.mean()-FEE_M), 100*(y > 0).mean()))

    say('\n(D) 逐周稳定性（最优条件，持有5分钟）')
    db = db.copy()
    db['wk'] = pd.to_datetime(db.t, unit='ms', utc=True).dt.isocalendar().week.values
    gw = db.groupby('wk').y5.agg(['count', 'mean'])
    gw = gw[gw['count'] >= 10]
    say('  周数=%d  正收益周=%d (%.1f%%)  周均值中位=%+.3f%%' % (
        len(gw), (gw['mean'] > 0).sum(), 100*(gw['mean'] > 0).mean(), 100*gw['mean'].median()))
    say('  各周: ' + ' '.join('%d:%+.2f%%(n%d)' % (w, 100*row['mean'], row['count'])
                              for w, row in gw.iterrows()))

with open(os.path.join(EV, 'bigwin_grid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.1fs' % (time.time() - t0))
