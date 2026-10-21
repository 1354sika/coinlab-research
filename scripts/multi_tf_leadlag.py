# -*- coding: utf-8 -*-
"""多粒度领先-滞后：15m / 1h / 日线（去市场因子）

5m 粒度实测：lag=0 占 66.6%、lag±1 完全对称、可交易性为负。
本脚本换更粗粒度重测（噪声更低，"深埋关系"若存在应更容易浮出）。

做法：把 5m 收盘价重采样成 15m / 1h / 日线
      每个粒度：去市场均值 -> 残差 -> lag=-4..+4 相关矩阵
      统计：lag=0 占比 / ±对称性 / 最强对 / 可交易性（毛收益 vs 成本）
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_5m_3y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

FEE_M, FEE_T = 0.0002, 0.0010
RULES = {'5m': None, '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
LAGS = list(range(-4, 5))

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
del df
px.index = pd.to_datetime(px.index, unit='ms', utc=True)
say('原始 matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

summary = []
for tf, rule in RULES.items():
    if rule is None:
        p = px
    else:
        p = px.resample(rule).last().dropna(how='all')
    P = p.values
    T, N = P.shape
    ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
    mkt = np.nanmean(ret1, axis=1)
    R = ret1 - mkt[:, None]
    R = np.where(np.isfinite(R), R, 0.0)
    mshare = 100 * (1 - np.nanvar(R) / np.nanvar(ret1))
    say('\n' + '=' * 92)
    say('【粒度 %s】T=%d 根  N=%d 币   市场因子占方差 %.1f%%' % (tf, T, N, mshare))
    best = np.zeros((N, N)); bestlag = np.zeros((N, N), np.int32)
    for k in LAGS:
        if k >= 0:
            A, B = R[:T - k], R[k:]
        else:
            A, B = R[-k:], R[:T + k]
        C = np.corrcoef(A.T, B.T)[:N, N:]
        upd = np.abs(C) > np.abs(best)
        best = np.where(upd, C, best)
        bestlag = np.where(upd, k, bestlag)
    np.fill_diagonal(best, 0.0); np.fill_diagonal(bestlag, 0)
    nz = int((bestlag != 0).sum()); z = int((bestlag == 0).sum()); tot = N * (N - 1)
    sym_p1 = int((bestlag == 1).sum()); sym_m1 = int((bestlag == -1).sum())
    say('  lag=0 的对 = %d (%.1f%%)   lag≠0 = %d (%.1f%%)' % (z, 100 * z / tot, nz, 100 * nz / tot))
    say('  对称性检验：lag=+1 有 %d 对，lag=-1 有 %d 对 —— 差值 %d（越接近 0 越像噪声）' % (
        sym_p1, sym_m1, abs(sym_p1 - sym_m1)))
    bi, bj = np.where(bestlag != 0)
    if len(bi):
        order = np.argsort(-np.abs(best[bi, bj]))[:8]
        say('  lag≠0 最强 8 对：')
        say('  %-12s -> %-12s %6s %9s' % ('A', 'B', 'lag', 'corr'))
        for t_ in order:
            i, j = bi[t_], bj[t_]
            say('  %-12s -> %-12s %+6d %+9.4f' % (p.columns[i], p.columns[j], bestlag[i, j], best[i, j]))
        # 可交易性：用最强那对
        i, j = bi[order[0]], bj[order[0]]
        lg = int(bestlag[i, j])
        if lg > 0:
            sig, nxt = R[:-lg, i], R[lg:, j]
        else:
            sig, nxt = R[-lg:, i], R[:T + lg, j]
        thr = np.nanpercentile(np.abs(sig), 90)
        m = np.abs(sig) >= thr
        d = np.sign(sig[m]); fwd = nxt[m]
        hit = (np.sign(fwd) == d).mean(); gross = (d * fwd).mean()
        say('  可交易性（最强对 %s->%s, lag=%+d）：n=%d 同向率=%.1f%% 毛=%+.4f%% 净maker=%+.4f%% 净taker=%+.4f%%' % (
            p.columns[i], p.columns[j], lg, int(m.sum()), 100 * hit, 100 * gross,
            100 * (gross - 2 * FEE_M), 100 * (gross - 2 * FEE_T)))
        summary.append((tf, tot, 100 * z / tot, sym_p1, sym_m1, p.columns[i], p.columns[j], lg,
                        100 * best[i, j], 100 * gross, 100 * (gross - 2 * FEE_M)))

say('\n' + '=' * 92)
say('【汇总：各粒度对比】')
say('  %-6s %8s %10s %8s %8s %-24s %9s %10s' % (
    '粒度', '对数', 'lag0占比%', 'lag+1', 'lag-1', '最强对', 'corr%', '净maker%'))
for r in summary:
    say('  %-6s %8d %9.1f%% %8d %8d %-24s %8.2f %+9.4f' % (
        r[0], r[1], r[2], r[3], r[4], '%s->%s(lag%+d)' % (r[5][:6], r[6][:6], r[7]), r[8], r[10]))

if summary:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xs = range(len(summary))
    ax[0].bar([x - 0.2 for x in xs], [r[3] for r in summary], 0.4, label='lag=+1')
    ax[0].bar([x + 0.2 for x in xs], [r[4] for r in summary], 0.4, label='lag=-1')
    ax[0].set_xticks(xs); ax[0].set_xticklabels([r[0] for r in summary])
    ax[0].legend(fontsize=8); ax[0].set_title('symmetry: #pairs at lag+1 vs lag-1')
    ax[1].bar(xs, [r[10] for r in summary])
    ax[1].set_xticks(xs); ax[1].set_xticklabels([r[0] for r in summary])
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_ylabel('net maker %')
    ax[1].set_title('tradability of strongest lead pair')
    plt.tight_layout()
    pth = os.path.join(EV, 'multi_tf_leadlag.png'); plt.savefig(pth, dpi=125, bbox_inches='tight')
    say('\nsaved: %s' % pth)

open(os.path.join(EV, 'multi_tf_leadlag.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
