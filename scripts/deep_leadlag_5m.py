# -*- coding: utf-8 -*-
"""深埋的关系：5m 粒度下的跨币领先-滞后（先去共同因子）

背景：1m 粒度下测过「最强 lag=0 占比 100%」——噪声太大，看不出领先。
本脚本在 5m 粒度上重测，并且：
  ① 先去共同因子（减去市场均值），只看残差 —— 否则看到的全是"市场一起动"
  ② 逐对算 lag = -6..+6 根（即 ±30 分钟）的互相关
  ③ 挑出 lag≠0 且相关最强的对
  ④ 可交易性检验：领先方的收益能否预测落后方下一根（扣费后是否还有剩）

判据：
  · 若最强 lag 仍集中于 0  -> 「更深关系」不在跨币领先里（至少这个粒度/这 60 币上没有）
  · 若存在稳定的非 0 lag -> 找到了，再谈能不能交易
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
LAGS = list(range(-6, 7))

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
del df
P = px.values
T, N = P.shape
tms = px.index.values
syms = list(px.columns)
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
mkt = np.nanmean(ret1, axis=1)
R = ret1 - mkt[:, None]                      # 残差：去掉共同因子
R = np.where(np.isfinite(R), R, 0.0)
say('市场因子方差占比 = %.1f%%' % (
    100 * (1 - np.nanvar(R) / np.nanvar(ret1))))

best = np.zeros((N, N)); bestlag = np.zeros((N, N), np.int32)
say('计算 lag 相关矩阵（%d 个 lag）...' % len(LAGS))
for k in LAGS:
    if k >= 0:
        A, B = R[:T - k], R[k:]
    else:
        A, B = R[-k:], R[:T + k]
    C = np.corrcoef(A.T, B.T)[:N, N:]        # C[i,j] = corr(i @ t, j @ t+k)
    upd = np.abs(C) > np.abs(best)
    best = np.where(upd, C, best)
    bestlag = np.where(upd, k, bestlag)
    say('  lag=%+d 完成  %.0fs' % (k, time.time() - t0))

# 排除自相关（i==j）
np.fill_diagonal(best, 0.0); np.fill_diagonal(bestlag, 0)

say('\n' + '=' * 92)
say('【一】最强 lag 的分布（%d 个币对）' % (N * (N - 1)))
vals, cnts = np.unique(bestlag[bestlag != 0], return_counts=True)
say('  %8s %10s %10s' % ('lag', '对数', '占比%'))
tot = int((bestlag != 0).sum())
for v, c in zip(vals, cnts):
    say('  %+8d %10d %9.2f%%' % (v, c, 100 * c / tot))
say('  lag=0 的对数 = %d（占全部 %d 对的 %.1f%%）' % (
    int((bestlag == 0).sum()), N * (N - 1), 100 * (bestlag == 0).sum() / (N * (N - 1))))

say('\n【二】lag≠0 且 |相关| 最强的 20 对')
bi, bj = np.where(bestlag != 0)
order = np.argsort(-np.abs(best[bi, bj]))[:20]
say('  %-12s -> %-12s %8s %8s' % ('A(领先)', 'B(落后)', 'lag', 'corr'))
for t_ in order:
    i, j = bi[t_], bj[t_]
    say('  %-12s -> %-12s %+8d %+8.4f' % (syms[i], syms[j], bestlag[i, j], best[i, j]))

say('\n【三】这些"领先关系"长期稳定吗？（第 4 节：可交易性）')
say('  说明：用最强领先对做检验 —— 领先方某一根大涨/大跌，落后方下一根跟不跟')
ok_pairs = [(syms[bi[t_]], syms[bj[t_]], int(bestlag[bi[t_], bj[t_]]), best[bi[t_], bj[t_]])
            for t_ in order[:5]]
for a, b, lg, cc in ok_pairs:
    ia, ib = syms.index(a), syms.index(b)
    if lg > 0:      # a 领先 b
        sig = R[:-lg, ia]; nxt = R[lg:, ib]
    else:
        sig = R[-lg:, ia]; nxt = R[:T + lg, ib]
    thr = np.nanpercentile(np.abs(sig), 90)
    m = np.abs(sig) >= thr
    direction = np.sign(sig[m])
    fwd = nxt[m]
    hit = (np.sign(fwd) == direction).mean()
    gross = (direction * fwd).mean()
    say('  %-10s->%-10s lag=%+d corr=%+.4f  n=%5d  同向率=%.1f%%  毛=%+.4f%%  净maker=%+.4f%%  净taker=%+.4f%%' % (
        a, b, lg, cc, int(m.sum()), 100 * hit, 100 * gross,
        100 * (gross - 2 * FEE_M), 100 * (gross - 2 * FEE_T)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].bar([str(v) for v, _ in zip(vals, cnts)][:20], cnts[:20])
ax[0].set_title('distribution of best lag'); ax[0].set_xlabel('best lag (bars)')
im = ax[1].imshow(best, cmap='viridis', vmin=-np.abs(best).max(), vmax=np.abs(best).max())
ax[1].set_title('best |corr| matrix (residual, 5m)'); plt.colorbar(im, ax=ax[1])
ax[2].hist(best.flatten(), bins=60); ax[2].set_title('|corr| distribution across pairs')
plt.tight_layout()
p = os.path.join(EV, 'deep_leadlag_5m.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'deep_leadlag_5m.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
