# -*- coding: utf-8 -*-
"""完美预测（前视）的理论上界：如果每分每秒都猜对涨跌，能赚多少？

三种口径：
  A 单币完美预测（零成本）   —— 纯理论天花板
  B 单币完美预测（扣费）     —— 一买一卖的成本能否被吃平
  C 全市场完美预测           —— 每分钟挑 |收益| 最大的那个币
对比：实际机制（+1.7%/笔、一年约 300 笔）

关键判据：平均 |5m 收益| 是否 > 2*fee（一买一卖）
  · maker 2*fee = 0.04%
  · taker 2*fee = 0.20%
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
CAP = 1000.0

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
ar = np.abs(ret1)
valid = np.isfinite(ar)
mean_abs = np.nanmean(ar)
say('全样本平均 |5m 收益| = %.4f%%（%.0f 个观测）' % (100 * mean_abs, valid.sum()))
say('对照成本：maker 往返 %.2f%%   taker 往返 %.2f%%' % (100 * 2 * FEE_M, 100 * 2 * FEE_T))
say('→ 完美预测单笔期望（maker）= %+.4f%%    （taker）= %+.4f%%' % (
    100 * (mean_abs - 2 * FEE_M), 100 * (mean_abs - 2 * FEE_T)))

# A. 单币完美预测（零成本）: 每根 bar 赚 |ret|
say('\n' + '=' * 84)
say('【A】单币完美预测（零成本）—— 理论天花板')
say('  每个币：累计赚 = sum(|ret|)，按 bar 数最多 %d 次' % T)
acc = []
for j in range(N):
    v = ar[:, j]; v = v[np.isfinite(v)]
    if len(v) < 5000:
        continue
    acc.append((syms[j], len(v), 100 * v.sum(), 100 * v.mean()))
acc.sort(key=lambda x: -x[2])
say('  前 10 个币（累计收益 %，假设零成本、每次全押、复利）：')
say('  %-12s %8s %16s %12s' % ('币', 'bar数', '累计|收益|和%', '平均|收益|%'))
for s_, n_, tot_, m_ in acc[:10]:
    # 复利口径
    v = ar[:, syms.index(s_)]; v = v[np.isfinite(v)]
    eq = np.prod(1 + v)
    say('  %-12s %8d %15.0f%% %11.4f%%   复利净值=%.3e' % (s_, n_, tot_, m_, eq))

# B. 单币完美预测（扣费）
say('\n【B】单币完美预测（扣费后，每 bar 一买一卖）')
say('  %-12s %12s %12s %12s' % ('币', '无成本%', 'maker%', 'taker%'))
for s_, n_, tot_, m_ in acc[:5]:
    v = ar[:, syms.index(s_)]; v = v[np.isfinite(v)]
    eqM = np.prod(1 + v - 2 * FEE_M)
    eqT = np.prod(1 + v - 2 * FEE_T)
    eq0 = np.prod(1 + v)
    say('  %-12s %12.3e %12.3e %12.3e' % (s_, eq0, eqM, eqT))

# C. 全市场完美预测：每分钟挑 |ret| 最大的币
say('\n【C】全市场完美预测：每根 bar 挑 |收益| 最大的币，全押')
mx = np.nanmax(ar, axis=1)
mx = mx[np.isfinite(mx)]
say('  平均最佳 |收益| = %.4f%%（比全样本平均 %.4f%% 高 %.1f 倍）' % (
    100 * mx.mean(), 100 * mean_abs, mx.mean() / mean_abs))
eqC0 = np.prod(1 + mx)
eqCM = np.prod(1 + mx - 2 * FEE_M)
eqCT = np.prod(1 + mx - 2 * FEE_T)
say('  零成本复利净值 = %.3e' % eqC0)
say('  maker 复利净值 = %.3e' % eqCM)
say('  taker 复利净值 = %.3e' % eqCT)

# 对比实际机制
say('\n【对照】实际机制（急跌+巨量+多币）')
say('  单笔净 +1.70%%，3 年约 300 笔，K=3 复利：1000 元 -> 约 2000~2300 元')

# 出图
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].hist(100 * ar[np.isfinite(ar)][:2000000], bins=100, range=(-2, 2))
ax[0].axvline(100 * 2 * FEE_M, color='k', ls='--', lw=1, label='maker cost 0.04%')
ax[0].axvline(100 * 2 * FEE_T, color='k', ls=':', lw=1, label='taker cost 0.20%')
ax[0].legend(fontsize=8); ax[0].set_title('|5m return| distribution vs cost')
cums = [np.cumsum(ar[np.isfinite(ar[:, syms.index(s_)]), syms.index(s_)])[-1] for s_, _, _, _ in acc[:10]]
ax[1].bar(range(len(cums)), [100 * c for c in cums])
ax[1].set_xticks(range(len(cums))); ax[1].set_xticklabels([a[0][:6] for a in acc[:10]], rotation=45, fontsize=7)
ax[1].set_title('perfect foresight cumulative |ret| % (no cost)')
cum_m = np.cumsum(mx - 2 * FEE_M); cum_t = np.cumsum(mx - 2 * FEE_T)
ax[2].plot(np.log10(np.maximum(np.cumprod(1 + mx), 1e-300)), label='no cost')
ax[2].plot(np.log10(np.maximum(np.cumprod(1 + np.maximum(mx - 2 * FEE_M, -0.99)), 1e-300)), label='maker')
ax[2].plot(np.log10(np.maximum(np.cumprod(1 + np.maximum(mx - 2 * FEE_T, -0.99)), 1e-300)), label='taker')
ax[2].legend(fontsize=8); ax[2].set_title('log10 equity: market-best every bar')
ax[2].set_xlabel('bar #')
plt.tight_layout()
p = os.path.join(EV, 'perfect_foresight.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'perfect_foresight.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
