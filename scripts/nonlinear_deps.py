# -*- coding: utf-8 -*-
"""非线性 / 条件依赖：线性相关之外，还藏着什么关系？

背景：跨币领先-滞后（线性）在所有粒度上都不成立（lag±1 完全对称 = 噪声）。
本脚本换四种"非线性视角"重挖，全部用 5m、60 币、去市场残差 R：

  ① 秩相关 vs 皮尔逊      —— 单调非线性是否被线性低估
  ② 尾部共现 lift         —— A 在下尾 10% 时 B 也在下尾的概率 / 独立期望
  ③ 波动传导              —— A 的 |ret| 在上尾 10% 时，B 下一根的 |ret| 是否变大
  ④ 条件收益              —— A 极端涨/跌后，B 下一根收益的偏移（扣费后是否还有剩）

判据：任何"发现"都必须过可交易性（毛收益 > 成本），否则只是统计现象
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

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
del df
P = px.values
T, N = P.shape
syms = list(px.columns)
ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
mkt = np.nanmean(ret1, axis=1)
R = ret1 - mkt[:, None]
R = np.where(np.isfinite(R), R, np.nan)
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))

# ============ ① 秩相关 vs 皮尔逊 ============
say('\n' + '=' * 92)
say('【①】秩相关（Spearman）vs 皮尔逊 —— 单调非线性是否被线性低估')
# 秩变换（对每币各自）
Rr = np.apply_along_axis(lambda v: pd.Series(v).rank().values, 0, R)
Pear = np.corrcoef(np.nan_to_num(R, nan=0.0).T)
Spear = np.corrcoef(np.nan_to_num(Rr, nan=0.0).T)
iu = np.triu_indices(N, 1)
dif = np.abs(Spear[iu] - Pear[iu])
say('  币对数 = %d' % len(dif))
say('  |Spearman - Pearson| 中位 = %.4f  均值 = %.4f  >0.05 的占比 = %.1f%%' % (
    np.median(dif), dif.mean(), 100 * (dif > 0.05).mean()))
say('  |Spearman| 中位 = %.4f   |Pearson| 中位 = %.4f' % (
    np.median(np.abs(Spear[iu])), np.median(np.abs(Pear[iu]))))
say('  → 秩相关普遍更弱/similar（若更强说明有单调非线性）')

# ============ ② 尾部共现 ============
say('\n【②】下尾共现 lift（A 在下尾 10%，B 也在下尾的概率 / 独立期望 0.01）')
q10 = np.nanpercentile(R, 10, axis=0)
tail = (R <= q10[None, :]).astype(np.float32)
n_valid = np.isfinite(R).sum(axis=0).astype(np.float32)
co = (tail.T @ tail) / T                          # N x N 共现率
lift = co / (0.10 * 0.10)
np.fill_diagonal(lift, np.nan)
lv = lift[iu]
say('  lift 中位 = %.3f  均值 = %.3f  P95 = %.3f  最大 = %.3f' % (
    np.nanmedian(lv), np.nanmean(lv), np.nanpercentile(lv, 95), np.nanmax(lv)))
top = np.argsort(-np.nan_to_num(lv))[:10]
say('  lift 最高 10 对：')
say('  %-12s -> %-12s %8s' % ('A', 'B', 'lift'))
for t_ in top:
    i, j = iu[0][t_], iu[1][t_]
    say('  %-12s    %-12s %8.3f' % (syms[i], syms[j], lv[t_]))
# 上尾
q90 = np.nanpercentile(R, 90, axis=0)
tailU = (R >= q90[None, :]).astype(np.float32)
liftU = ((tailU.T @ tailU) / T) / 0.01
np.fill_diagonal(liftU, np.nan)
say('  上尾 lift：中位 = %.3f  P95 = %.3f  最大 = %.3f' % (
    np.nanmedian(liftU[iu]), np.nanpercentile(liftU[iu], 95), np.nanmax(liftU[iu])))

# ============ ③ 波动传导 ============
say('\n【③】波动传导：A 的 |ret| 在上尾 10% 时，B 下一根 |ret| 是否变大')
av = np.abs(R)
qa = np.nanpercentile(av, 90, axis=0)
hiA = av > qa[None, :]
avn = np.vstack([av[1:], np.full((1, N), np.nan)])      # 下一根
base = np.nanmean(av, axis=0)
ratio = np.full((N, N), np.nan)
for i in range(N):
    m = hiA[:, i] & np.isfinite(avn).all(axis=1)
    if m.sum() > 500:
        ratio[i, :] = np.nanmean(avn[m, :], axis=0) / base
np.fill_diagonal(ratio, np.nan)
rv = ratio[iu]
say('  |ret| 提升倍数：中位 = %.3f  均值 = %.3f  P95 = %.3f' % (
    np.nanmedian(rv), np.nanmean(rv), np.nanpercentile(rv, 95)))
say('  （>1 表示 A 剧烈波动时 B 下一根也更剧烈；但这里没有方向性 -> 不可直接交易）')
say('  注意：这一项即使成立也【没有方向】，无法直接变成买卖信号')

# ============ ④ 条件收益（这才可能可交易） ============
say('\n【④】条件收益：A 极端涨/跌后，B 下一根的收益（扣费后）')
fwd = np.vstack([R[1:], np.full((1, N), np.nan)])
res4 = []
for i in range(N):
    s_ = R[:, i]
    for thr, lb in ((np.nanpercentile(s_, 5), 'A 跌5%分位'), (np.nanpercentile(s_, 95), 'A 涨95%分位')):
        m = ((s_ <= thr) if '跌' in lb else (s_ >= thr)) & np.isfinite(fwd[:, 0])
        if m.sum() < 200:
            continue
        prof = np.nanmean(fwd[m, :], axis=0)          # 各 B 的下一根均值
        prof[i] = np.nan
        k = np.nanargmax(np.abs(prof))
        res4.append((syms[i], lb, syms[k], 100 * prof[k], 100 * (prof[k] - 2 * FEE_M), m.sum()))
res4.sort(key=lambda x: -abs(x[3]))
say('  %-12s %-10s -> %-12s %10s %11s %8s' % ('A', '条件', 'B', '毛均值%', '净maker%', 'n'))
for r in res4[:12]:
    say('  %-12s %-10s -> %-12s %+9.4f%% %+10.4f%% %8d' % r)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].hist(dif, bins=60); ax[0].axvline(0.05, color='k', ls='--', lw=.8)
ax[0].set_title('|Spearman - Pearson|  (median %.3f)' % np.median(dif))
ax[1].hist(lv[np.isfinite(lv)], bins=60); ax[1].axvline(1.0, color='k', ls='--', lw=.8, label='independent=1')
ax[1].legend(fontsize=8); ax[1].set_title('lower-tail co-occurrence lift')
ax[2].hist(rv[np.isfinite(rv)], bins=60); ax[2].axvline(1.0, color='k', ls='--', lw=.8)
ax[2].set_title('vol transmission ratio (next bar)')
plt.tight_layout()
p = os.path.join(EV, 'nonlinear_deps.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'nonlinear_deps.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
