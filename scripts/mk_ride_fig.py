# -*- coding: utf-8 -*-
"""跟盘 v2 结果图（数据源：job-14824-13 回执，gambler_ride2.txt）"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
# (入场 A, 止损, 笔数, 期望净%, 中位%, 胜率%)
rows = [
    (0.30, 'none', 1557, -2.390, -3.54, 40.0),
    (0.30, '15%',  1557, -1.108, -3.18, 40.8),
    (0.30, '25%',  1557, -1.700, -3.22, 40.9),
    (0.50, 'none',  531, -2.663, -0.68, 47.6),
    (0.50, '15%',   531, -0.479, -0.04, 49.9),
    (0.50, '25%',   531, -1.088, -0.11, 49.5),
    (1.00, 'none',  132, -0.829, +3.38, 71.2),
    (1.00, '15%',   132, +2.562, +3.44, 76.5),
    (1.00, '25%',   132, +1.595, +3.44, 73.5),
]
labels = ['entry 4h>=%d%%  stop %s' % (100 * r[0], r[1]) for r in rows]
vals = [r[3] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(14, 4.9))
cols = ['C0' if v > 0 else '0.6' for v in vals]
ax[0].barh(range(len(vals)), vals, color=cols)
ax[0].axvline(0, color='k', lw=1)
ax[0].axvline(-0.030, ls='--', color='k', lw=1, label='random entry -0.03%')
ax[0].set_yticks(range(len(vals))); ax[0].set_yticklabels(labels, fontsize=8)
ax[0].set_xlabel('expected net return per trade (%)')
ax[0].set_title('ride v2: no-lookahead, entry = past-4h gain (304 coins / 3y)')
for i, v in enumerate(vals):
    ax[0].text(v + (0.05 if v > 0 else -0.55), i, '%+.2f%%' % v, va='center', fontsize=8)
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis='x')

w = [r[5] for r in rows]
ax[1].barh(range(len(w)), w, color='C1')
ax[1].axvline(50, ls='--', color='k', lw=1, label='50%')
ax[1].set_yticks(range(len(w))); ax[1].set_yticklabels(labels, fontsize=8)
ax[1].set_xlabel('win rate (%)'); ax[1].set_xlim(0, 100)
ax[1].set_title('win rate of each rule')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis='x')
plt.tight_layout()
p = os.path.join(EV, 'gambler_ride2.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
