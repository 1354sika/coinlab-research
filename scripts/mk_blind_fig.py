# -*- coding: utf-8 -*-
"""gambler 双盲检验结果图（数据源：job-14824-8 回执，gambler_blind.txt）"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
# 数据来自 job-14824-8 回执（AUC）
dn = [('coin-holdout', 0.9610), ('label-shuffle', 0.4984),
      ('walk-f1', 0.9047), ('walk-f2', 0.9569), ('walk-f3', 0.9202), ('walk-f4', 0.8980),
      ('y2023', 0.8727), ('y2024', 0.8646), ('y2025', 0.9560), ('y2026', 0.9003)]
up = [('coin-holdout', 0.9152), ('label-shuffle', 0.4984),
      ('walk-f1', 0.9165), ('walk-f2', 0.7876), ('walk-f3', 0.9099), ('walk-f4', 0.9062),
      ('y2023', 0.8623), ('y2024', 0.9082), ('y2025', 0.9070), ('y2026', 0.8751)]

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for k, (data, title) in enumerate(((dn, 'collapse (2h -20%)'), (up, 'rise (2h +20%)'))):
    labs = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = ['C2' if l == 'label-shuffle' else 'C0' for l in labs]
    ax[k].bar(range(len(vals)), vals, color=cols)
    ax[k].axhline(0.5, ls='--', color='k', lw=1, label='0.5 = no skill')
    ax[k].set_xticks(range(len(labs)))
    ax[k].set_xticklabels(labs, rotation=40, ha='right', fontsize=8)
    ax[k].set_ylim(0.4, 1.0)
    ax[k].set_title('gambler blind tests: %s' % title)
    ax[k].grid(alpha=.3, axis='y')
    for i, v in enumerate(vals):
        ax[k].text(i, v + 0.01, '%.2f' % v, ha='center', fontsize=7)
ax[0].set_ylabel('AUC')
ax[0].legend(fontsize=9)
plt.tight_layout()
p = os.path.join(EV, 'gambler_blind.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
