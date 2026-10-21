# -*- coding: utf-8 -*-
"""闭环两版对照图（数据源：job-14824-9 有前视泄漏 / job-14824-10 无前视）"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
labels = ['loop v1\n(leaky:\nuses event start)', 'loop v2\n(clean:\nno event start)', 'random\nhold 24 bars']
win = [88.7, 36.7, 46.5]
net = [22.22, -0.674, 0.000]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
xs = range(3)
ax[0].bar(xs, win, color=['0.6', 'C0', 'C2'])
ax[0].axhline(50, ls='--', color='k', lw=1, label='50%')
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel('win rate (%)'); ax[0].set_ylim(0, 100)
ax[0].set_title('win rate: leaky vs clean vs random')
for i, v in enumerate(win):
    ax[0].text(i, v + 2, '%.1f%%' % v, ha='center', fontsize=10)
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis='y')

ax[1].bar(xs, net, color=['0.6', 'C0', 'C2'])
ax[1].axhline(0, ls='--', color='k', lw=1)
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(labels, fontsize=8)
ax[1].set_ylabel('net return per trade (%)')
ax[1].set_title('net maker return per trade')
for i, v in enumerate(net):
    ax[1].text(i, v + (0.6 if v >= 0 else -1.6), '%+.3f%%' % v, ha='center', fontsize=10)
ax[1].grid(alpha=.3, axis='y')
plt.tight_layout()
p = os.path.join(EV, 'gambler_loop_compare.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
