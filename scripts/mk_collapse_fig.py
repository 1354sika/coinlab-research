# -*- coding: utf-8 -*-
"""把塌盘概率模型 + 峰值前信号 出成图（数据源：job-14824-3 / job-14824-4 回执）"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
txt = open(os.path.join(EV, 'collapse_proba.txt'), encoding='utf-8').read()
rows = []
for line in txt.splitlines():
    m = re.match(r'\s*\[([\d.]+),([\d.]+)\)\s+([\d,]+)\s+([\d.]+)%\s+([+-][\d.]+)%', line)
    if m:
        rows.append((float(m.group(1)), float(m.group(2)),
                     int(m.group(3).replace(',', '')), float(m.group(4))))

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.7))

# 左：概率分档 vs 真实塌盘率
labs = ['%.2f-%.2f' % (r[0], r[1]) for r in rows]
ax[0].bar(range(len(rows)), [r[3] for r in rows])
ax[0].axhline(0.92, ls='--', color='k', lw=1, label='base rate 0.92%')
ax[0].set_xticks(range(len(rows))); ax[0].set_xticklabels(labs, fontsize=9)
ax[0].set_xlabel('predicted P(collapse in 1h)')
ax[0].set_ylabel('actual collapse rate (%)')
ax[0].set_title('collapse-probability model v1 (test AUC 0.761)')
for i, r in enumerate(rows):
    ax[0].text(i, r[3] + 0.1, 'n=%s' % format(r[2], ','), ha='center', fontsize=8)
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis='y')

# 右：峰值前的信号倍数（相对全样本）
names = ['vol ratio', 'jump rate D4']
base = [3.05, 0.0106]; pre1h = [25.08, 0.1217]; pre15 = [36.12, 0.2696]
x = range(len(names))
mult1 = [pre1h[i] / base[i] for i in range(2)]
mult15 = [pre15[i] / base[i] for i in range(2)]
ax[1].bar([i - 0.2 for i in x], mult1, width=0.4, label='1h before peak')
ax[1].bar([i + 0.2 for i in x], mult15, width=0.4, label='15m before peak')
ax[1].axhline(1, ls='--', color='k', lw=1, label='whole-market baseline')
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names)
ax[1].set_ylabel('x times of baseline')
ax[1].set_title('signals before the top (+100% events, n=230)')
for i in x:
    ax[1].text(i - 0.2, mult1[i] + 0.5, '%.1fx' % mult1[i], ha='center', fontsize=9)
    ax[1].text(i + 0.2, mult15[i] + 0.5, '%.1fx' % mult15[i], ha='center', fontsize=9)
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis='y')

plt.tight_layout()
p = os.path.join(EV, 'collapse_signal.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
print('parsed bins:', rows)
