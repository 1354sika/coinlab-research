# -*- coding: utf-8 -*-
"""真实资金曲线 -> PNG（英文标注，避免字体问题）。"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVI = r'D:\中信证券\coinlab\evidence'
d = json.load(open(os.path.join(EVI, 'dd_curve.json')))
cv = d['curve']; pi = d['peak_i']; ti = d['trough_i']; mdd = d['mdd']

fig, ax = plt.subplots(figsize=(13, 5.6), dpi=130)
ax.plot(range(len(cv)), cv, color='#4da3ff', lw=1.1, label='equity (pool=20, 1/20 per trade)')
ax.axhline(1.0, color='#8a8f9c', lw=0.8, ls='--', label='principal 1.00')
ax.plot([pi], [cv[pi]], 'o', color='#c8a34a', ms=6)
ax.plot([ti], [cv[ti]], 'o', color='#9b7bd4', ms=6)
ax.annotate('peak %.3f' % cv[pi], (pi, cv[pi]), textcoords='offset points',
            xytext=(8, -14), color='#c8a34a', fontsize=10)
ax.annotate('trough %.3f' % cv[ti], (ti, cv[ti]), textcoords='offset points',
            xytext=(8, 8), color='#9b7bd4', fontsize=10)
ax.annotate('', xy=(ti, cv[ti]), xytext=(ti, cv[pi]),
            arrowprops=dict(arrowstyle='<->', color='#9b7bd4', lw=1.2))
ax.text(ti - 8, (cv[pi] + cv[ti]) / 2, 'drawdown %.1f%%' % (mdd * 100),
        color='#9b7bd4', fontsize=11, ha='right', va='center')
ax.set_title('Real equity curve from 676 trades  |  134 days  |  final %.4fx' % cv[-1], fontsize=12)
ax.set_xlabel('trade # (sorted by exit time)')
ax.set_ylabel('equity multiple')
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc='lower right', fontsize=9)
ax.set_facecolor('#fbfcfe')
fig.tight_layout()
out = os.path.join(EVI, 'drawdown_curve.png')
fig.savefig(out)
print('written ' + out)
