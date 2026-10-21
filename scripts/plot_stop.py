# -*- coding: utf-8 -*-
"""画出止损距离 vs 触发率 / 净收益 的关系（用 stoprefine 的结果，硬编码）"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence')

sl = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])          # stop distance %
trig5 = np.array([0.0, 77.7, 71.0, 55.6, 40.0, 29.6, 17.0])   # trigger % (hold 5m)
trig15 = np.array([0.0, 82.9, 77.9, 64.5, 48.3, 36.7, 21.8])
trig60 = np.array([0.0, 86.2, 82.5, 72.3, 57.6, 46.1, 28.7])
net5 = np.array([0.511, 0.095, 0.073, 0.056, -0.062, -0.087, -0.031])
net15 = np.array([0.552, 0.119, 0.089, 0.092, 0.006, -0.011, 0.026])
net60 = np.array([0.751, 0.144, 0.123, 0.157, 0.066, 0.104, 0.206])

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axs[0]
ax.plot(sl[1:], trig5[1:], 'o-', label='hold 5m')
ax.plot(sl[1:], trig15[1:], 's-', label='hold 15m')
ax.plot(sl[1:], trig60[1:], '^-', label='hold 60m')
ax.set_xlabel('stop-loss distance (%)')
ax.set_ylabel('trigger rate (%)')
ax.set_title('Tighter stop -> triggers almost always')
ax.legend(); ax.grid(alpha=.3)

ax = axs[1]
ax.axhline(0, color='gray', lw=.8)
ax.plot(sl, net5, 'o-', label='hold 5m')
ax.plot(sl, net15, 's-', label='hold 15m')
ax.plot(sl, net60, '^-', label='hold 60m')
ax.scatter([0], [0.751], s=120, marker='*', zorder=5)
ax.annotate('no stop,\nhold 60m: +0.751%', (0, 0.751), textcoords='offset points',
            xytext=(12, -18), fontsize=9)
ax.set_xlabel('stop-loss distance (%)  (0 = no stop)')
ax.set_ylabel('net return per trade after maker fee (%)')
ax.set_title('Every stop-loss LOWERS the return')
ax.legend(); ax.grid(alpha=.3)

plt.tight_layout()
p = os.path.join(EV, 'stoprefine.png')
plt.savefig(p, dpi=125)
print('saved', p)

# 止损口径对比图
mid = np.array([-21.0, -28.7, -20.4, -27.3, -21.0, -28.9, -9.8, -14.3, -10.1, -14.3, -9.7, -14.3])
labels = ['all\n0.5%', 'all\n1%', 'disp\n0.5%', 'disp\n1%', 'nondisp\n0.5%', 'nondisp\n1%']
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(range(6), mid[:6], label='hold 15m')
ax.bar(range(6, 12), mid[6:], label='hold 60m')
ax.set_xticks(range(6))
ax.set_xticklabels(labels)
ax.axhline(0, color='gray', lw=.8)
ax.set_ylabel('relative diff in trigger rate (%)')
ax.set_title('mid-price stop triggers LESS often than close (all layers, incl. dispersion)')
ax.legend()
plt.tight_layout()
p2 = os.path.join(EV, 'stoprefine_mid.png')
plt.savefig(p2, dpi=125)
print('saved', p2)
