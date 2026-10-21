# -*- coding: utf-8 -*-
"""gambler_chart.py — 出图：状态条件概率 / 协同倍数 / 真实滑点（三子图）
数据来源（均为本会话已跑出的 job 结果）：
  gambler_state（job-36080-32）、gambler_sync_state（job-36080-35/34）、gambler_depth（job-36080-37）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EV, 'gambler_synopsis.png')

fig = plt.figure(figsize=(16, 10))

# ---- (1) 状态条件概率 ----
ax = fig.add_subplot(2, 2, 1)
names = ['S0\nidle', 'S1\naccum', 'S2\npump', 'S3\nhigh-stall', 'S4\ncrash', 'S5\nother']
p50 = [0.17, 0.18, 10.45, 0.26, 10.07, 0.36]
p30 = [0.16, 0.44, 8.24, 0.14, 13.79, 0.38]
x = np.arange(len(names)); w = 0.38
ax.bar(x - w / 2, p50, w, label='P(>=+50% in 24h)', color='#4C78A8')
ax.bar(x + w / 2, p30, w, label='P(<=-30% in 24h)', color='#9E7BB5')
ax.axhline(0.27, ls='--', lw=1, color='#666666')
ax.text(4.6, 0.5, 'baseline 0.27%', fontsize=8, color='#666666')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('probability (%)', fontsize=10)
ax.set_title('A. State -> conditional probability (150 coins, 3y 5m)\nS2 pump 10.45% / S4 crash 13.79% vs baseline 0.27%', fontsize=11)
ax.legend(fontsize=9)

# ---- (2) 协同倍数 ----
ax = fig.add_subplot(2, 2, 2)
lbl = ['0', '1-2', '3-5', '6-10', '11-20', '21+']
pump_mult = [0.97, 2.15, 3.74, np.nan, np.nan, 26.19]
crash_mult = [0.99, 3.22, 19.18, 3.01, 46.68, 26.02]
x = np.arange(len(lbl)); w = 0.38
ax.bar(x - w / 2, pump_mult, w, label='# of pumping coins -> lift', color='#4C78A8')
ax.bar(x + w / 2, crash_mult, w, label='# of crashing coins -> lift', color='#7EA6C9')
ax.axhline(1.0, ls='--', lw=1, color='#666666')
for i, v in enumerate(pump_mult):
    if np.isfinite(v):
        ax.text(i - w / 2, v + 1, f'{v:.1f}x', ha='center', fontsize=8)
for i, v in enumerate(crash_mult):
    ax.text(i + w / 2, v + 1, f'{v:.1f}x', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=9)
ax.set_xlabel('number of coins in that state at the same time', fontsize=10)
ax.set_ylabel('lift vs baseline (x)', fontsize=10)
ax.set_title('B. Cross-sectional resonance\npump 26x / crash 47x', fontsize=11)
ax.legend(fontsize=9)

# ---- (3) 真实滑点 ----
ax = fig.add_subplot(2, 2, 3)
sizes = [100, 500, 1000, 5000, 10000, 50000]
main = [0.000, 0.000, 0.001, 0.001, 0.003, 0.008]
pump = [0.023, 0.030, 0.046, 0.098, 0.153, 0.640]
dump = [0.087, 0.135, 0.178, 0.386, 0.530, 1.062]
xs = np.arange(len(sizes))
ax.plot(xs, main, 'o-', label='major coins', color='#4C78A8')
ax.plot(xs, pump, 's-', label='pumping coins (24h>=+20%)', color='#9E7BB5')
ax.plot(xs, dump, '^-', label='crashing coins (24h<=-20%)', color='#B07A5A')
ax.axhline(0.5, ls='--', lw=1, color='#888888')
ax.text(0.05, 0.52, 'old assumption: 0.5% = death line', fontsize=8, color='#888888')
ax.set_xticks(xs); ax.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in sizes], fontsize=9)
ax.set_xlabel('order size (USDT)', fontsize=10)
ax.set_ylabel('market-order slippage (%)', fontsize=10)
ax.set_title('C. REAL order-book slippage (measured)\npumping coin @1000$ = 0.046% (11x lower than assumed)', fontsize=11)
ax.legend(fontsize=9)

# ---- (4) 提前性 ----
ax = fig.add_subplot(2, 2, 4)
labels = ['momentum\n(1h ret>=5%)', 'accum+falsebreak', 'accum only']
already = [6.42, -1.72, 0.24]
fwd24 = [-1.53, 0.65, -0.25]
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w / 2, already, w, label='already moved before signal (%)', color='#B07A5A')
ax.bar(x + w / 2, fwd24, w, label='forward 24h (%)', color='#4C78A8')
ax.axhline(0, lw=1, color='#333333')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('%', fontsize=10)
ax.set_title('D. Lead vs lag\naccum signal fires at 0% move but has no edge; false-break has both', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=110)
print('saved', OUT)
