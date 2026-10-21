# -*- coding: utf-8 -*-
"""从 pump_hunt_events.csv 出图：操盘手（短期暴涨）事件分布"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
C = {}
df = pd.read_csv(os.path.join(EV, 'pump_hunt_events.csv'))
fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.9))

# 1 各门槛事件数 + 覆盖币数
g = df.groupby('thr')
cnt = g.size()
coin = g['symbol'].nunique()
x = np.arange(len(cnt))
ax[0].bar(x - 0.2, cnt.values, width=0.4, label='events')
ax[0].bar(x + 0.2, coin.values, width=0.4, label='coins')
ax[0].set_xticks(x); ax[0].set_xticklabels(['+%d%%' % (t * 100) for t in cnt.index])
ax[0].set_ylabel('count')
ax[0].set_title('pump events (4h window, 304 coins / 3y)')
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis='y')

# 2 涨幅分布（+100% 档）
d = df[df.thr == 1.0]
ax[1].hist(100 * d.gain, bins=30)
ax[1].set_xlabel('gain at/within 4h (%)'); ax[1].set_ylabel('events')
ax[1].set_title('gain distribution (+100%% threshold, n=%d)' % len(d))
ax[1].grid(alpha=.3, axis='y')

# 3 到峰值耗时（分钟）分布，按门槛
for t, lab in ((0.5, '+50%'), (1.0, '+100%'), (2.0, '+200%')):
    dd = df[df.thr == t]
    if len(dd):
        ax[2].hist(dd.min_to_peak, bins=np.arange(0, 245, 15), alpha=.55, label='%s (n=%d)' % (lab, len(dd)))
ax[2].set_xlabel('minutes from window start to peak')
ax[2].set_ylabel('events')
ax[2].set_title('time to peak')
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3, axis='y')

plt.tight_layout()
p = os.path.join(EV, 'pump_hunt.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
print('events by threshold:', dict(cnt))
print('coins by threshold:', dict(coin))
print('gain median by thr:', {k: round(float(np.median(v)), 3) for k, v in df.groupby('thr')['gain']})
