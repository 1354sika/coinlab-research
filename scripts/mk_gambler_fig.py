# -*- coding: utf-8 -*-
"""gambler 出图：精确率-覆盖率 曲线 + 阈值曲线（数据源 gambler_proba.npz / job-14824-5）"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
z = np.load(os.path.join(EV, 'gambler_proba.npz'))
fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.0))
ts = np.arange(0.05, 0.995, 0.005)
for tag, pk, yk, col in (('collapse  2h  -20%', 'p_dn', 'y_dn', 'C0'),
                         ('rise      2h  +20%', 'p_up', 'y_up', 'C1')):
    p = z[pk]; y = z[yk]
    prec, cov, rec = [], [], []
    for t in ts:
        m = p >= t
        if m.sum() < 20:
            prec.append(np.nan); cov.append(np.nan); rec.append(np.nan); continue
        prec.append(y[m].mean()); cov.append(m.mean()); rec.append(y[m].sum() / y.sum())
    ax[0].plot(cov, prec, label=tag, color=col, lw=2)
    ax[1].plot(ts, prec, label=tag + '  precision', color=col, lw=2)
    ax[1].plot(ts, rec, label=tag + '  recall', color=col, lw=1.2, ls='--')
ax[0].axhline(0.85, ls='--', color='k', lw=1, label='85% target')
ax[0].set_xlabel('coverage (share of reported signals)')
ax[0].set_ylabel('precision (real event rate)')
ax[0].set_ylim(0.3, 1.02); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title('gambler v1: precision vs coverage')
ax[1].axhline(0.85, ls='--', color='k', lw=1)
ax[1].set_xlabel('threshold on P'); ax[1].set_ylabel('rate')
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title('gambler v1: precision / recall vs threshold')
plt.tight_layout()
p = os.path.join(EV, 'gambler.png')
plt.savefig(p, dpi=125, bbox_inches='tight')
print('saved:', p)
