# -*- coding: utf-8 -*-
"""把 v3 的成交按时间四分，看信号是不是只在某一段活着（跨期稳定性）"""
import pandas as pd, numpy as np

P = r'D:\中信证券\coinlab\evidence\gambler_adaptive_v3_trades.csv'
d = pd.read_csv(P)
for tag in ('REAL', 'SHUF'):
    r = d[d.tag == tag]
    if len(r) < 20:
        print(f'== {tag} 样本太少 n={len(r)}')
        continue
    q = np.quantile(r.t.values, [0, .25, .5, .75, 1.0])
    print(f'== {tag}  n={len(r)}  单笔均值={r.ret.mean()*100:+.4f}%')
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        g = r[(r.t >= lo) & (r.t <= hi)] if i == 3 else r[(r.t >= lo) & (r.t < hi)]
        a = g.ret.values
        if len(a) < 10:
            continue
        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
        print(f'   段{i+1}  n={len(g):>6}  均值={a.mean()*100:+.4f}%  中位={np.median(a)*100:+.4f}%  胜率={(a>0).mean()*100:.1f}%  t={t:+.2f}')
print('\n读法：4 段同号且 |均值| 同量级 ⇒ 稳；只有 1 段撑住 ⇒ 不可信')
