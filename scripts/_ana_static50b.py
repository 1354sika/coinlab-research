# -*- coding: utf-8 -*-
"""_ana_static50b.py — A 阶段 79,954 条结果：哪些条件是"关键驱动"？（不占 job 并发）

做法：**分阶**（2元/3元/4元）统计——
    高成绩组（测试段 lift > 10x）里条件 X 的出现率  vs  低成绩组（lift <= 2x）里的出现率
    两者的比值 = 该条件对"成绩好"的**富集倍数**。
分阶是为了消除"4 元组合天然含更多条件"的偏差。
"""
import re, os
import numpy as np
import pandas as pd

P = r'D:\中信证券\coinlab\evidence\butterfly\static_50_all.csv'
if not os.path.exists(P):
    print('NOT FOUND', P); raise SystemExit(1)

d = pd.read_csv(P)
d = d[np.isfinite(d['lift_test'])].copy()
d['ncond'] = d['combo'].str.count('&') + 1

# 拆出全部条件名
allc = set()
for c in d['combo']:
    for t in c.split(' & '):
        allc.add(t.strip())
ALLC = sorted(allc)

def parts(c):
    return set(x.strip() for x in c.split(' & '))

d['p'] = d['combo'].apply(parts)

def rate(sub, t):
    if not len(sub): return np.nan
    return np.mean([t in s for s in sub['p']])

print(f'A 阶段可复核组合 {len(d):,}   条件 {len(ALLC)} 个')
print(f'测试段 lift 分布：中位 {d["lift_test"].median():.2f}x  '
      f'p90 {d["lift_test"].quantile(.90):.2f}x  max {d["lift_test"].max():.2f}x')

for k in (2, 3, 4):
    sk = d[d['ncond'] == k]
    if not len(sk): continue
    hi = sk[sk['lift_test'] > 10]
    lo = sk[sk['lift_test'] <= 2]
    print(f'\n{"="*100}')
    print(f'{k} 元组合：{len(sk):,} 条 | 测试 lift>10x: {len(hi):,} | lift<=2x: {len(lo):,}')
    print(f'{"="*100}')
    rows = []
    for t in ALLC:
        rh, rl = rate(hi, t), rate(lo, t)
        if np.isnan(rh) or np.isnan(rl) or rh < 0.02: continue
        rows.append((t, rh, rl, rh / max(rl, 1e-9), len(hi), len(lo)))
    rows.sort(key=lambda x: -x[3])
    print(f'{"条件":<24} {"高分组出现率":>12} {"低分组出现率":>12} {"富集倍数":>10}')
    for t, rh, rl, en, nh, nl in rows[:20]:
        print(f'{t:<24} {100*rh:>11.1f}% {100*rl:>11.1f}% {en:>10.2f}x')
    if rows:
        print(f'  --- 最不重要的 8 个 ---')
        for t, rh, rl, en, nh, nl in rows[-8:]:
            print(f'{t:<24} {100*rh:>11.1f}% {100*rl:>11.1f}% {en:>10.2f}x')

# 全体：条件出现率总览
print(f'\n{"="*100}')
print('全体组合（不分阶）里各条件的出现率')
print(f'{"="*100}')
rr = [(t, rate(d, t), rate(d[d["lift_test"] > 5], t)) for t in ALLC]
rr.sort(key=lambda x: -(x[2] or 0))
for t, r0, r5 in rr[:25]:
    print(f'  {t:<24} 全体 {100*r0:>5.1f}%   高lift组 {100*r5:>5.1f}%')
