# -*- coding: utf-8 -*-
"""_ana_static50.py — 快速看 A 阶段（核心池50，已穷举 79,954 条）的真实成绩

关键：训练段 lift 会被多重比较严重膨胀 → **只看测试段 lift**
"""
import os
import numpy as np
import pandas as pd

P = r'D:\中信证券\coinlab\evidence\butterfly\static_50_all.csv'
if not os.path.exists(P):
    print('NOT FOUND', P); raise SystemExit(1)
d = pd.read_csv(P)
print(f'总组合 {len(d):,}')
ok = d[np.isfinite(d['lift_test'])].copy()
print(f'测试段可复核 {len(ok):,}')
print(f'测试段 lift：中位 {ok["lift_test"].median():.2f}x  p90 {ok["lift_test"].quantile(.90):.2f}x  max {ok["lift_test"].max():.2f}x')
print(f'  >10x: {int((ok["lift_test"]>10).sum()):,}   >5x: {int((ok["lift_test"]>5).sum()):,}   '
      f'>2x: {int((ok["lift_test"]>2).sum()):,}   <1x(崩): {int((ok["lift_test"]<1).sum()):,}')
print()
print('=' * 110)
print('【按测试段 lift 排序 Top-25】（≥100 测试样本）')
print('=' * 110)
big = ok[ok['n_test'] >= 100].sort_values('lift_test', ascending=False)
for _, r in big.head(25).iterrows():
    print(f'  {r["combo"][:72]:<72} 训lift {r["lift_train"]:>7.1f}x(n={int(r["n_train"]):>6,}) | '
          f'测lift {r["lift_test"]:>7.1f}x(n={int(r["n_test"]):>6,}) 测真局率 {100*r["rate_test"]:.2f}%')
print()
print('=' * 110)
print('【按训练段 lift 排序 Top-15 → 看测试段衰减（过拟合诊断）】')
print('=' * 110)
tt = d.sort_values('lift_train', ascending=False).head(15)
for _, r in tt.iterrows():
    lt = r['lift_test']
    tag = '★稳' if (np.isfinite(lt) and lt > 5) else ('衰减' if (np.isfinite(lt) and lt > 1) else '崩')
    print(f'  {r["combo"][:66]:<66} 训 {r["lift_train"]:>8.1f}x | 测 {lt:>7.1f}x  [{tag}]')
