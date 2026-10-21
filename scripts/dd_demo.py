# -*- coding: utf-8 -*-
"""用 676 笔真实明细算资金曲线与最大回撤，作为「回撤」概念的真实例子。"""
import pandas as pd, json, os

EVI = r'D:\中信证券\coinlab\evidence'
d = pd.read_csv(os.path.join(EVI, 'trades_1m_tp500_sl35_p20.csv'))
POOL = 20

d = d.sort_values('exit_ms').reset_index(drop=True)
eq = 1.0
peak = 1.0
peak_i = 0
curve = []
mdd = 0.0
mdd_i = 0
mdd_peak_i = 0

for i, r in d.iterrows():
    eq *= (1.0 + float(r['net']) / POOL)
    curve.append(eq)
    if eq > peak:
        peak = eq
        peak_i = i
    dd = eq / peak - 1.0
    if dd < mdd:
        mdd = dd
        mdd_i = i
        mdd_peak_i = peak_i

ts = d['exit_ms'].tolist()

L = []
L.append('资金曲线与最大回撤（676 笔真实明细，池 20，每笔占 1/20 资金）')
L.append('=' * 70)
L.append('起点 1.0000   终点 %.4f   （134 天，累计 %+.2f%%）' % (eq, (eq - 1) * 100))
L.append('最大回撤 %.2f%%' % (mdd * 100))
L.append('')
L.append('峰值：第 %d 笔，equity %.4f，时间 %s' % (
    mdd_peak_i, curve[mdd_peak_i], pd.to_datetime(ts[mdd_peak_i], unit='ms')))
L.append('谷底：第 %d 笔，equity %.4f，时间 %s' % (
    mdd_i, curve[mdd_i], pd.to_datetime(ts[mdd_i], unit='ms')))
L.append('峰到谷相隔：%d 笔 / %.1f 天' % (
    mdd_i - mdd_peak_i,
    (ts[mdd_i] - ts[mdd_peak_i]) / 86400000.0))
L.append('')
L.append('--- 换成 1000 元本金看得见的样子 ---')
L.append('峰值时        : %.0f 元' % (1000 * curve[mdd_peak_i]))
L.append('谷底时        : %.0f 元' % (1000 * curve[mdd_i]))
L.append('这一段蒸发掉  : %.0f 元  （= 峰值的 %.1f%%）' % (
    1000 * (curve[mdd_peak_i] - curve[mdd_i]),
    (1 - curve[mdd_i] / curve[mdd_peak_i]) * 100))
L.append('要回到峰值还需涨: %.1f%%' % (
    (curve[mdd_peak_i] / curve[mdd_i] - 1) * 100))
L.append('')
L.append('--- 回撤与「回本难度」的关系（通用） ---')
for dd in (0.10, 0.20, 0.30, 0.50, 0.70):
    L.append('  回撤 -%2.0f%%  =>  需要涨 %5.1f%% 才能回本' % (dd * 100, (1 / (1 - dd) - 1) * 100))

out = os.path.join(EVI, 'drawdown_demo.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))

json.dump({'curve': [round(x, 5) for x in curve],
           'peak_i': mdd_peak_i, 'trough_i': mdd_i, 'mdd': mdd,
           'ts': ts},
          open(os.path.join(EVI, 'dd_curve.json'), 'w'))
print('\nwritten ' + out)
