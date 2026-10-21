# -*- coding: utf-8 -*-
"""analyze_cashout.py — 读 combo_scan_all.csv，看「落袋为安」机制的效果。

落袋定义：账上权益**首次**触到 cashout 倍本金，就把多出的部分提走（现金落袋），只留本金继续；**只提一次，之后永不再提**。
  cashout=2.0 → 首次到 2000 提走 1000，回 1000 继续（之后不再提）
  cashout=3.0 → 首次到 3000 提走 2000，回 1000 继续（之后不再提）
  cashout=5.0 → 首次到 5000 提走 4000，回 1000 继续（之后不再提）
  0           → 不提，全额复利
总资产 = 账上权益 + 累计提出。
"""
import os
import pandas as pd
import numpy as np

EVI = r'D:\中信证券\coinlab\evidence'
CSV = os.path.join(EVI, 'combo_scan_all.csv')
R = pd.read_csv(CSV)

L = []
L.append('落袋为安（本金提取）机制 · 结果分析')
L.append('=' * 130)
L.append('数据源 combo_scan_all.csv：%d 行（1440 = 12 组信号 × 6 池N × 5 日停手 × 4 落袋档）' % len(R))
L.append('初始本金 1000 元。总资产 = 账上权益 + 已提出（只提一次）。')
L.append('')

# ---- 1. 用户点名的三组 ----
L.append('--- 一、你点名的三组（池 20、不停手）在三档落袋下的表现 ---')
L.append('%-12s %6s %6s %8s | %-14s %14s %14s %8s' % (
    '信号', 'TP', 'SL', '落袋档', '账上权益', '累计提出', '总资产', '提出次数'))
L.append('-' * 130)
targets = [
    ('r1<-1%', 0.03, 0.01),
    ('r5<-2%', 0.03, 0.01),
    ('r5<-2%', 0.02, 0.01),
]
for (sig, tp, sl) in targets:
    sub = R[(R['sig'] == sig) & (np.isclose(R['tp'], tp)) & (np.isclose(R['sl'], sl))
            & (R['pool'] == 20) & (R['stop'] == 0.0)].sort_values('cashout')
    for _, x in sub.iterrows():
        L.append('%-12s %5.1f%% %5.1f%% %8s | %14.2f %14.2f %14.2f %8d' % (
            sig, tp * 100, sl * 100,
            ('%.0fx' % x['cashout']) if x['cashout'] > 1.0 else '不提',
            x['final'], x['withdrawn'], x['total'], int(x['n_cashout'])))
    L.append('')

# ---- 2. 落袋档对终值的影响（全局）----
L.append('--- 二、四档落袋的全局对比（每种落袋档下，按总资产排前 15；提=只提一次）---')
for co in [0.0, 2.0, 3.0, 5.0]:
    sub = R[np.isclose(R['cashout'], co)]
    if len(sub) == 0:
        continue
    top = sub.sort_values('total', ascending=False).head(15)
    L.append('')
    L.append('【%s】' % (('到 %.0fx 提走' % co) if co > 1.0 else '不提，全额复利'))
    L.append('%-10s %6s %6s %5s %8s %14s %12s %12s %8s' % (
        '信号', 'TP', 'SL', '池N', '停手', '账上权益', '累计提出', '总资产', '回撤'))
    L.append('-' * 130)
    for _, x in top.iterrows():
        L.append('%-10s %5.1f%% %5.1f%% %5d %8s %14.2f %12.2f %12.2f %7.2f%%' % (
            x['sig'], x['tp'] * 100, x['sl'] * 100, x['pool'],
            ('%.2fx' % x['stop']) if x['stop'] > 1.0 else '不停',
            x['final'], x['withdrawn'], x['total'], x['mdd'] * 100))

# ---- 3. 落袋到底削不削收益 ----
L.append('')
L.append('--- 三、落袋的代价：同一组配置下，提走钱等于砍掉后续复利（只提一次）---')
L.append('%-10s %6s %6s %5s | %14s %14s %14s' % (
    '信号', 'TP', 'SL', '池N', '不提(总资产)', '到2x提(总资产)', '到5x提(总资产)'))
L.append('-' * 130)
rows3 = []
for (sig, tp, sl, pool) in R[['sig', 'tp', 'sl', 'pool']].drop_duplicates().itertuples(index=False):
    g = R[(R['sig'] == sig) & (np.isclose(R['tp'], tp)) & (np.isclose(R['sl'], sl))
          & (R['pool'] == pool) & (R['stop'] == 0.0)]
    if len(g) < 4:
        continue
    v = {float(r['cashout']): float(r['total']) for _, r in g.iterrows()}
    if 0.0 not in v or 2.0 not in v or 5.0 not in v:
        continue
    rows3.append((sig, tp, sl, pool, v[0.0], v[2.0], v[5.0]))
rows3.sort(key=lambda x: -x[4])
for (sig, tp, sl, pool, v0, v2, v5) in rows3[:20]:
    L.append('%-10s %5.1f%% %5.1f%% %5d | %14.2f %14.2f %14.2f' % (
        sig, tp * 100, sl * 100, pool, v0, v2, v5))

L.append('')
L.append('--- 四、落袋真正保住了什么 ---')
L.append('落袋不改最大回撤（回撤是在落袋之前发生的），它改的是「最坏情况的绝对金额」。')
ine = R[(np.isclose(R['cashout'], 2.0)) & (R['busted'] == 0)]
worst = ine.nsmallest(5, 'total')[['sig', 'tp', 'sl', 'pool', 'stop', 'total', 'withdrawn', 'mdd']]
L.append('提 2x 后，总资产最差的 5 组（即"跑完全程还亏"的极端情形）：')
for _, x in worst.iterrows():
    L.append('  %-10s TP%.1f%%/SL%.1f%% 池%d 停%s → 总资产 %.2f 元（已提出 %.2f，回撤 %.1f%%）' % (
        x['sig'], x['tp'] * 100, x['sl'] * 100, x['pool'],
        ('%.2fx' % x['stop']) if x['stop'] > 1.0 else '不停',
        x['total'], x['withdrawn'], x['mdd'] * 100))
L.append('')
L.append('注：即使 total 归零，你输掉的也只有最初 1000 元——已经提出的现金不受影响。')
L.append('    但要拿到那笔提出，必须先有一组行情让权益翻到 2x；翻不上去就一次也提不到。')

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘价确认触线，不用 high/low；实盘还有排队与滑点。')
L.append('2. 提出动作假设即时按账上权益全额提出，无手续费、无提币延迟、无税务。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实回撤更深。')
L.append('4. 已下架归零币不在池内（幸存者偏差）。')
L.append('5. 极端终值（>1e10）是 9 年复利 + 零滑点的数学产物，不可当预测。')

out = os.path.join(EVI, 'cashout_analysis.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
