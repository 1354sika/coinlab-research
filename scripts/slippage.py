# -*- coding: utf-8 -*-
"""slippage.py — 滑点/手续费敏感性：这套算法到底靠不靠成本假设。

事件缓存里的 r 已扣 0.15% 往返。成本是线性可调的：
  r_new = r_old - (c_new - 0.0015)
对 TP/SL 触发的行（r = tp - COST / -sl - COST）和持满行（r = seg[-1] - COST）都成立。

输出：evidence/slippage_sensitivity.txt
"""
import os, sys, pickle, time
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
BASE_COST = 0.0015
INIT = 1000.0
DAYMS = 86400000

COSTS = [
    (0.0015, '0.15%  假设值（maker 0.02%/侧 ×2，零滑点）'),
    (0.0025, '0.25%  maker 费 + 0.1% 往返滑点'),
    (0.0035, '0.35%  taker 0.075%/侧 + 0.1% 往返滑点'),
    (0.0055, '0.55%  taker + 0.2%/侧 滑点（急跌追价）'),
]

POOLS = [1, 3, 10, 20, 50]

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded %d groups' % len(EV), flush=True)


def simulate(ev, pool):
    ev = sorted(ev)
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []
    taken = 0
    days = set()
    for (t_in, t_out, r) in ev:
        while pos and pos[0][0] <= t_in:
            to, stake, rr = heapq.heappop(pos)
            eq += stake * rr
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
        days.add(t_in // DAYMS)
        if len(pos) >= pool:
            continue
        heapq.heappush(pos, (t_out, eq / pool, r))
        taken += 1
    while pos:
        to, stake, rr = heapq.heappop(pos)
        eq += stake * rr
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
    return eq, mdd, taken, max(len(days), 1)


L = []
L.append('滑点 / 手续费敏感性分析')
L.append('=' * 132)
L.append('同一批信号与出场规则（收盘价确认触线），只改成本假设。事件缓存里 r 已扣 0.15%%，按差量线性调整。')
L.append('池 N 固定、不停手、不落袋。初始本金 1000 元。')
L.append('')

for c, note in COSTS:
    d = c - BASE_COST
    L.append('=' * 132)
    L.append('成本 %.2f%% 往返 —— %s' % (c * 100, note))
    L.append('%-12s %6s %6s %6s | %9s %11s %12s %10s %8s' % (
        '信号', 'TP', 'SL', '池N', '笔数', '单笔均值', '终值(元)', '最大回撤', '笔/天'))
    L.append('-' * 132)
    best = []
    for key in sorted(EV.keys()):
        sig, tp, sl = key
        ev0 = EV[key]
        if not ev0:
            continue
        ev = [(a, b, r - d) for (a, b, r) in ev0]
        mean_r = float(np.mean([x[2] for x in ev]))
        for pool in POOLS:
            fin, mdd, taken, nd = simulate(ev, pool)
            L.append('%-12s %5.1f%% %5.1f%% %6d | %9d %10.4f%% %12.2f %9.2f%% %8.2f' % (
                sig, tp * 100, sl * 100, pool, taken, mean_r * 100, fin, mdd * 100, taken / nd))
            best.append((mean_r, fin, sig, tp, sl, pool, mdd))
    L.append('')
    pos_cnt = sum(1 for b in best if b[0] > 0)
    L.append('  → 单笔均值为正的组合：%d / %d' % (pos_cnt, len(best)))
    if best:
        bb = max(best, key=lambda x: x[0])
        L.append('  → 单笔均值最高：%s TP%.1f%%/SL%.1f%% 池%d  单笔 %+.4f%%  终值 %.2f  回撤 %.2f%%' % (
            bb[2], bb[3] * 100, bb[4] * 100, bb[5], bb[0] * 100, bb[1], bb[6] * 100))
    L.append('')

L.append('=' * 132)
L.append('--- 诚实边界 ---')
L.append('1. 滑点是按"每笔往返固定加 d"线性处理，真实滑点与单笔金额、盘口深度、下单时刻相关，急跌时更差。')
L.append('2. TP/SL 仍然假设"收盘价确认后能在下一根收盘成交"；真实市价单在急跌时可能被打穿。')
L.append('3. 挂单（maker）能拿到低费率，但急跌时买单挂在下方可能根本不成交——成交率完全未模拟。')
L.append('4. 未含资金费率、最小下单量、API 限频、交易所风控。')
L.append('5. 已下架归零币不在池内（幸存者偏差）。')

out = os.path.join(EVI, 'slippage_sensitivity.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
