# -*- coding: utf-8 -*-
"""daily_weekly.py — 把收益拆到「天」和「周」，看真实节奏。

对池 2/4/10/20（15m跌5% + TP3%/SL1%，成本 0.25%），输出：
  日收益：均值/中位/正日占比/最好/最差/分位、最长连亏
  周收益：同上
输出：evidence/daily_weekly.txt
"""
import os, pickle
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
INIT = 1000.0
BASE, TARGET_COST = 0.0015, 0.0025
D = TARGET_COST - BASE
DAYMS = 86400000

POOLS = [2, 4, 10, 20]
SIG, TP, SL = 'r15<-5%', 0.03, 0.01

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded %d groups' % len(EV), flush=True)

ev0 = EV[(SIG, TP, SL)]
EVS = sorted([(a, b, r - D) for (a, b, r) in ev0])


def simulate(pool):
    eq = INIT
    pos = []
    day_eq = {}          # day_index -> equity at end of that day
    cur_day = None
    day_last_eq = eq
    t0 = EVS[0][0]
    taken = 0
    peak = eq
    mdd = 0.0
    for (t_in, t_out, r) in EVS:
        while pos and pos[0][0] <= t_in:
            to, stake, rr = heapq.heappop(pos)
            eq += stake * rr
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
        d = int((t_in - t0) // DAYMS)
        if cur_day is None:
            cur_day = d
        while cur_day < d:
            day_eq[cur_day] = eq
            cur_day += 1
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
    day_eq[cur_day] = eq
    return day_eq, taken, mdd


def stats(rets, label):
    a = np.array(rets, dtype=np.float64)
    if len(a) == 0:
        return ['  %s: 无数据' % label]
    pos_rate = float((a > 0).mean())
    # 最长连亏
    worst_run = 0
    run = 0
    for x in a:
        if x < 0:
            run += 1
            worst_run = max(worst_run, run)
        else:
            run = 0
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return [
        '  %-6s n=%5d  均值 %+7.3f%%  中位 %+7.3f%%  正占比 %5.1f%%' % (
            label, len(a), a.mean() * 100, q[3] * 100, pos_rate * 100),
        '         最好 %+9.2f%%  最差 %+9.2f%%  最长连亏 %d 个' % (
            a.max() * 100, a.min() * 100, worst_run),
        '         分位 p1 %+7.2f%%  p5 %+7.2f%%  p25 %+7.2f%%  p75 %+7.2f%%  p95 %+7.2f%%  p99 %+7.2f%%' % (
            q[0] * 100, q[1] * 100, q[2] * 100, q[4] * 100, q[5] * 100, q[6] * 100),
    ]


L = []
L.append('收益拆到天/周：15m跌5% + TP3%/SL1%，成本 0.25% 往返')
L.append('=' * 124)
L.append('初始本金 1000 元｜日=UTC 自然日（以首笔为第 0 天）｜周=连续 7 个日索引')
L.append('日/周收益 = 期末权益 / 上期末权益 − 1，只计已平仓盈亏')
L.append('')

for pool in POOLS:
    day_eq, taken, mdd = simulate(pool)
    days = sorted(day_eq.keys())
    eqs = [day_eq[d] for d in days]
    daily = [eqs[i] / eqs[i - 1] - 1.0 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
    weekly = []
    for i in range(7, len(eqs)):
        if eqs[i - 7] > 0:
            weekly.append(eqs[i] / eqs[i - 7] - 1.0)

    fin = eqs[-1] if eqs else INIT
    ny = max(len(eqs), 1) / 365.0
    cagr = (fin / INIT) ** (1.0 / ny) - 1 if fin > 0 and ny > 0 else -1

    L.append('=' * 124)
    L.append('池 N=%d ｜终值 %.2f 元｜%d 个交易日｜年化 %.2f%%｜最大回撤 %.2f%%｜%d 笔' % (
        pool, fin, len(eqs), cagr * 100, mdd * 100, taken))
    L.append('  【天】')
    L.extend(stats(daily, '日'))
    L.append('  【周】')
    L.extend(stats(weekly, '周'))
    L.append('')
    # 第一年（前 365 个交易日）
    if len(eqs) > 365:
        y1 = eqs[365] / INIT - 1
        d1 = [eqs[i] / eqs[i - 1] - 1.0 for i in range(1, 366) if eqs[i - 1] > 0]
        L.append('  第一年：%.2f → %.2f 元（%+.2f%%）｜日均 %+.4f%%｜最好日 %+.2f%%｜最差日 %+.2f%%' % (
            INIT, eqs[365], y1 * 100,
            np.mean(d1) * 100 if d1 else 0,
            max(d1) * 100 if d1 else 0,
            min(d1) * 100 if d1 else 0))
    L.append('')

L.append('=' * 124)
L.append('--- 目标对照：1000 元 → 100 万 = 1000 倍/年 ---')
L.append('  折算：日均需 +1.911%（365 天复利）；或日均 +0.272%%（若按 250 个交易日复利）')
L.append('  上表日均最高的池子是多少？看【天】的"均值"那一行。')
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 成本 0.25% 是线性近似（maker 费 + 0.1% 往返滑点），急跌时真实滑点更差。')
L.append('2. 成交率完全未模拟：急跌挂 maker 单可能不成交。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实日/周波动更大（浮亏未体现）。')
L.append('4. 日收益按 UTC 日切，日内多次进出被压成一个数。')
L.append('5. 已下架归零币不在池内（幸存者偏差）。')

out = os.path.join(EVI, 'daily_weekly.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
