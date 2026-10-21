# -*- coding: utf-8 -*-
"""final_board.py — 扣掉手续费与滑点后，还活着的策略榜单。

成本档：
  0.25% = maker 0.02%/侧 ×2 + 0.1% 往返滑点（乐观但可能）
  0.35% = taker 0.075%/侧 + 0.1% 往返滑点（现实偏保守）
三族信号 × 池 N 1~50 全扫，按 Calmar（年化/|回撤|）与年化分别排序。
输出：evidence/final_board.txt
"""
import os, pickle
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
INIT = 1000.0
BASE = 0.0015
DAYMS = 86400000
YEARMS = 365 * DAYMS

COSTS = [(0.0025, '0.25%（maker 费 + 0.1% 往返滑点）'),
         (0.0035, '0.35%（taker 费 + 0.1% 往返滑点）')]
FAMILIES = [('15m跌5%', 'r15<-5%', 0.03, 0.01),
            ('5m跌2%',  'r5<-2%',  0.03, 0.01),
            ('1m跌1%',  'r1<-1%',  0.03, 0.01)]
NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded %d groups' % len(EV), flush=True)


def run(ev, pool):
    ev = sorted(ev)
    if not ev:
        return None
    t0 = ev[0][0]
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []
    taken = 0
    cur_y = 0
    years = []
    for (t_in, t_out, r) in ev:
        while pos and pos[0][0] <= t_in:
            to, stake, rr = heapq.heappop(pos)
            eq += stake * rr
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
        y = int((t_in - t0) // YEARMS)
        while cur_y < y:
            years.append(eq)
            cur_y += 1
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
    ny = max(cur_y + 1, 1)
    cagr = (eq / INIT) ** (1.0 / ny) - 1 if eq > 0 else -1.0
    return dict(final=eq, mdd=mdd, cagr=cagr, taken=taken, ny=ny, per_year=taken / ny)


L = []
L.append('扣掉手续费与滑点后，还活着的策略榜单')
L.append('=' * 132)
L.append('数据：E 盘 1m 全量（318 币，2017-08 ~ 2026-10）｜出场=收盘价确认触线｜满仓新信号丢弃')
L.append('初始本金 1000 元｜策略族固定为 TP3%/SL1%（前轮扫描出的最优盈亏比）')
L.append('')

for cost, note in COSTS:
    d = cost - BASE
    L.append('=' * 132)
    L.append('成本 %.2f%% 往返 —— %s' % (cost * 100, note))
    L.append('')
    rows = []
    for (fam, sig, tp, sl) in FAMILIES:
        ev0 = EV.get((sig, tp, sl))
        if not ev0:
            continue
        ev = [(a, b, r - d) for (a, b, r) in ev0]
        for n in NS:
            s = run(ev, n)
            if not s:
                continue
            rows.append(dict(fam=fam, n=n, **s))

    alive = [x for x in rows if x['cagr'] > 0]
    L.append('  组合总数 %d｜年化为正 %d 个' % (len(rows), len(alive)))
    L.append('')
    L.append('  【按年化排序 前 12】')
    L.append('  %-9s %5s %13s %9s %9s %8s %9s' % (
        '信号族', '池N', '终值(元)', '年均复合', '最大回撤', '笔/年', 'Calmar'))
    L.append('  ' + '-' * 128)
    for x in sorted(alive, key=lambda v: -v['cagr'])[:12]:
        calmar = x['cagr'] / abs(x['mdd']) if x['mdd'] < 0 else float('inf')
        L.append('  %-9s %5d %13.2f %8.2f%% %8.2f%% %8.0f %9.2f' % (
            x['fam'], x['n'], x['final'], x['cagr'] * 100, x['mdd'] * 100,
            x['per_year'], calmar))

    L.append('')
    L.append('  【按 Calmar（年化/回撤）排序 前 12】—— 每单位回撤换多少年化')
    L.append('  %-9s %5s %13s %9s %9s %8s %9s' % (
        '信号族', '池N', '终值(元)', '年均复合', '最大回撤', '笔/年', 'Calmar'))
    L.append('  ' + '-' * 128)
    for x in sorted(alive, key=lambda v: -(v['cagr'] / abs(v['mdd']) if v['mdd'] < 0 else 1e9))[:12]:
        calmar = x['cagr'] / abs(x['mdd']) if x['mdd'] < 0 else float('inf')
        L.append('  %-9s %5d %13.2f %8.2f%% %8.2f%% %8.0f %9.2f' % (
            x['fam'], x['n'], x['final'], x['cagr'] * 100, x['mdd'] * 100,
            x['per_year'], calmar))

    L.append('')
    L.append('  【按回撤从小到大 前 8】—— 最坐得住的那几档')
    L.append('  %-9s %5s %13s %9s %9s %8s' % (
        '信号族', '池N', '终值(元)', '年均复合', '最大回撤', '笔/年'))
    L.append('  ' + '-' * 128)
    for x in sorted(alive, key=lambda v: -v['mdd'])[:8]:
        L.append('  %-9s %5d %13.2f %8.2f%% %8.2f%% %8.0f' % (
            x['fam'], x['n'], x['final'], x['cagr'] * 100, x['mdd'] * 100, x['per_year']))
    L.append('')

L.append('=' * 132)
L.append('--- 怎么读 ---')
L.append('1. 年化排序 = 谁赚得最凶（代价通常是最深的回撤）。')
L.append('2. Calmar 排序 = 每扛 1% 回撤换到多少年化，是"性价比"排序，最适合长期心态。')
L.append('3. 回撤排序 = 最不容易中途放弃的那几档。')
L.append('4. 三族里 15m跌5% 的笔数最少（约 3,000/年），1m跌1% 最多（约 5,800/年），后者对成本更敏感。')
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 成本 0.25%/0.35% 都是线性近似（费 + 固定滑点），真实滑点与金额/盘口/时刻强相关，急跌时更差。')
L.append('2. 成交率完全未模拟：急跌挂 maker 单可能不成交，真实交易笔数会更少。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实回撤更深。')
L.append('4. 已下架归零币不在池内（幸存者偏差）。')
L.append('5. 未模拟最小下单量：池 20 + 1000 元时每笔 50 元，接近币安 5 USDT 下限；资金再小就不可行。')
L.append('6. 全部为回测，零实盘验证。')

out = os.path.join(EVI, 'final_board.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
