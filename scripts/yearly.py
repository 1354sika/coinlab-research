# -*- coding: utf-8 -*-
"""yearly.py — 现实成本档（0.25% 往返）下，本金 1000 元逐年赚多少。

读 combo_events.pkl（r 已扣 0.15%），按 r - 0.001 折成 0.25% 成本。
年份用「从第一笔事件起每 365 天」切，避免早期币少导致自然年失真。
输出：evidence/yearly_025.txt
"""
import os, pickle
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
BASE, TARGET = 0.0015, 0.0025
D = TARGET - BASE
INIT = 1000.0
YEARMS = 365 * 86400000

# (标签, 信号, TP, SL, 池N)
COMBOS = [
    ('15m跌5% TP3/SL1 池3',  'r15<-5%', 0.03, 0.01, 3),
    ('15m跌5% TP3/SL1 池10', 'r15<-5%', 0.03, 0.01, 10),
    ('15m跌5% TP3/SL1 池20', 'r15<-5%', 0.03, 0.01, 20),
    ('15m跌5% TP3/SL1 池50', 'r15<-5%', 0.03, 0.01, 50),
    ('5m跌2%  TP3/SL1 池10', 'r5<-2%',  0.03, 0.01, 10),
]

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded %d groups' % len(EV), flush=True)


def run(ev, pool):
    """返回 (逐年equity列表, 年标签起点ms, 总终值, 回撤)"""
    ev = sorted(ev)
    if not ev:
        return [], 0, INIT, 0.0
    t0 = ev[0][0]
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []
    curves = []
    year_idx = 0
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
        while year_idx < y:
            curves.append(eq)
            year_idx += 1
        if len(pos) >= pool:
            continue
        heapq.heappush(pos, (t_out, eq / pool, r))
    while pos:
        to, stake, rr = heapq.heappop(pos)
        eq += stake * rr
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
    curves.append(eq)
    return curves, t0, eq, mdd


L = []
L.append('现实成本档 0.25%% 往返 · 本金 1000 元 · 逐年赚多少')
L.append('=' * 118)
L.append('数据：E 盘 1m 全量（318 币，2017-07 ~ 2026-10）｜出场=收盘价确认触线｜满仓新信号丢弃')
L.append('年份 = 从第一笔事件起每 365 天一年')
L.append('')

allcurves = {}
for (lab, sig, tp, sl, pool) in COMBOS:
    ev0 = EV.get((sig, tp, sl))
    if not ev0:
        L.append('缺 %s' % lab)
        continue
    ev = [(a, b, r - D) for (a, b, r) in ev0]
    curves, t0, fin, mdd = run(ev, pool)
    allcurves[lab] = curves
    ny = len(curves)
    L.append('-' * 118)
    L.append('%s   首笔 %s   共 %d 年' % (
        lab,
        __import__('pandas').to_datetime(t0, unit='ms').strftime('%Y-%m-%d'),
        ny))
    L.append('%-8s %14s %12s %12s %14s' % ('年', '年末资金', '本年收益', '本年收益率', '累计倍数'))
    L.append('-' * 118)
    prev = INIT
    for i, v in enumerate(curves):
        L.append('%-8s %14.2f %12.2f %11.2f%% %13.2fx' % (
            'Y%d' % (i + 1), v, v - prev, (v / prev - 1) * 100, v / INIT))
        prev = v
    yrs = max(ny, 1)
    cagr = (fin / INIT) ** (1.0 / yrs) - 1 if fin > 0 else -1
    L.append('')
    L.append('  9 年终值 %.2f 元｜年均复合 %.2f%%｜最大回撤 %.2f%%' % (fin, cagr * 100, mdd * 100))
    if curves:
        L.append('  **第一年末 %.2f 元（第一年赚 %.2f 元，%.2f%%）**' % (
            curves[0], curves[0] - INIT, (curves[0] / INIT - 1) * 100))
    L.append('')

L.append('=' * 118)
L.append('--- 逐年对照（各组合第一年末资金）---')
for lab, c in allcurves.items():
    if c:
        L.append('  %-24s 第一年末 %10.2f 元（赚 %+9.2f）' % (lab, c[0], c[0] - INIT))

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 成本 0.25% 往返是线性近似（maker 费 + 0.1% 往返滑点），滑点与金额/盘口强相关，急跌时更差。')
L.append('2. 成交率完全未模拟：急跌挂 maker 单可能不成交，真实信号数与收益都会变。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实回撤更深。')
L.append('4. 早期年份（2017-2018）参与统计的币很少，Y1 的样本与后期不可比。')
L.append('5. 已下架归零币不在池内（幸存者偏差）。')
L.append('6. 逐年资金曲线是把当年最后一笔平仓后的权益记到年末，年内波动未体现。')

out = os.path.join(EVI, 'yearly_025.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
