# -*- coding: utf-8 -*-
"""target_1e6.py — 「1000 元一年翻到 100 万」在实测数据上到底差多远。

目标：1000 → 1,000,000 = 1000 倍/年 = 每天 +1.907%（365 天复利）
      或 (1+x)^365 = 1000  =>  x = 1000^(1/365) - 1 = 1.907%/天

关键：复利看的是**几何均值**，不是算术均值。亏损笔虽然少但幅度大，
      会大幅拉低几何均值——这是"数学上能到、实测到不了"的根因。

输出：evidence/target_1e6.txt
"""
import os, pickle
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
INIT = 1000.0
TARGET = 1_000_000.0
YEARMS = 365 * 86400000
BASE, TARGET_COST = 0.0015, 0.0025
D = TARGET_COST - BASE

COMBOS = [
    ('15m跌5% TP3/SL1 池1',  'r15<-5%', 0.03, 0.01, 1),
    ('15m跌5% TP3/SL1 池3',  'r15<-5%', 0.03, 0.01, 3),
    ('15m跌5% TP3/SL1 池10', 'r15<-5%', 0.03, 0.01, 10),
    ('15m跌5% TP3/SL1 池20', 'r15<-5%', 0.03, 0.01, 20),
    ('5m跌2%  TP3/SL1 池1',  'r5<-2%',  0.03, 0.01, 1),
    ('5m跌2%  TP3/SL1 池3',  'r5<-2%',  0.03, 0.01, 3),
    ('1m跌1%  TP3/SL1 池1',  'r1<-1%',  0.03, 0.01, 1),
]

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded' % (), flush=True) if False else print('loaded %d groups' % len(EV), flush=True)


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
        days.add(t_in // YEARMS * 0 + t_in // 86400000)
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
L.append('「1000 元一年翻到 100 万」可行性核算')
L.append('=' * 126)
L.append('目标：1000 → 1,000,000 = **1000 倍/年**')
L.append('折算：需要每天 +%.3f%%（365 天复利），或每 2 笔各 +%.3f%%（一天 2 笔）' % (
    (TARGET / INIT) ** (1 / 365) * 100 - 100,
    ((TARGET / INIT) ** (1 / 365)) ** 0.5 * 100 - 100))
L.append('成本档：0.25% 往返（现实档）｜出场=收盘价确认触线')
L.append('')
L.append('**复利看几何均值，不看算术均值。下面两列都算，差异就是"数学陷阱"。**')
L.append('')
L.append('%-22s %8s %10s %10s %8s %9s %9s %14s %12s' % (
    '组合', '笔/年', '算术均值', '几何均值', '胜率', '赢均', '输均', '理论年倍数', '实测年倍数'))
L.append('-' * 126)

rows = []
for (lab, sig, tp, sl, pool) in COMBOS:
    ev0 = EV.get((sig, tp, sl))
    if not ev0:
        continue
    ev = [(a, b, r - D) for (a, b, r) in ev0]
    rs = np.array([x[2] for x in ev], dtype=np.float64)
    fin, mdd, taken, ndays = simulate(ev, pool)
    am = float(rs.mean())
    # 几何均值：E[ln(1+r)] -> expm1
    gm = float(np.expm1(np.mean(np.log1p(np.clip(rs, -0.999, None)))))
    win = float((rs > 0).mean())
    wm = float(rs[rs > 0].mean()) if (rs > 0).any() else 0.0
    lm = float(rs[rs <= 0].mean()) if (rs <= 0).any() else 0.0
    years = ndays / 365.0
    per_year = taken / years if years > 0 else 0
    theo = (1 + gm / pool) ** per_year if gm > -1 else 0.0
    real = (fin / INIT) ** (1 / years) if fin > 0 and years > 0 else 0.0
    L.append('%-22s %8.0f %9.4f%% %9.4f%% %7.1f%% %8.3f%% %8.3f%% %14.3e %12.3fx' % (
        lab, per_year, am * 100, gm * 100, win * 100, wm * 100, lm * 100, theo, real))
    rows.append(dict(lab=lab, per_year=per_year, am=am, gm=gm, win=win, wm=wm, lm=lm,
                     theo=theo, real=real, fin=fin, mdd=mdd, pool=pool, taken=taken))

L.append('')
L.append('--- 差在哪里：算术均值 vs 几何均值 ---')
for x in rows:
    gap = (x['am'] - x['gm']) * 100
    L.append('  %-22s 算术 %+.4f%%  几何 %+.4f%%  差 %.4f 个百分点  （胜率 %.1f%%, 赢 %.2f%% / 输 %.2f%%）' % (
        x['lab'], x['am'] * 100, x['gm'] * 100, gap, x['win'] * 100, x['wm'] * 100, x['lm'] * 100))

L.append('')
L.append('--- 要到 1000 倍/年，需要什么 ---')
L.append('公式：(1 + 几何均值/池) ^ (每年笔数) = 1000')
for x in rows:
    if x['gm'] <= 0:
        L.append('  %-22s 几何均值为负/零 → 复利必然衰减，永久达不到' % x['lab'])
        continue
    need_gm = 1000 ** (1 / x['per_year']) * x['pool'] - x['pool']
    L.append('  %-22s 现有 %.4f%%/笔 × %.0f 笔/年 → 实测 %.2fx/年；要 1000x 需每笔 %.4f%%（差 %.1f 倍）' % (
        x['lab'], x['gm'] * 100, x['per_year'], x['real'], need_gm * 100,
        need_gm / x['gm'] if x['gm'] > 0 else float('inf')))

L.append('')
L.append('--- 池 1 的悖论：数学上最猛，路径上先死 ---')
for x in rows:
    if x['pool'] != 1:
        continue
    L.append('  %-22s 几何 %+.4f%%/笔 × %.0f 笔/年 → 理论上 %.3e 倍/年' % (
        x['lab'], x['gm'] * 100, x['per_year'], x['theo']))
    L.append('  %-22s 实测终值 %.2f 元（回撤 %.2f%%）' % ('', x['fin'], x['mdd'] * 100))
    if x['fin'] < INIT:
        L.append('  %-22s ← 理论上是天文数字，实测却亏光：**中途账户被打到接近 0，复利断了**' % '')

L.append('')
L.append('--- 结论 ---')
L.append('1. 数学上成立：1000 倍/年 = 每天 +1.907%，或一天两笔各 +0.951%。')
L.append('2. 实测最好：15m跌5%% TP3/SL1 池3 → 年均 %.2fx（不是 1000x，差 %.0f 倍）。' % (
    rows[1]['real'] if len(rows) > 1 else 0, 1000 / (rows[1]['real'] if len(rows) > 1 and rows[1]['real'] > 0 else 1)))
L.append('3. 根因一：几何均值远低于算术均值（亏损笔幅度大，复利对亏损不对称）。')
L.append('4. 根因二：池 1（全仓）理论上最猛，但中途被打到接近 0，复利链断裂。')
L.append('5. 根因三：满仓跳过——信号来的时候手上已有仓位，最肥的那几笔可能没接住。')

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 成本 0.25% 是线性近似（maker 费 + 0.1% 往返滑点），急跌时真实滑点更差。')
L.append('2. 成交率完全未模拟：急跌挂 maker 单可能不成交。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实回撤更深。')
L.append('4. 已下架归零币不在池内（幸存者偏差）。')
L.append('5. 理论年倍数按"几何均值/pool 逐笔复利"，未考虑满仓跳过与资金占用，是上界。')

out = os.path.join(EVI, 'target_1e6.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
