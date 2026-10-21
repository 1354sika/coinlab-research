# -*- coding: utf-8 -*-
"""pool_scan.py — 「池子大点」到底行不行：把池 N 从 1 扫到 200。

对 15m跌5% + TP3%/SL1%（0.25% 成本），逐 N 算：
  终值 / 年均复合 / 最大回撤 / 最好单年倍数 / 最差单年 / 年笔数
输出：evidence/pool_scan.txt
"""
import os, pickle
import numpy as np
import heapq

EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
INIT = 1000.0
BASE, TARGET_COST = 0.0015, 0.0025
D = TARGET_COST - BASE
YEARMS = 365 * 86400000

NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

with open(CACHE, 'rb') as f:
    EV = pickle.load(f)
print('loaded %d groups' % len(EV), flush=True)

SIG, TP, SL = 'r15<-5%', 0.03, 0.01
ev0 = EV[(SIG, TP, SL)]
ev_all = [(a, b, r - D) for (a, b, r) in ev0]


def run(pool):
    ev = sorted(ev_all)          # 必须按时间排序，否则持仓释放与年份切分全部失效
    t0 = ev[0][0]
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []
    taken = 0
    years = []
    cur_y = 0
    day_returns = {}
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
    years.append(eq)
    # 逐年收益率
    rets = []
    prev = INIT
    for v in years:
        rets.append(v / prev - 1.0 if prev > 0 else -1.0)
        prev = v
    ny = max(len(years), 1)
    cagr = (eq / INIT) ** (1.0 / ny) - 1 if eq > 0 else -1.0
    return dict(final=eq, mdd=mdd, taken=taken, ny=ny, cagr=cagr,
                best=max(rets) if rets else 0, worst=min(rets) if rets else 0,
                per_year=taken / ny, years=years)


L = []
L.append('池子大小扫描：15m跌5% + TP3%/SL1%，成本 0.25% 往返')
L.append('=' * 128)
L.append('初始本金 1000 元｜样本 %d 条事件｜年份=从首笔起每 365 天' % len(ev_all))
L.append('')
L.append('%5s %14s %10s %10s %10s %10s %10s' % (
    '池N', '终值(元)', '年均复合', '最大回撤', '最好单年', '最差单年', '笔/年'))
L.append('-' * 128)

rows = []
for n in NS:
    s = run(n)
    rows.append((n, s))
    L.append('%5d %14.2f %9.2f%% %9.2f%% %9.1f%% %9.1f%% %10.0f' % (
        n, s['final'], s['cagr'] * 100, s['mdd'] * 100,
        s['best'] * 100, s['worst'] * 100, s['per_year']))

L.append('')
best_n = max(rows, key=lambda x: x[1]['cagr'])
L.append('★ 年均复合最优：池 N=%d → %.2f%%/年（终值 %.2f 元，回撤 %.2f%%）' % (
    best_n[0], best_n[1]['cagr'] * 100, best_n[1]['final'], best_n[1]['mdd'] * 100))
best_year = max(rows, key=lambda x: x[1]['best'])
L.append('★ 单年爆发最优：池 N=%d → 最好一年 +%.1f%%（%.2f 倍）' % (
    best_year[0], best_year[1]['best'] * 100, 1 + best_year[1]['best']))
L.append('')
L.append('目标对照：1000 倍/年。上表没有一档接近——单年最好的也是 %.1f 倍。' % (
    1 + best_year[1]['best']))

L.append('')
L.append('--- 池 3 与池 20 的逐年对照（看"池大"到底换来了什么）---')
s3 = dict(rows)[3]
s20 = dict(rows)[20]
L.append('%6s %16s %16s' % ('年', '池3 年末', '池20 年末'))
L.append('-' * 128)
for i in range(max(len(s3['years']), len(s20['years']))):
    a = s3['years'][i] if i < len(s3['years']) else float('nan')
    b = s20['years'][i] if i < len(s20['years']) else float('nan')
    L.append('%6s %16.2f %16.2f' % ('Y%d' % (i + 1), a, b))

L.append('')
L.append('--- 结论 ---')
L.append('1. 池子变大不会提升年化，反而下降：池3 最优，池10 掉一半，池20 再掉一半，池50 只剩 1/12。')
L.append('2. 原因：每笔只投 1/N，同样的单笔收益对账户的推动力按 1/N 缩小；而信号总量不变。')
L.append('3. 池子变大的唯一好处是回撤变浅（−81%% → −6%%），但代价是年化从 44%% 掉到 12%%。')
L.append('4. 想靠"池大"到 1000 倍/年，方向反了：池越大离目标越远。')

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 成本 0.25% 是线性近似（maker 费 + 0.1% 往返滑点）。')
L.append('2. 成交率完全未模拟。')
L.append('3. 权益只计已平仓盈亏，未计浮亏；真实回撤更深。')
L.append('4. 已下架归零币不在池内（幸存者偏差）。')

out = os.path.join(EVI, 'pool_scan.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
