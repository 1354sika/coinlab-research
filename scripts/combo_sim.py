# -*- coding: utf-8 -*-
"""combo_sim.py — 急跌买信号的组合层模拟（池并发 + 日停手 + 单账号复利）

一次数据遍历收集全部 12 组（3 信号 × 4 组 TP/SL）事件，再对每组做组合模拟。
事件驱动：同一时间轴上，先结算已到期持仓（用该笔开仓时的 stake），再开新仓。

用法：
  python -X utf8 combo_sim.py [LIMIT_SYMS]
    LIMIT_SYMS=0 表示全部币

出口：
  evidence/combo_sim.txt
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict

D = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
COST = 0.0015          # 往返成本
MAXH = 240             # 最长持 4 小时
GAP = 60               # 同币去重：相邻信号间隔 >= 60 根
MIN_BARS = 5000
DAYMS = 86400000
POOL = 20
INIT = 1000.0

# 信号：名字 -> (回看根数, 阈值)
SIGS = [
    ('r5<-2%', 5, -0.02),
    ('r1<-1%', 1, -0.01),
    ('r15<-5%', 15, -0.05),
]
# 出场
TSSL = [(0.02, 0.01), (0.03, 0.01), (0.02, 0.02), (0.03, 0.02)]
# 日停手倍数（0 = 不停）
STOPS = [0.0, 1.10, 1.25, 1.50]

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

files = sorted(glob.glob(os.path.join(D, '*.csv')))
if LIMIT:
    files = files[:LIMIT]
print('files %d' % len(files), flush=True)

t0 = time.time()
EV = defaultdict(list)   # (sig, tp, sl) -> [(t_in, t_out, r_net)]
done = 0
for fp in files:
    sym = os.path.basename(fp)[:-4]
    try:
        df = pd.read_csv(fp, usecols=['open_time', 'close'])
    except Exception as e:
        print('SKIP %s %s' % (sym, e), flush=True)
        continue
    if len(df) < MIN_BARS:
        continue
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna()
    if len(df) < MIN_BARS:
        continue
    ot = df['open_time'].to_numpy().astype('int64')
    c = df['close'].to_numpy(np.float64)
    n = len(c)

    def ret(k):
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = c[k:] / c[:-k] - 1.0
        return r

    cache = {}
    for name, k, thr in SIGS:
        r = ret(k)
        m = np.where(np.isfinite(r) & (r < thr))[0]
        # 去掉头部预热与尾部无足够前瞻
        m = m[(m >= 200) & (m < n - MAXH - 1)]
        if len(m) == 0:
            continue
        # 同币去重
        keep = []
        last = -10**9
        for i in m:
            if i - last >= GAP:
                keep.append(int(i)); last = int(i)
        for i in keep:
            seg = c[i + 1:i + 1 + MAXH] / c[i] - 1.0
            if len(seg) == 0:
                continue
            for tp, sl in TSSL:
                hu = np.where(seg >= tp)[0]
                hd = np.where(seg <= -sl)[0]
                ju = int(hu[0]) if len(hu) else 10**9
                jd = int(hd[0]) if len(hd) else 10**9
                j = ju if ju <= jd else jd
                if j == 10**9:
                    r_out = float(seg[-1]) - COST
                    j = len(seg) - 1
                elif ju <= jd:
                    r_out = tp - COST
                else:
                    r_out = -sl - COST
                EV[(name, tp, sl)].append((int(ot[i]), int(ot[i] + (j + 1) * 60000), r_out))
    done += 1
    if done % 20 == 0:
        print('  [%d/%d] %s  事件 %d  %.0fs' % (done, len(files), sym,
              sum(len(v) for v in EV.values()), time.time() - t0), flush=True)

print('collect done  syms %d  事件 %d  %.0fs' % (done, sum(len(v) for v in EV.values()),
      time.time() - t0), flush=True)


def simulate(ev, pool, daystop):
    """事件驱动组合模拟。返回 dict。"""
    ev = sorted(ev)                    # 按开仓时间
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []                           # 最小堆 (t_out, stake, r)
    cur_day = -1
    day_start = eq
    day_stopped = False
    taken = 0
    skip_occ = 0
    skip_stop = 0
    hit_days = 0
    days = set()
    day_ret = {}

    for (t_in, t_out, r) in ev:
        # 结算到期持仓
        while pos and pos[0][0] <= t_in:
            to, stake, rr = heapq.heappop(pos)
            eq += stake * rr
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
        d = t_in // DAYMS
        if d != cur_day:
            cur_day = d
            day_start = eq
            day_stopped = False
        days.add(d)
        if daystop > 1.0 and not day_stopped and eq >= day_start * daystop:
            day_stopped = True
            hit_days += 1
        if day_stopped:
            skip_stop += 1
            continue
        if len(pos) >= pool:
            skip_occ += 1
            continue
        stake = eq / pool
        heapq.heappush(pos, (t_out, stake, r))
        taken += 1

    # 收尾：把剩余持仓按出场时间清掉
    while pos:
        to, stake, rr = heapq.heappop(pos)
        eq += stake * rr
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd

    ndays = max(len(days), 1)
    years = ndays / 365.0
    ann = (eq / INIT) ** (1.0 / years) - 1.0 if eq > 0 else -1.0
    return dict(final=eq, mdd=mdd, taken=taken, skip_occ=skip_occ, skip_stop=skip_stop,
                days=ndays, hit_days=hit_days, ann=ann,
                per_day=taken / ndays)


L = []
L.append('急跌买信号 · 组合层模拟（299 币 E 盘 1m，2017-2026）')
L.append('=' * 120)
L.append('成本 %.2f%% 往返｜最长持 %d 根｜GAP=%d 根｜池 N=%d｜初始本金 %.0f 元' % (
    COST * 100, MAXH, GAP, POOL, INIT))
L.append('出场：收盘价确认触线（不用影线）｜事件驱动：先结算已到期持仓再开新仓')
L.append('')
L.append('样本币 %d｜事件总数 %d｜采集耗时 %.0fs' % (
    done, sum(len(v) for v in EV.values()), time.time() - t0))
L.append('')

for (sig, tp, sl) in sorted(EV.keys()):
    ev = EV[(sig, tp, sl)]
    if not ev:
        continue
    L.append('-' * 120)
    L.append('信号 %s  TP=%.1f%%  SL=%.1f%%   事件 %d 条' % (sig, tp * 100, sl * 100, len(ev)))
    L.append('%-10s %14s %10s %8s %8s %8s %10s' % (
        '日停手', '终值(元)', '最大回撤', '成交笔', '跳占用', '跳停手', '笔/天'))
    for st in STOPS:
        s = simulate(ev, POOL, st)
        L.append('%-10s %14.2f %10.2f%% %8d %8d %8d %10.2f' % (
            ('%.2fx' % st) if st > 1.0 else '不停',
            s['final'], s['mdd'] * 100, s['taken'], s['skip_occ'], s['skip_stop'], s['per_day']))
    L.append('')

L.append('=' * 120)
L.append('--- 诚实边界 ---')
L.append('1. 收盘价确认触线，不用 high/low；实盘还要排队与滑点，成交价只会更差。')
L.append('2. 持仓 stake = 开仓时 equity/POOL；equity 只计已平仓盈亏，未计浮动盈亏。')
L.append('3. 日停手按 UTC 日切；达到目标后当日剩余信号丢弃，次日以新本金继续。')
L.append('4. 满仓时新信号直接丢弃（不做排队），这就是单账号真实约束。')
L.append('5. 同币去重 GAP=60 根；跨币同波未去重（同一波行情可能多币同时开仓）。')
L.append('6. 数据为 Binance 现货 1m；已下架归零币不在池内（幸存者偏差）。')
L.append('7. 未模拟交易所最小下单量、资金费率、API 限频。')

out = os.path.join(EVI, 'combo_sim.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
