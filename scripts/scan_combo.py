# -*- coding: utf-8 -*-
"""scan_combo.py — 急跌买信号 · 全组合扫描（信号 × TP/SL × 池 N × 日停手）

一次采集事件并缓存到 evidence/combo_events.pkl，之后改参数扫描不再重采。

用法：
  python -X utf8 scan_combo.py            # 用缓存（没有就采集）
  python -X utf8 scan_combo.py recompute  # 强制重采

输出：
  evidence/combo_scan_all.csv   （每组合一行，便于排序）
  evidence/combo_scan_all.txt   （分组可读表 + Top 榜）
"""
import os, sys, glob, time, pickle
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict

D = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
CACHE = os.path.join(EVI, 'combo_events.pkl')
COST = 0.0015
MAXH = 240
GAP = 60
MIN_BARS = 5000
DAYMS = 86400000
INIT = 1000.0

SIGS = [('r5<-2%', 5, -0.02), ('r1<-1%', 1, -0.01), ('r15<-5%', 15, -0.05)]
TSSL = [(0.02, 0.01), (0.03, 0.01), (0.02, 0.02), (0.03, 0.02)]
POOLS = [1, 3, 5, 10, 20, 50]
STOPS = [0.0, 1.05, 1.10, 1.25, 1.50]
CASHOUTS = [0.0, 2.0, 3.0, 5.0]   # 0=不提；2.0=首次翻到 2x 提走 1x（只提一次）

force = 'recompute' in sys.argv

if os.path.exists(CACHE) and not force:
    print('load cache %s' % CACHE, flush=True)
    t0 = time.time()
    with open(CACHE, 'rb') as f:
        EV = pickle.load(f)
    print('loaded %d groups  %.1fs' % (len(EV), time.time() - t0), flush=True)
else:
    files = sorted(glob.glob(os.path.join(D, '*.csv')))
    print('collect files %d' % len(files), flush=True)
    t0 = time.time()
    EV = defaultdict(list)
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

        for name, k, thr in SIGS:
            r = ret(k)
            m = np.where(np.isfinite(r) & (r < thr))[0]
            m = m[(m >= 200) & (m < n - MAXH - 1)]
            if len(m) == 0:
                continue
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
        if done % 40 == 0:
            print('  [%d/%d] %s  事件 %d  %.0fs' % (done, len(files), sym,
                  sum(len(v) for v in EV.values()), time.time() - t0), flush=True)
    print('collect done syms %d  ev %d  %.0fs' % (done, sum(len(v) for v in EV.values()),
          time.time() - t0), flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(dict(EV), f, protocol=5)
    print('cached %s  %.0f MB' % (CACHE, os.path.getsize(CACHE) / 1e6), flush=True)


def simulate(ev, pool, daystop, cashout=0.0):
    ev = sorted(ev)
    eq = INIT
    peak = eq
    mdd = 0.0
    pos = []
    cur_day = -1
    day_start = eq
    day_stopped = False
    taken = skip_occ = skip_stop = hit_days = 0
    days = set()
    busted = False
    bust_ms = 0
    withdrawn = 0.0
    n_cashout = 0
    cashed = False          # 只提一次：触发后永不再提
    for (t_in, t_out, r) in ev:
        while pos and pos[0][0] <= t_in:
            to, stake, rr = heapq.heappop(pos)
            eq += stake * rr
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
            if eq <= 0:
                busted = True
                bust_ms = to
                break
            # 落袋：只提一次。到 cashout 倍就把多出部分提走（现金落袋），之后永不再提
            if cashout > 1.0 and not cashed and eq >= INIT * cashout:
                withdrawn += eq - INIT
                eq = INIT
                n_cashout = 1
                cashed = True
        if busted:
            break
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
    while pos and not busted:
        to, stake, rr = heapq.heappop(pos)
        eq += stake * rr
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
        if eq <= 0:
            busted = True
            bust_ms = to
            break
    nd = max(len(days), 1)
    total = (eq if not busted else 0.0) + withdrawn
    return dict(final=eq if not busted else 0.0, mdd=mdd, taken=taken, skip_occ=skip_occ,
                skip_stop=skip_stop, days=nd, hit_days=hit_days, per_day=taken / nd,
                busted=busted, bust_ms=bust_ms, withdrawn=withdrawn, total=total,
                n_cashout=n_cashout)


rows = []
t1 = time.time()
keys = sorted(EV.keys())
for (sig, tp, sl) in keys:
    ev = EV[(sig, tp, sl)]
    if not ev:
        continue
    for pool in POOLS:
        for st in STOPS:
            for co in CASHOUTS:
                s = simulate(ev, pool, st, co)
                rows.append(dict(sig=sig, tp=tp, sl=sl, pool=pool, stop=st, cashout=co,
                                 final=s['final'], mdd=s['mdd'], taken=s['taken'],
                                 skip_occ=s['skip_occ'], skip_stop=s['skip_stop'],
                                 days=s['days'], per_day=s['per_day'],
                                 busted=int(s['busted']), withdrawn=s['withdrawn'],
                                 total=s['total'], n_cashout=s['n_cashout']))
    print('  done %s %.1f/%.1f  %.0fs' % (sig, tp * 100, sl * 100, time.time() - t1), flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(EVI, 'combo_scan_all.csv'), index=False, encoding='utf-8-sig')

L = []
L.append('急跌买信号 · 全组合扫描（%d 组）' % len(R))
L.append('=' * 140)
L.append('数据：E 盘 1m 全量（%d 个币，2017-2026）｜成本 %.2f%% 往返｜最长持 %d 根｜GAP=%d 根' % (
    len(glob.glob(os.path.join(D, '*.csv'))), COST * 100, MAXH, GAP))
L.append('池 N ∈ %s｜日停手 ∈ %s｜初始本金 %.0f 元' % (POOLS, ['不停'] + [('%.2fx' % x) for x in STOPS[1:]], INIT))
L.append('出场：收盘价确认触线｜事件驱动：先结算已到期持仓再开新仓｜满仓新信号丢弃不做排队')
L.append('')

L.append('--- 全部结果（按终值降序，前 40；爆仓=中途归零后收益为数学虚构，标 BUST）---')
L.append('%-10s %6s %6s %6s %8s %16s %10s %9s %9s %8s' % (
    '信号', 'TP', 'SL', '池N', '停手', '终值(元)', '最大回撤', '成交笔', '跳停手', '笔/天'))
L.append('-' * 140)
top = R.sort_values('final', ascending=False).head(40)
for _, x in top.iterrows():
    L.append('%-10s %5.1f%% %5.1f%% %6d %8s %16.2f %9.2f%% %9d %9d %8.2f %s' % (
        x['sig'], x['tp'] * 100, x['sl'] * 100, x['pool'],
        ('%.2fx' % x['stop']) if x['stop'] > 1.0 else '不停',
        x['final'], x['mdd'] * 100, x['taken'], x['skip_stop'], x['per_day'],
        'BUST' if x['busted'] else ''))

L.append('')
L.append('--- 低回撤榜（mdd > -20% 且未爆仓，按终值降序，前 20）---')
L.append('%-10s %6s %6s %6s %8s %16s %10s %9s %8s' % (
    '信号', 'TP', 'SL', '池N', '停手', '终值(元)', '最大回撤', '成交笔', '笔/天'))
L.append('-' * 140)
low = R[(R['mdd'] > -0.20) & (R['busted'] == 0)].sort_values('final', ascending=False).head(20)
if len(low) == 0:
    L.append('  （无：没有任何组合回撤浅于 -20%）')
for _, x in low.iterrows():
    L.append('%-10s %5.1f%% %5.1f%% %6d %8s %16.2f %9.2f%% %9d %8.2f' % (
        x['sig'], x['tp'] * 100, x['sl'] * 100, x['pool'],
        ('%.2fx' % x['stop']) if x['stop'] > 1.0 else '不停',
        x['final'], x['mdd'] * 100, x['taken'], x['per_day']))

L.append('')
L.append('--- 日停手到底起了多少作用 ---')
L.append('%-10s %6s %6s %6s %14s %14s %10s' % (
    '信号', 'TP', 'SL', '池N', '不停终值', '最好停手终值', '停手触发笔'))
L.append('-' * 140)
for key, g in R.groupby(['sig', 'tp', 'sl', 'pool']):
    if len(g) < 2:
        continue
    base = g[g['stop'] == 0.0]
    if len(base) == 0:
        continue
    base_v = float(base.iloc[0]['final'])
    best = g.sort_values('final', ascending=False).iloc[0]
    trig = int(g['skip_stop'].sum())
    L.append('%-10s %5.1f%% %5.1f%% %6d %14.2f %14.2f %10d' % (
        key[0], key[1] * 100, key[2] * 100, key[3], base_v, best['final'], trig))

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘价确认触线，不用 high/low；实盘还有排队与滑点，成交价只会更差。')
L.append('2. 持仓 stake = 开仓时 equity/pool；equity 只计已平仓盈亏，未计浮亏（真实回撤更深）。')
L.append('3. 日停手按 UTC 日切；达标后当日剩余信号丢弃，次日以新本金继续。')
L.append('4. 满仓时新信号直接丢弃（不做排队），这是单账号真实约束。')
L.append('5. 同币去重 GAP=60 根；跨币同波未去重。')
L.append('6. 数据仅 Binance 现货 1m；已下架归零币不在池内（幸存者偏差）。')
L.append('7. 终值含 9 年复利，极端值（>1e10）对初始误差、滑点、成交率极敏感，不可当预测。')
L.append('8. 未模拟最小下单量、资金费率、API 限频、交易所风控。')

out = os.path.join(EVI, 'combo_scan_all.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
print('csv ' + os.path.join(EVI, 'combo_scan_all.csv'))
