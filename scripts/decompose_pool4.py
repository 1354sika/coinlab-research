# -*- coding: utf-8 -*-
"""decompose_pool4.py — 池 4 那档（45.6%/-59.9%）到底靠什么撑起来

要回答四个问题：
1) 逐年收益：是不是靠某一年的极端行情
2) 按币贡献：是不是靠少数几个币
3) 去掉最赚的 3 个币后还剩多少（幸存者偏差检验）
4) 前 70% / 后 30% 时间切分（样本外检验）

口径：15m 跌 5% 信号、TP 3% / SL 1%、池 4（每笔 25% 资金）、满仓跳过、成本 0.25%
数据：E 盘 1m，字母序前 60 币（与 sl_pierce 同一批，便于对照）
"""
import os, time
import numpy as np
import pandas as pd
from collections import defaultdict

DATA = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'decompose_pool4.txt')

LOOKBACK = 15
DROP = 0.05
TP = 0.03
SL = 0.01
MAXBARS = 2880
COST = 0.0025
POOL = 4
NSYM = 60

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


def pool_run(ev, pool, capital=1.0):
    """ev: [(t_in, t_out, r, sym)] 已按 t_in 排序"""
    eq = capital
    peak = eq
    mdd = 0.0
    open_until = []
    taken = 0
    curve = []
    for (t_in, t_out, r, sym) in ev:
        open_until = [x for x in open_until if x > t_in]
        if len(open_until) >= pool:
            continue
        stake = eq / pool
        before = eq
        eq += stake * r
        open_until.append(t_out)
        taken += 1
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        mdd = min(mdd, dd)
        curve.append((t_in, eq, r, sym))
    return dict(eq=eq, mdd=mdd, taken=taken, curve=curve)


t0 = time.time()
files = sorted(x for x in os.listdir(DATA) if x.endswith('.csv'))[:NSYM]
p('=' * 96)
p('池 4 分解检验  15m跌5%% + TP3%%/SL1%%  池 %d  成本 %.2f%%' % (POOL, COST * 100))
p('=' * 96)

ev = []
tmin = tmax = None
for fi in files:
    sym = fi[:-4]
    try:
        df = pd.read_csv(os.path.join(DATA, fi), usecols=['open_time', 'open', 'close'])
    except Exception:
        continue
    if len(df) < 50000:
        continue
    t = df['open_time'].to_numpy(np.int64)
    o = df['open'].to_numpy(np.float32)
    c = df['close'].to_numpy(np.float32)
    n = len(c)
    ret = c / np.roll(c, LOOKBACK) - 1.0
    ret[:LOOKBACK] = 0.0
    idx = np.where(ret <= -DROP)[0]
    idx = idx[(idx >= LOOKBACK) & (idx + MAXBARS + 2 < n)]
    if not len(idx):
        continue
    if tmin is None:
        tmin, tmax = int(t[idx[0]]), int(t[-1])
    else:
        tmin = min(tmin, int(t[idx[0]])); tmax = max(tmax, int(t[-1]))
    for i in idx:
        iv = i + 1
        ep = o[iv]
        if not np.isfinite(ep) or ep <= 0:
            continue
        seg = c[iv:iv + MAXBARS] / ep - 1.0
        m = seg <= -SL
        js = int(np.argmax(m)) if m.any() else 10 ** 9
        jt = int(np.argmax(seg >= TP)) if (seg >= TP).any() else 10 ** 9
        if js < jt and js < MAXBARS:
            k = js
        elif jt < MAXBARS:
            k = jt
        else:
            k = MAXBARS - 1
        ev.append((int(t[iv]), int(t[iv + k]), float(seg[k] - COST), sym))

ev.sort()
span_days = (tmax - tmin) / 86400000.0 if tmin else 1
p('信号 %d 笔  跨度 %.1f 年（%.0f 天）' % (len(ev), span_days / 365.25, span_days))

r = pool_run(ev, POOL)
ann = r['eq'] ** (365.25 / span_days) - 1 if span_days > 0 else 0
p('')
p('=== 基准结果')
p('  成交 %d 笔  期末 %.3fx  年化 %.2f%%  最大回撤 %.2f%%'
  % (r['taken'], r['eq'], 100 * ann, 100 * r['mdd']))

# 1) 逐年
p('')
p('=== 一、逐年收益（每一年内按同一池规则滚动）')
cur = r['curve']
by_year = defaultdict(list)
for (t_in, eq, rr, sym) in cur:
    y = time.strftime('%Y', time.gmtime(t_in / 1000.0))
    by_year[y].append(eq)
p('  %-6s %8s %12s %12s' % ('年', '成交笔', '年末权益', '年内变化'))
p('  ' + '-' * 44)
prev = 1.0
years_sorted = sorted(by_year)
for y in years_sorted:
    e = by_year[y][-1]
    p('  %-6s %8d %12.3f %11.1f%%' % (y, len(by_year[y]), e, 100 * (e / prev - 1)))
    prev = e
pos_years = sum(1 for y in years_sorted if True)
p('  （注：年末权益是累计值，年内变化按上一年的年末算）')

# 2) 按币贡献
p('')
p('=== 二、按币贡献（该币所有笔的 r 之和，未加权）')
by_sym = defaultdict(list)
for (t_in, t_out, rr, sym) in ev:
    by_sym[sym].append(rr)
rows = []
for sym, arr in by_sym.items():
    a = np.array(arr)
    rows.append((sym, len(a), a.mean(), a.sum()))
rows.sort(key=lambda x: -x[3])
p('  %-15s %7s %10s %12s' % ('币', '笔数', '均值/笔', '累计贡献'))
p('  ' + '-' * 48)
for s, n, mu, tot in rows[:10]:
    p('  %-15s %7d %9.3f%% %11.2f' % (s, n, 100 * mu, tot))
p('  ...')
for s, n, mu, tot in rows[-5:]:
    p('  %-15s %7d %9.3f%% %11.2f' % (s, n, 100 * mu, tot))
pos = sum(1 for s, n, mu, tot in rows if mu > 0)
p('  正期望币：%d / %d' % (pos, len(rows)))

# 3) 去掉最赚的 3 个币
p('')
p('=== 三、去掉最赚的 3 个币后（幸存者偏差检验）')
top3 = [x[0] for x in rows[:3]]
p('  去掉：%s' % ', '.join(top3))
ev2 = [x for x in ev if x[3] not in set(top3)]
r2 = pool_run(ev2, POOL)
ann2 = r2['eq'] ** (365.25 / span_days) - 1 if span_days > 0 else 0
p('  剩余 %d 笔  期末 %.3fx  年化 %.2f%%  回撤 %.2f%%'
  % (r2['taken'], r2['eq'], 100 * ann2, 100 * r2['mdd']))
p('  年化变化：%.2f%% -> %.2f%%（掉 %.1f 个百分点）'
  % (100 * ann, 100 * ann2, 100 * (ann - ann2)))

# 4) 时间切分
p('')
p('=== 四、时间切分（前 70% / 后 30%）')
mid = ev[int(len(ev) * 0.7)][0]
tr = [x for x in ev if x[0] <= mid]
va = [x for x in ev if x[0] > mid]
for nm, sub in [('前 70%', tr), ('后 30%', va)]:
    if not sub:
        continue
    span = (sub[-1][0] - sub[0][0]) / 86400000.0
    rr = pool_run(sub, POOL)
    a = rr['eq'] ** (365.25 / span) - 1 if span > 0 else 0
    p('  %-8s 信号 %6d  成交 %5d  跨度 %5.1f 年  期末 %.3fx  年化 %+7.2f%%  回撤 %.2f%%'
      % (nm, len(sub), rr['taken'], span / 365.25, rr['eq'], 100 * a, 100 * rr['mdd']))

p('')
p('--- 诚实边界 ---')
p('1. 只用字母序前 60 币（不是 299 币），与 final_board 的 45.6%% 不可直接比较。')
p('2. 收盘价确认触线，不含执行延迟（15m 下延迟占 0.28%%，可忽略）。')
p('3. 成本 0.25%% 线性近似；equity 只计已平仓盈亏，未计浮亏。')
p('4. 逐年"年末权益"是累计值，未按年单独复利。')
p('5. 去掉 top3 币的检验只是粗略下界，未考虑资金重新分配。')
p('6. 已下架归零币不在池内。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
