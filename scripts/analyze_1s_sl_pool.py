# -*- coding: utf-8 -*-
"""analyze_1s_sl_pool.py — 两件事

1) 秒级止损线细扫（0.3%~3.0%，步长 0.2%），找最优 SL
2) 用最优 SL 做池 N 模拟，算"每天/每年本金翻多少"

口径：1 秒跌 1% 信号、TP 2%、成交延迟 3 秒（实测 2.54 秒向上取整）、
      成本 0.25% 往返、最长持有 3600 秒。
样本：4 天、30 币、约 65% 覆盖 —— 数字是探路，不是预测。
"""
import os, glob, time
import numpy as np
import pandas as pd
from collections import defaultdict

LIVE = r'D:\中信证券\coinlab\data\live'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'sl_pool_1s.txt')

DROP = 0.010
TP = 0.020
DELAY = 3
MAXSEC = 3600
COST = 0.0025
SLS = [round(0.003 + 0.002 * k, 3) for k in range(14)]   # 0.3 ~ 2.9
POOLS = [1, 3, 5, 10, 20, 50]

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(glob.glob(os.path.join(LIVE, '*', 'klines_1s.csv')))
p('=' * 100)
p('秒级 SL 细扫 + 池模拟  信号=1秒跌1%%  TP=2%%  延迟=%d秒  成本=%.2f%%' % (DELAY, COST * 100))
p('=' * 100)

frames = []
for f in files:
    day = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f, usecols=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        p('  %s FAIL %s' % (day, str(e)[:60])); continue
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df = df.dropna(subset=['open_time'])
    df = df[df['symbol'].astype(str).str.endswith('USDT')]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'close'])[df['close'] > 0]
    p('  %s rows=%d syms=%d' % (day, len(df), df['symbol'].nunique()))
    frames.append(df)

D = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['symbol', 'open_time'])
D = D.sort_values(['symbol', 'open_time'])
p('合并 rows=%d 币=%d' % (len(D), D['symbol'].nunique()))

# 收集信号
sigs = []     # (sym, t_in, t_out_by_sl{sl: (t_out, r)})
n_cand = 0
for sym, g in D.groupby('symbol', sort=False):
    t = g['open_time'].to_numpy(np.int64)
    c = g['close'].to_numpy(np.float64)
    n = len(c)
    if n < 200:
        continue
    gap = np.diff(t) / 1000.0
    cont = np.concatenate([[True], gap <= 5.0])
    LB = 60
    ret = np.zeros(n)
    ret[LB:] = c[LB:] / c[:-LB] - 1.0
    ok = np.zeros(n, dtype=bool); ok[LB:] = cont[LB:]
    idx = np.where(ok & (ret <= -DROP))[0]
    idx = idx[(idx + 60 < n)]
    n_cand += len(idx)

    for i in idx:
        j = i + DELAY
        if j >= n:
            continue
        ep = c[j]
        if not np.isfinite(ep) or ep <= 0:
            continue
        seg = c[j + 1:j + 1 + MAXSEC] / ep - 1.0
        if len(seg) < 10:
            continue
        tpm = seg >= TP
        jt = int(np.argmax(tpm)) if tpm.any() else 10 ** 9
        by_sl = {}
        for sl in SLS:
            m = seg <= -sl
            js = int(np.argmax(m)) if m.any() else 10 ** 9
            if js < jt and js < MAXSEC:
                by_sl[sl] = (int(t[j + 1 + js]), float(seg[js] - COST))
            elif jt < MAXSEC:
                by_sl[sl] = (int(t[j + 1 + jt]), float(seg[jt] - COST))
            else:
                by_sl[sl] = (int(t[j + 1 + len(seg) - 1]), float(seg[-1] - COST))
        sigs.append((sym, int(t[j]), by_sl))

p('')
p('信号数 %d（1秒跌1%%）' % n_cand)

p('')
p('=== 一、SL 细扫（TP 固定 2%%，扣 0.25%%）')
p('  %-7s %8s %8s %9s %9s %10s' % ('SL', '笔数', '胜率', '均值/笔', '中位', 'SL出场占比'))
p('  ' + '-' * 60)
best = None
for sl in SLS:
    arr = np.array([d[sl][1] for _, _, d in sigs], dtype=np.float64)
    nsl = sum(1 for _, _, d in sigs if d[sl][1] <= -sl - COST + 1e-9)
    p('  %-6.1f%% %8d %7.1f%% %8.3f%% %8.2f%% %8.1f%%'
      % (sl * 100, len(arr), 100 * (arr > 0).mean(), 100 * arr.mean(),
         100 * np.median(arr), 100.0 * nsl / len(arr)))
    if best is None or arr.mean() > best[1]:
        best = (sl, arr.mean(), (arr > 0).mean())
p('  => 最优 SL = %.1f%%  均值 %+.3f%%  胜率 %.1f%%'
  % (best[0] * 100, 100 * best[1], 100 * best[2]))
SL_BEST = best[0]

p('')
p('=== 二、池模拟（SL=%.1f%%，每笔占 equity/池N，满仓跳过）' % (SL_BEST * 100))
ev = sorted([(t_in, d[SL_BEST][0], d[SL_BEST][1]) for _, t_in, d in sigs])
ts = [x[0] for x in ev]
span_days = (max(ts) - min(ts)) / 86400000.0 if ts else 1
p('  信号时间跨度 %.2f 天' % span_days)
p('  %-6s %9s %9s %13s %11s %10s' % ('池N', '成交笔', '日均笔', '期末倍数', '日均复利', '年化'))
p('  ' + '-' * 72)
for N in POOLS:
    eq = 1.0
    open_until = []
    taken = 0
    for (t_in, t_out, r) in ev:
        open_until = [x for x in open_until if x > t_in]
        if len(open_until) >= N:
            continue
        stake = eq / N
        eq += stake * r
        open_until.append(t_out)
        taken += 1
    if span_days <= 0:
        continue
    per_day = eq ** (1.0 / span_days) - 1
    ann = eq ** (365.25 / span_days) - 1
    p('  %-6d %9d %8.1f %13.3f %10.3f%% %9.1f%%'
      % (N, taken, taken / span_days, eq, 100 * per_day, 100 * ann))

p('')
p('=== 三、"每天本金翻倍"需要什么')
p('  每天翻倍 = 日复利 100%')
p('  按本表最激进的池 N，日复利已列在上面 —— 看差多少倍。')

p('')
p('--- 诚实边界 ---')
p('1. 4 天 / 30 币 / 约 65%% 的秒有数据 —— 样本做不了统计，本表是探路。')
p('2. 延迟按 3 秒（实测 2.54 向上取整）；成交价用延迟后那一秒的 close 近似。')
p('3. 65%% 覆盖时 i+3 秒可能不存在，只能取到存在的行。')
p('4. 未模拟成交率：急跌时市价单可能部分成交或不成交。')
p('5. 池 N 满仓跳过；equity 只计已平仓盈亏，未计浮亏。')
p('6. 4 天外推到 365 天是纯数学，实际行情分布会变。')
p('7. 已下架归零币不在池内。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
