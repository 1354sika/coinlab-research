# -*- coding: utf-8 -*-
"""analyze_sl_pierce.py — 止损到底被打穿多少？

问题：15m跌5% + TP3%/SL1% 这一族，亏损是不是因为"下跌轻松打穿止损线"？

回测口径：收盘价确认触线（不用 low）。所以"打穿"= 触发 SL 的那根 1m K 线，
它的收盘价本身已经远低于 -1%。真实市价止损还会在收盘价上再滑。

输出：
  1. 止损笔的出场跌幅分布（-1.0% ~ -1.1% / -1.1~-1.2 / ... / <-10%）
  2. 打穿造成的额外损失 = Σ(实际出场跌幅 - (-1%))  /  Σ|-1%|
  3. 若"零打穿"（每笔都精确 -1% 成交），策略均值会是多少
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'

LOOKBACK = 15
DROP = 0.05
TP = 0.03
SL = 0.01
COST = 0.0025
MAXBARS = 2880
NSYM = 60

L = []
P = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(x for x in os.listdir(DATA) if x.endswith('.csv'))[:NSYM]

rows = []   # (exit_r, what)  what: -1=sl, 1=tp, 0=timeout
nsig = 0

for fi in files:
    sym = fi[:-4]
    try:
        df = pd.read_csv(os.path.join(DATA, fi),
                         usecols=['open_time', 'open', 'close'])
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
    idx = idx[(idx >= LOOKBACK) & (idx + MAXBARS < n)]
    nsig += len(idx)

    for i in idx:
        iv = i + 1
        ep = o[iv]
        if not np.isfinite(ep) or ep <= 0:
            continue
        path = c[iv:iv + MAXBARS]
        rr = path / ep - 1.0
        slm = rr <= -SL
        tpm = rr >= TP
        i_sl = int(np.argmax(slm)) if slm.any() else 10 ** 9
        i_tp = int(np.argmax(tpm)) if tpm.any() else 10 ** 9
        if i_sl < i_tp:
            rows.append((float(rr[i_sl]), -1))
        elif i_tp < 10 ** 9:
            rows.append((float(rr[i_tp]), 1))
        else:
            rows.append((float(rr[-1]), 0))
    if len(rows) % 5000 < len(idx):
        p('  %-12s sig=%d rows=%d  %.0fs' % (sym, len(idx), len(rows), time.time() - t0))

A = np.array(rows, dtype=np.float64)
r = A[:, 0]
w = A[:, 1].astype(np.int32)
n = len(r)

p('')
p('=' * 96)
p('止损打穿分析  15m跌5%% + TP3%%/SL1%%  成本 0.25%%  |  %d 币 / %d 信号 / %d 笔 / %.0fs'
  % (NSYM, nsig, n, time.time() - t0))
p('出场口径：收盘价确认（不用 low），实盘市价止损还会在收盘价上再滑')
p('=' * 96)

sl_mask = w == -1
tp_mask = w == 1
to_mask = w == 0
slr = r[sl_mask]
p('总笔数 %d  |  止损 %d (%.1f%%)  止盈 %d (%.1f%%)  超时 %d (%.1f%%)'
  % (n, sl_mask.sum(), 100 * sl_mask.mean(), tp_mask.sum(), 100 * tp_mask.mean(),
     to_mask.sum(), 100 * to_mask.mean()))
p('')

p('=== 一、止损笔实际出场跌幅分布（看看有多少"打穿"了 -1%）')
bins = [(-0.0105, '精确 -1.0%~-1.05%'),
        (-0.011, '-1.05%~-1.10%'),
        (-0.012, '-1.10%~-1.20%'),
        (-0.015, '-1.20%~-1.50%'),
        (-0.02, '-1.50%~-2.00%'),
        (-0.03, '-2.00%~-3.00%'),
        (-0.05, '-3.00%~-5.00%'),
        (-0.10, '-5.00%~-10.00%'),
        (-1.00, '< -10%')]
prev = list(slr)
cnt_done = 0
for hi, name in bins:
    if name.startswith('精确'):
        m = (slr >= -0.0105)
    else:
        pass
# 直接分箱
edges = [-0.0105, -0.011, -0.012, -0.015, -0.02, -0.03, -0.05, -0.10, -1.01]
names = ['精确 -1.00%~-1.05%', '-1.05%~-1.10%', '-1.10%~-1.20%', '-1.20%~-1.50%',
         '-1.50%~-2.00%', '-2.00%~-3.00%', '-3.00%~-5.00%', '-5.00%~-10.00%', '< -10%']
cum = 0
for k in range(len(edges) - 1):
    lo, hi = edges[k], edges[k + 1]
    m = (slr < lo) & (slr >= hi)
    cnt = int(m.sum())
    cum += cnt
    p('  %-22s %6d  %5.1f%%   均 %+7.3f%%' %
      (names[k], cnt, 100.0 * cnt / len(slr), 100 * slr[m].mean() if cnt else 0))
p('  %-22s %6d  %5.1f%%' % ('合计', cum, 100.0 * cum / len(slr)))

p('')
p('=== 二、打穿有多严重')
p('  止损笔平均出场      %+7.4f%%' % (100 * slr.mean()))
p('  止损笔中位出场      %+7.4f%%' % (100 * np.median(slr)))
p('  止损笔最惨一笔      %+7.4f%%' % (100 * slr.min()))
p('  打穿到 <= -2%% 的    %d 笔  (%.1f%%)' % ((slr <= -0.02).sum(), 100 * (slr <= -0.02).mean()))
p('  打穿到 <= -3%% 的    %d 笔  (%.1f%%)' % ((slr <= -0.03).sum(), 100 * (slr <= -0.03).mean()))
p('  打穿到 <= -5%% 的    %d 笔  (%.1f%%)' % ((slr <= -0.05).sum(), 100 * (slr <= -0.05).mean()))

p('')
p('=== 三、打穿造成多少额外损失')
extra = np.where(slr < -SL, slr - (-SL), 0.0)   # 负值=多亏
p('  若每笔都精确 -1%% 成交，止损笔合计亏   %+10.2f 个百分点' % (100 * (-SL) * len(slr)))
p('  实际止损笔合计亏                     %+10.2f 个百分点' % (100 * slr.sum()))
p('  打穿造成的额外亏损                    %+10.2f 个百分点（占多亏的 %.1f%%）'
  % (100 * extra.sum(), 100 * extra.sum() / (slr.sum() * 100 + (-SL) * len(slr) * 100) if slr.sum() else 0))
p('')

p('=== 四、去掉打穿后策略会变成什么')
net_now = r - COST
p('  现状（扣 0.25%% 成本）      均值 %+7.4f%%/笔  胜率 %.1f%%'
  % (100 * net_now.mean(), 100 * (net_now > 0).mean()))
r_fixed = r.copy()
r_fixed[sl_mask] = -SL   # 假设每笔止损都精确 -1%
net_fix = r_fixed - COST
p('  若止损永远精确 -1%% 成交   均值 %+7.4f%%/笔  胜率 %.1f%%'
  % (100 * net_fix.mean(), 100 * (net_fix > 0).mean()))
p('  打穿吃掉了每笔 %+7.4f%%（占总收益差的 %.1f%%）'
  % (100 * (net_now.mean() - net_fix.mean()),
     100 * (net_fix.mean() - net_now.mean()) / net_fix.mean() if net_fix.mean() else 0))

p('')
p('=== 五、时间分布（打穿是否集中在某些年）')
p('  待补：本脚本按笔聚合，未按年份切')

p('')
p('--- 诚实边界 ---')
p('1. 收盘价确认触线；真实市价止损在急跌时还会在收盘价上再滑 0.05~0.3%%。')
p('2. 每根 1m K 线内 high/low 的顺序未知，"先触 SL 还是先触 TP"按收盘序列判断，与真实 tick 顺序有差。')
p('3. 未模拟成交率：急跌时市价单可能部分成交或成交价更差。')
p('4. 已下架归零币不在池内（幸存者偏差），这类币恰恰是最容易打穿的。')
p('5. 单一样本，含样本内参数。')

os.makedirs(EVI, exist_ok=True)
with open(os.path.join(EVI, 'sl_pierce.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written D:/中信证券/coinlab/evidence/sl_pierce.txt  TOTAL %.0fs' % (time.time() - t0))
