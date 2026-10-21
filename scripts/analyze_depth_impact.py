# -*- coding: utf-8 -*-
"""analyze_depth_impact.py — 用真实盘口快照算"卖出冲击成本"

这是"数学预测滑点"的第一步：不是算模型，是先用真实盘口量出来。

口径：
  · 数据 = data/live/<day>/depth20.csv（20 档买卖盘，10 秒一张，只覆盖 30 个主流币）
  · 模拟：持有某个币，想吃掉 bid 侧若干档成交 Q 元
  · 输出：成交均价相对最优买价 BBO 的滑点（bp）+ 吃穿档数

为什么这件事重要：
  上一轮 sl_pierce 的 -1.89% 是"收盘价确认 + 按收盘价成交"，不含任何执行延迟；
  所以那部分是市场自己跳空。真正"下单造成"的滑点必须用盘口量。
"""
import os, glob, time
import numpy as np
import pandas as pd

LIVE = r'D:\中信证券\coinlab\data\live'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'depth_impact.txt')

QTY_LIST = [50.0, 200.0, 1000.0, 5000.0]     # 卖出金额（USDT）
L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(glob.glob(os.path.join(LIVE, '*', 'depth20.csv')))
p('=' * 104)
p('盘口冲击成本实测  数据文件 %d 个' % len(files))
p('=' * 104)

rows_all = []
for f in files:
    day = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f, usecols=['ts_ms', 'symbol', 'side', 'level', 'price', 'qty'])
    except Exception as e:
        p('  %s READ_FAIL %s' % (day, str(e)[:60]))
        continue
    if not len(df):
        continue
    p('  %s rows=%d  syms=%d' % (day, len(df), df['symbol'].nunique()))

    # 只取 bid 侧（卖出吃买盘）
    b = df[df['side'].astype(str).str.lower().isin(['bid', 'b', 'buy'])]
    if not len(b):
        b = df[df['side'].astype(str).str.lower() == 'bid']
    if not len(b):
        p('    (无 bid 侧，跳过)')
        continue

    for (sym, ts), g in b.groupby(['symbol', 'ts_ms'], sort=False):
        g = g.sort_values('level')
        px = g['price'].to_numpy(np.float64)
        qt = g['qty'].to_numpy(np.float64)
        if len(px) < 2 or px[0] <= 0:
            continue
        bbo = px[0]
        # 累计名义金额（从最优价往下吃）
        cum_usd = np.cumsum(px * qt)
        rec = {'day': day, 'symbol': sym}
        for Q in QTY_LIST:
            if cum_usd[-1] < Q:
                rec['slip_%g' % Q] = np.nan      # 盘口不够深，吃不到
                rec['lvl_%g' % Q] = np.nan
                continue
            k = int(np.searchsorted(cum_usd, Q))
            # 均价 = Q / 累计数量（最后一档部分成交）
            qty_before = cum_usd[k - 1] / px[k] if k > 0 else 0.0
            # 更准确：按名义金额反算数量
            if k == 0:
                px_avg = px[0]
            else:
                # 前 k 档全吃，第 k 档吃剩余
                usd_before = cum_usd[k - 1]
                qty_before = (usd_before / px[:k]).sum() if False else 0.0
                # 直接积分：前 k 档数量总和 + 第 k 档部分
                qsum = qt[:k].sum()
                rest_usd = Q - usd_before
                rest_qty = rest_usd / px[k]
                qsum += rest_qty
                px_avg = Q / qsum
            rec['slip_%g' % Q] = (bbo / px_avg - 1.0) * 1e4    # bp，正=卖低了
            rec['lvl_%g' % Q] = k + 1
        rows_all.append(rec)

if not rows_all:
    p('  没有可用快照'); raise SystemExit

D = pd.DataFrame(rows_all)
p('')
p('快照数 %d  |  币 %d  |  %.0fs' % (len(D), D['symbol'].nunique(), time.time() - t0))

p('')
p('=== 一、不同卖出金额的平均滑点（bp，1bp = 0.01%）')
p('  说明：正值 = 卖价比最优买价低多少；负数 = 盘口异常（不应出现）')
hdr = '  %-10s' % '金额(USDT)'
for Q in QTY_LIST:
    hdr += '%10s' % ('%g' % Q)
p(hdr)
p('  ' + '-' * (10 + 10 * len(QTY_LIST)))
for name, fn in [('中位滑点', np.nanmedian), ('均值滑点', np.nanmean),
                 ('P90 滑点', lambda x: np.nanpercentile(x, 90)),
                 ('能吃到的比例', None)]:
    line = '  %-10s' % name
    for Q in QTY_LIST:
        col = D['slip_%g' % Q]
        if fn is None:
            line += '%9.1f%%' % (100 * col.notna().mean())
        else:
            line += '%9.2fbp' % fn(col)
    p(line)

p('')
p('=== 二、吃穿档数（20 档里吃到第几档）')
hdr = '  %-10s' % '金额(USDT)'
for Q in QTY_LIST:
    hdr += '%10s' % ('%g' % Q)
p(hdr)
p('  ' + '-' * (10 + 10 * len(QTY_LIST)))
for name, fn in [('中位档数', np.nanmedian), ('P90 档数', lambda x: np.nanpercentile(x, 90))]:
    line = '  %-10s' % name
    for Q in QTY_LIST:
        line += '%9.1f' % fn(D['lvl_%g' % Q])
    p(line)

p('')
p('=== 三、按币看（金额 1000 USDT）')
g = D.groupby('symbol').agg(
    n=('slip_1000', 'size'),
    med_slip=('slip_1000', 'median'),
    p90_slip=('slip_1000', lambda x: np.nanpercentile(x, 90)),
    med_lvl=('lvl_1000', 'median'),
    hit=('slip_1000', lambda x: x.notna().mean()),
).sort_values('med_slip')
p('  %-12s %6s %10s %10s %8s %8s' % ('symbol', '快照', '中位bp', 'P90bp', '中位档', '可成交'))
for sym, r in g.head(30).iterrows():
    p('  %-12s %6d %10.2f %10.2f %8.1f %7.1f%%'
      % (sym, r['n'], r['med_slip'], r['p90_slip'], r['med_lvl'], 100 * r['hit']))

p('')
p('=== 四、结论')
for Q in QTY_LIST:
    med = np.nanmedian(D['slip_%g' % Q])
    cov = D['slip_%g' % Q].notna().mean()
    p('  卖 %-7g USDT  中位滑点 %6.2f bp (%.3f%%)  盘口吃得下 %5.1f%% 的快照'
      % (Q, med, med / 100.0, 100 * cov))

p('')
p('--- 诚实边界 ---')
p('1. depth20 只 20 档，且每 10 秒一张快照，看不到秒内变化与更深档位。')
p('2. 只覆盖 live_recorder 的 30 个主流币，不是全市场；小币盘口更薄。')
p('3. 静态快照估算，没有模拟"你的卖单本身会抽走流动性、别人会撤单"（真实冲击更大）。')
p('4. 未含手续费，只算价差滑点。')
p('5. 快照时间跨度仅数天，行情状态单一。')

os.makedirs(EVI, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
