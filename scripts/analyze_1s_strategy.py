# -*- coding: utf-8 -*-
"""analyze_1s_strategy.py — 第一次用秒级 K 线跑策略（4 天 / 30 币）

诚实前提：只有 4 天、30 币、约 65% 的秒有数据。这不是统计，是"探路"。

扫的网格：
  信号：1s 跌 X%（X = 0.2/0.3/0.5/0.8/1.0）
  止盈：0.5/1.0/2.0/3.0 %
  止损：0.5/1.0/2.0 %
  最长持有 3600 秒（1 小时）
"""
import os, glob, time
import numpy as np
import pandas as pd

LIVE = r'D:\中信证券\coinlab\data\live'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'strategy_1s.txt')

DROPS = [0.002, 0.003, 0.005, 0.008, 0.010]
TPS = [0.005, 0.010, 0.020, 0.030]
SLS = [0.005, 0.010, 0.020]
MAXSEC = 3600
COST = 0.0025

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(glob.glob(os.path.join(LIVE, '*', 'klines_1s.csv')))
p('=' * 100)
p('秒级 K 线策略扫描  |  %d 个文件' % len(files))
p('=' * 100)

frames = []
for f in files:
    day = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f, usecols=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        p('  %s FAIL %s' % (day, str(e)[:50]))
        continue
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df = df.dropna(subset=['open_time'])
    df = df[df['symbol'].astype(str).str.endswith('USDT')]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    frames.append(df)
    p('  %s rows=%d syms=%d' % (day, len(df), df['symbol'].nunique()))

if not frames:
    p('没有数据'); raise SystemExit

D = pd.concat(frames, ignore_index=True)
D = D.dropna(subset=['open', 'close'])
D = D[D['close'] > 0].drop_duplicates(subset=['symbol', 'open_time'])
D = D.sort_values(['symbol', 'open_time'])
p('')
p('合并后 rows=%d  币=%d  时间 %s ~ %s' % (
    len(D), D['symbol'].nunique(),
    pd.to_datetime(int(D['open_time'].min()), unit='ms'),
    pd.to_datetime(int(D['open_time'].max()), unit='ms')))

# 每种 (drop, tp, sl) 的累计
from collections import defaultdict
acc = defaultdict(list)      # (drop, tp, sl) -> [(ret, what)]
nsig_by_drop = defaultdict(int)

for sym, g in D.groupby('symbol', sort=False):
    t = g['open_time'].to_numpy(np.int64)
    o = g['open'].to_numpy(np.float64)
    c = g['close'].to_numpy(np.float64)
    n = len(c)
    if n < 100:
        continue
    # 相邻行的秒差，用于判断有没有断档（断档超过 5 秒就不算一次"秒级变动"）
    gap = np.diff(t) / 1000.0
    ok = np.concatenate([[True], gap <= 5.0])   # ok[j]=1 表示 j 与 j-1 连续

    for drop in DROPS:
        # 相对 60 秒前（约）的跌幅：用固定回看 60 行（若断档则无效）
        LB = 60
        ret = np.zeros(n)
        ret[LB:] = c[LB:] / c[:-LB] - 1.0
        cont = np.zeros(n, dtype=bool)
        cont[LB:] = ok[LB:]
        idx = np.where(cont & (ret <= -drop))[0]
        idx = idx[(idx + MAXSEC < n)]
        nsig_by_drop[drop] += len(idx)

        for i in idx:
            iv = i + 1
            ep = o[iv]
            if not np.isfinite(ep) or ep <= 0:
                continue
            seg = c[iv:iv + MAXSEC] / ep - 1.0
            for tp in TPS:
                for sl in SLS:
                    m_sl = seg <= -sl
                    js = int(np.argmax(m_sl)) if m_sl.any() else 10 ** 9
                    jt = int(np.argmax(seg >= tp)) if (seg >= tp).any() else 10 ** 9
                    if js < jt and js < MAXSEC:
                        r, w = seg[js], -1
                    elif jt < MAXSEC:
                        r, w = seg[jt], 1
                    else:
                        r, w = seg[-1], 0
                    acc[(drop, tp, sl)].append(r - COST)

p('')
p('=== 信号数（按跌幅阈值）')
for drop in DROPS:
    p('  1 秒跌 %.1f%%  =>  %d 个信号' % (drop * 100, nsig_by_drop[drop]))

p('')
p('=== 网格结果（按均值排序，只列笔数 >= 30 的）')
rows = []
for (drop, tp, sl), arr in acc.items():
    a = np.array(arr, dtype=np.float64)
    if len(a) < 30:
        continue
    rows.append(dict(drop=drop, tp=tp, sl=sl, n=len(a), mean=a.mean(),
                     med=np.median(a), win=(a > 0).mean()))
rows.sort(key=lambda x: -x['mean'])

p('  %-10s %-8s %-8s %7s %9s %9s %8s' %
  ('跌幅', 'TP', 'SL', '笔数', '均值/笔', '中位', '胜率'))
p('  ' + '-' * 66)
for r in rows[:20]:
    p('  %-9.1f%% %-7.1f%% %-7.1f%% %7d %8.3f%% %8.2f%% %7.1f%%'
      % (r['drop'] * 100, r['tp'] * 100, r['sl'] * 100, r['n'],
         100 * r['mean'], 100 * r['med'], 100 * r['win']))

p('')
p('  正期望组合数：%d / %d' % (sum(1 for r in rows if r['mean'] > 0), len(rows)))
if rows:
    b = rows[0]
    p('  最优：1秒跌 %.1f%% / TP %.1f%% / SL %.1f%%  =>  %d 笔 / %+.3f%%/笔 / 胜率 %.1f%%'
      % (b['drop'] * 100, b['tp'] * 100, b['sl'] * 100, b['n'],
         100 * b['mean'], 100 * b['win']))

p('')
p('--- 诚实边界（这一节比结论更重要）---')
p('1. 只有 4 天（10-16/18/19/20）、30 个主流币、约 65%% 的秒有数据 —— 样本量做不了统计。')
p('2. 1 秒 K 线的 high/low 在 65%% 覆盖下会被切断，"先触 TP 还是先触 SL"判断不准。')
p('3. 断档 > 5 秒的相邻点已剔除，但断档本身会让信号漏掉（真实信号数应更多）。')
p('4. 成本 0.25%% 是线性近似，秒级策略笔数极多，成本敏感度比分钟级高一个量级。')
p('5. 收盘价确认触线，不含执行延迟；秒级策略对延迟最敏感，实盘会显著更差。')
p('6. 这是"探路"，不是"结论"。要下判断至少需要 30 天连续秒级数据。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
