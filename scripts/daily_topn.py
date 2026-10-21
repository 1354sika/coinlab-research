# -*- coding: utf-8 -*-
"""每天只做 Top-N 笔 —— 直接检验「一天两次、每次 +1%」这个目标。

口径（全部无前视）：
  · 入场候选：当根 1m 收盘时 ret24 <= -2% 且 volr >= 3（只用当根及之前）
  · 每天（UTC）只在候选里挑 volr 最大的 Top-N 笔（N=1/2/3），其余丢弃
  · 出场：之后每根 1m 的 **收盘价** 触线才成交（不用 high/low）
  · 单笔独立结算（不并发），复利 = 逐笔累乘
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'
FN = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
COST = 0.0015
DROP = 0.02
VOLR = 3.0
MAXH = 1440

t0 = time.time()
df = pd.read_csv(os.path.join(DATA, FN),
                 usecols=['symbol', 'open_time', 'close', 'volume'])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
df = df.dropna(subset=['open_time'])
df['open_time'] = df['open_time'].astype('int64')
for c in ('close', 'volume'):
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
df = df.dropna()
print('rows %d  %.1fs' % (len(df), time.time() - t0))

recs = []
paths = {}
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    c = g['close'].to_numpy()
    v = g['volume'].to_numpy()
    t = g['open_time'].to_numpy()
    n = len(c)
    if n < 3000:
        continue
    paths[sym] = c
    vma = pd.Series(v).rolling(1440).mean().to_numpy() * 1440
    ret24 = np.full(n, np.nan)
    ret24[1440:] = c[1440:] / c[:-1440] - 1.0
    volr = np.where(vma > 0, v * 1440 / vma, 0.0)
    idx = np.where((ret24 <= -DROP) & (volr >= VOLR))[0]
    for i in idx:
        if i + MAXH >= n:
            continue
        recs.append((int(t[i]), sym, int(i), float(volr[i])))

C = pd.DataFrame(recs, columns=['t_ms', 'sym', 'i', 'volr'])
C['day'] = (C['t_ms'] // 86400000).astype('int64')
print('candidates %d  days %d  %.1fs' % (len(C), C['day'].nunique(), time.time() - t0))

TPs = [0.010, 0.020, 0.030]
SLs = [0.010, 0.020, 0.030]

L = []
L.append('每天只做 Top-N 笔　（入场 ret24<=-%.0f%% & volr>=%.1f，按当日 volr 排序取前 N）' % (DROP * 100, VOLR))
L.append('出场：收盘价触线才成交（无前视）　成本 %.2f%% 往返　最长持 24h' % (COST * 100))
L.append('样本 %d 天，候选 %d 个（日均 %.1f 个）' % (
    C['day'].nunique(), len(C), len(C) / C['day'].nunique()))
L.append('')
L.append('%-4s %-6s %-6s %8s %8s %9s %10s %11s' % (
    'N', 'TP', 'SL', '笔数', '笔/天', '胜率', '单笔均值', '复利终值'))
L.append('-' * 78)

best = None
for N in (1, 2, 3):
    top = C.sort_values(['day', 'volr'], ascending=[True, False]).groupby('day').head(N)
    for TP in TPs:
        for SL in SLs:
            rs = []
            exits = []
            for _, r in top.iterrows():
                c = paths[r['sym']]
                i = int(r['i'])
                ep = c[i]
                up = ep * (1 + TP)
                dn = ep * (1 - SL)
                end = min(i + MAXH, len(c) - 1)
                seg = c[i + 1:end + 1]
                hu = np.where(seg >= up)[0]
                hd = np.where(seg <= dn)[0]
                ju = hu[0] if len(hu) else 10**9
                jd = hd[0] if len(hd) else 10**9
                j = min(ju, jd)
                if j == 10**9:
                    px = seg[-1]
                elif ju <= jd:
                    px = up
                else:
                    px = dn
                rs.append(px / ep - 1.0 - COST)
                exits.append(int(r['t_ms']) + (j + 1) * 60000)
            r = np.array(rs)
            order = np.argsort(exits)
            r = r[order]
            eq = float(np.prod(1.0 + r))
            nd = top['day'].nunique()
            L.append('%-4d %-6.1f%% %-6.1f%% %8d %8.2f %8.1f%% %9.4f%% %11.3f' % (
                N, TP * 100, SL * 100, len(r), len(r) / nd,
                (r > 0).mean() * 100, r.mean() * 100, eq))
            if best is None or eq > best[0]:
                best = (eq, N, TP, SL, r.mean(), len(r) / nd, (r > 0).mean())

L.append('')
L.append('=== 目标对照 ===')
L.append('目标：一天 2 笔、每笔净 +1%% => 每天 +2.01%% => 一年 1.0201^365 = %.1f 倍' % (1.0201 ** 365))
if best:
    eq, N, TP, SL, mean, pd_, win = best
    L.append('实测最好：N=%d TP=%.1f%% SL=%.1f%%  单笔 %+.4f%%  笔/天 %.2f  胜率 %.1f%%  终值 %.3f 倍' % (
        N, TP * 100, SL * 100, mean * 100, pd_, win * 100, eq))
    L.append('折年化（若分布不变、按 %.2f 笔/天）: %.3e 倍' % (pd_, (1 + mean) ** (pd_ * 365)))
    L.append('与目标差：单笔 %+.4f%%  vs  目标 +1.0000%%  => 差 %.2f 个百分点' % (
        mean * 100, 1.0 - mean * 100))
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘确认，不用 high/low 触价；实盘还需排队、滑点，这里只扣 0.15% 固定成本。')
L.append('2. 「按 volr 排序取 Top-N」是事前可观测的，无前视。')
L.append('3. 单笔独立结算（同时只持一笔），未模拟并发占用。')
L.append('4. 单一数据集，含样本内；未做时间切分外推。')

out = os.path.join(EVI, 'daily_topn.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
