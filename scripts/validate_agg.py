# -*- coding: utf-8 -*-
"""用 aggTrades（逐笔成交）重建 1m K 线，再跑同一套入场/出场验证。
K 线文件录得不全，但逐笔成交是全的，所以这是唯一能拿到完整覆盖的路。
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = r'D:\中信证券\coinlab'
LIVE = os.path.join(ROOT, 'data', 'live')
EVI = os.path.join(ROOT, 'evidence')

day  = sys.argv[1] if len(sys.argv) > 1 else '2026-10-19'
TP   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
SL   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
COST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
POOL = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DROP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
VOLR = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

f = os.path.join(LIVE, day, 'aggTrades.csv')
if not os.path.exists(f):
    p('MISSING %s' % f); sys.exit(2)

p('=' * 104)
p('AGGTRADES -> 1m REBUILD VALIDATE  day=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d'
  % (day, TP*100, SL*100, COST*100, POOL))
p('=' * 104)

# 先探表头
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = fh.readline().strip().replace('\ufeff', '')
    cols_h = [c.strip() for c in hdr.split(',')]
p('header: %s' % cols_h)

use = [c for c in ['symbol', 'price', 'qty', 'time'] if c in cols_h]
if len(use) < 4:
    p('UNEXPECTED HEADER, need symbol/price/qty/time'); sys.exit(3)

df = pd.read_csv(f, usecols=use)
p('rows = %d  symbols = %d' % (len(df), df.symbol.nunique()))

df['time'] = pd.to_numeric(df['time'], errors='coerce')
df = df[df['time'].notna()]
df['time'] = df['time'].astype('int64')

try:
    day0 = int(pd.Timestamp(day + ' 00:00:00', tz='UTC').value // 10**6)
except Exception:
    p('BAD DIRNAME %s' % day); sys.exit(4)
day1 = day0 + 86400 * 1000

d = df[(df.time >= day0) & (df.time < day1)].copy()
p('in-day trades = %d / %d  (%.1f%%)' % (len(d), len(df), 100.0 * len(d) / max(len(df), 1)))
if not len(d):
    p('NO TRADES IN DAY'); sys.exit(0)

d['t'] = d.time // 60000
d = d.sort_values('time')
g = (d.groupby(['symbol', 't'], sort=True)
      .agg(o=('price', 'first'), h=('price', 'max'), l=('price', 'min'),
           c=('price', 'last'), v=('qty', 'sum'))
      .reset_index())
p('1m bars rebuilt = %d   minutes covered = %d  (24h=1440)' % (len(g), g.t.nunique()))
per = g.groupby('symbol').size()
p('per-symbol 1m bars: min=%d median=%d max=%d' % (per.min(), per.median(), per.max()))

trades = []
for sym, b in g.groupby('symbol', sort=False):
    b = b.sort_values('t')
    n = len(b)
    if n < 30:
        continue
    c = b.c.to_numpy(np.float64); h = b.h.to_numpy(np.float64)
    lo = b.l.to_numpy(np.float64); v = b.v.to_numpy(np.float64)
    tt = b.t.to_numpy(np.int64)
    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR)
    for i in np.where(sig)[0]:
        entry = c[i]
        end = min(i + 1440, n)
        if end <= i + 1 or not np.isfinite(entry) or entry <= 0:
            continue
        hh = h[i+1:end]; ll = lo[i+1:end]
        tp_hit = np.where(hh >= entry * (1 + TP))[0]
        sl_hit = np.where(ll <= entry * (1 - SL))[0]
        i_tp = tp_hit[0] if len(tp_hit) else 10**9
        i_sl = sl_hit[0] if len(sl_hit) else 10**9
        if i_tp == 10**9 and i_sl == 10**9:
            if end - i - 1 < 1440:
                continue
            r = c[end-1] / entry - 1.0; reason = 'timeout'; j = end - 1
        elif i_tp <= i_sl:
            r = TP; reason = 'tp'; j = i + 1 + i_tp
        else:
            r = -SL; reason = 'sl'; j = i + 1 + i_sl
        trades.append((int(tt[i]), sym, float(r - COST), reason, int(tt[j] - tt[i])))

p('')
if not trades:
    p('SIGNALS = 0  —— 当天没有满足入场条件的点。')
else:
    T = pd.DataFrame(trades, columns=['t_in', 'sym', 'r', 'reason', 'hold_min']).sort_values('t_in').reset_index(drop=True)
    p('signals = %d   symbols hit = %d' % (len(T), T.sym.nunique()))
    p('mean   %+.4f%%   median %+.4f%%   win %.2f%% (n=%d)'
      % (T.r.mean()*100, T.r.median()*100, (T.r > 0).mean()*100, (T.r > 0).sum()))
    if (T.r > 0).any(): p('avg win  %+.3f%%' % (T.r[T.r > 0].mean()*100))
    if (T.r <= 0).any(): p('avg loss %+.3f%%' % (T.r[T.r <= 0].mean()*100))
    p('hold median %.0f min' % T.hold_min.median())
    p('exit mix: ' + '  '.join('%s=%d' % (k, v) for k, v in T.reason.value_counts().items()))

    slots, eq, peak, mdd, taken, skipped = [], 1.0, 1.0, 0.0, [], 0
    for _, row in T.iterrows():
        slots = [s for s in slots if s[0] > row.t_in]
        if len(slots) >= POOL:
            skipped += 1; continue
        slots.append((row.t_in + row.hold_min, row.r))
        taken.append(row.r)
        eq *= (1.0 + row.r / POOL)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    p('')
    p('--- pool N=%d ---' % POOL)
    p('taken %d  skipped %d' % (len(taken), skipped))
    p('equity 1.0 -> %.6f  (%.2f%%)   mdd %.2f%%' % (eq, (eq-1)*100, mdd*100))

p('')
p('--- honest boundary -----------------------------------------------')
p('1. 1m K 线由逐笔成交重建（open/close=首末成交价，high/low=极值，volume=数量求和），')
p('   与交易所官方 K 线在"同一毫秒多笔"的边界上可能有极小差异。')
p('2. 只有 live_recorder 监控的那批币（约 30 个），不是当时成交额 Top100 动态池。')
p('3. TP/SL 按触价即成交，未计排队与滑点（成本 %.2f%% 已含一次往返）。' % (COST*100))
p('4. 单日样本，只能回答"这套规则在当天有没有信号、结果如何"，不能作收益结论。')

out = os.path.join(EVI, 'validate_agg_%s.txt' % day)
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
