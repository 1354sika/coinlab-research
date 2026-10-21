# -*- coding: utf-8 -*-
"""用 live 录到的单日 1s K 线，按选定组合跑一遍验证。
入场: ret24 <= -2% 且 volr(24) >= 3.0
出场: TP / SL(同一根同触按SL保守) / 24h 到期
口径: 多币并发池 N，每笔占 1/N 资金，下单动作串行；含成本。
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = r'D:\中信证券\coinlab'
LIVE = os.path.join(ROOT, 'data', 'live')
EVI = os.path.join(ROOT, 'evidence')

day   = sys.argv[1] if len(sys.argv) > 1 else '2026-10-19'
TP    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
SL    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
COST  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
POOL  = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DROP  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
VOLR  = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

f = os.path.join(LIVE, day, 'klines_1s.csv')
if not os.path.exists(f):
    p('MISSING %s' % f); sys.exit(2)

p('=' * 100)
p('LIVE DAY VALIDATE  day=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d' % (day, TP*100, SL*100, COST*100, POOL))
p('entry: ret24<=%.1f%% & volr>=%.1f' % (DROP*100, VOLR))
p('=' * 100)

df = pd.read_csv(f, usecols=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
p('raw 1s rows = %d   symbols = %d' % (len(df), df.symbol.nunique()))
p('time span = %s ~ %s' % (
    pd.to_datetime(df.open_time.min(), unit='ms'),
    pd.to_datetime(df.open_time.max(), unit='ms')))

df['t'] = (df.open_time // 60000).astype(np.int64)
g = (df.groupby(['symbol', 't'], sort=True)
       .agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'),
            c=('close', 'last'), v=('volume', 'sum'))
       .reset_index())

nmin = g.t.nunique()
p('1m bars = %d   minutes covered = %d  (24h=1440)' % (len(g), nmin))
per = g.groupby('symbol').size()
p('per-symbol 1m bars: min=%d median=%d max=%d' % (per.min(), per.median(), per.max()))

trades = []
for sym, b in g.groupby('symbol', sort=False):
    b = b.sort_values('t')
    n = len(b)
    if n < 30:
        continue
    c = b.c.to_numpy(np.float64)
    h = b.h.to_numpy(np.float64)
    lo = b.l.to_numpy(np.float64)
    v = b.v.to_numpy(np.float64)
    tt = b.t.to_numpy(np.int64)

    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR)

    for i in np.where(sig)[0]:
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(i + 1440, n)
        if end <= i + 1:
            continue
        hh = h[i+1:end]
        ll = lo[i+1:end]
        tp_hit = np.where(hh >= entry * (1 + TP))[0]
        sl_hit = np.where(ll <= entry * (1 - SL))[0]
        i_tp = tp_hit[0] if len(tp_hit) else 10**9
        i_sl = sl_hit[0] if len(sl_hit) else 10**9
        if i_tp == 10**9 and i_sl == 10**9:
            if end - i - 1 < 1440:
                continue  # 未到期且当天结束 -> 不计（避免右截断偏差）
            r = c[end-1] / entry - 1.0
            reason = 'timeout'
            j = end - 1
        elif i_tp <= i_sl:
            r = TP
            reason = 'tp'
            j = i + 1 + i_tp
        else:
            r = -SL
            reason = 'sl'
            j = i + 1 + i_sl
        trades.append((int(tt[i]), sym, float(r - COST), reason, int(tt[j] - tt[i])))

if not trades:
    p('')
    p('NO SIGNAL — 当天没有满足入场条件的点。')
    out = os.path.join(EVI, 'validate_%s.txt' % day)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    p('written %s' % out)
    sys.exit(0)

T = pd.DataFrame(trades, columns=['t_in', 'sym', 'r', 'reason', 'hold_min'])
T = T.sort_values('t_in').reset_index(drop=True)
p('')
p('signals = %d   symbols hit = %d' % (len(T), T.sym.nunique()))
p('')
p('--- single-trade --------------------------------------------------')
p('mean   %+.4f%%' % (T.r.mean() * 100))
p('median %+.4f%%' % (T.r.median() * 100))
p('win    %.2f%%  (n=%d)' % ((T.r > 0).mean() * 100, (T.r > 0).sum()))
if (T.r > 0).any():
    p('avg win  %+.3f%%' % (T.r[T.r > 0].mean() * 100))
if (T.r <= 0).any():
    p('avg loss %+.3f%%' % (T.r[T.r <= 0].mean() * 100))
p('hold   median %.0f min' % T.hold_min.median())
p('exit mix: ' + '  '.join('%s=%d' % (k, v) for k, v in T.reason.value_counts().items()))

# ---- 池 N 并发模拟 ----
slots = []   # list of (t_out, r)
eq = 1.0
peak = 1.0
mdd = 0.0
taken = []
skipped = 0
for _, row in T.iterrows():
    slots = [s for s in slots if s[0] > row.t_in]
    if len(slots) >= POOL:
        skipped += 1
        continue
    slots.append((row.t_in + row.hold_min, row.r))
    taken.append(row.r)
    eq *= (1.0 + row.r / POOL)
    peak = max(peak, eq)
    mdd = min(mdd, eq / peak - 1.0)

p('')
p('--- pool N=%d ------------------------------------------------------' % POOL)
p('taken %d  skipped(fully occupied) %d' % (len(taken), skipped))
p('equity 1.0 -> %.6f  (%.2f%%)' % (eq, (eq - 1) * 100))
p('max drawdown %.2f%%' % (mdd * 100))
p('per-day trades %.2f' % len(taken))

p('')
p('--- honest boundary -----------------------------------------------')
p('1. live 1s K 线每币只覆盖部分秒（本日见上：每币 1m bars 中位数）。')
p('2. 只有 live_recorder 监控的那批币，不是当时成交额 Top100 动态池。')
p('3. TP/SL 按触价即成交，未计排队与滑点（成本 0.15% 已含一次往返）。')
p('4. 只有 1 个交易日，样本量小，只能算"当天跑通了没有"，不能作收益结论。')

out = os.path.join(EVI, 'validate_%s.txt' % day)
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
