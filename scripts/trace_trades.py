# -*- coding: utf-8 -*-
"""采集每一笔的完整明细：入场/出场时间价格、出场原因、持有小时数、收益。
目的：把「止损单」和「止盈单」分开看——各需要多久、各亏多少赚多少。
"""
import os, sys
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'

fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
TP    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
SL    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
COST  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
POOL  = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DROP  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
VOLR  = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
HARG  = int(sys.argv[8]) if len(sys.argv) > 8 else 0

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

p('=' * 100)
p('TRACE TRADES  file=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d  hold=%s'
  % (fname, TP*100, SL*100, COST*100, POOL, 'UNLIMITED' if HARG == 0 else HARG))

f = os.path.join(DATA, fname)
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = [c.strip().replace('\ufeff', '') for c in fh.readline().split(',')]
need = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(f, usecols=[c for c in need if c in hdr])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
df = df.sort_values(['symbol', 'open_time'])

tf = '5m' if '5m' in fname else ('1d' if '1d' in fname else ('1h' if '1h' in fname else '1m'))
BARMS = {'1m': 60000, '5m': 300000, '1h': 3600000, '1d': 86400000}[tf]
H = HARG if HARG > 0 else 10**12
p('timeframe = %s  bars = %d  symbols = %d' % (tf, len(df), df.symbol.nunique()))

rows = []
for sym, b in df.groupby('symbol', sort=False):
    b = b.sort_values('open_time')
    n = len(b)
    if n < 30:
        continue
    o = b.open.to_numpy(np.float64); c = b.close.to_numpy(np.float64)
    h = b.high.to_numpy(np.float64); lo = b.low.to_numpy(np.float64)
    v = b.volume.to_numpy(np.float64); ts = b.open_time.to_numpy(np.int64)
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
        end = min(i + H, n)
        if end <= i + 1:
            continue
        hh = h[i+1:end]; ll = lo[i+1:end]
        tp_hit = np.where(hh >= entry * (1 + TP))[0]
        sl_hit = np.where(ll <= entry * (1 - SL))[0]
        i_tp = tp_hit[0] if len(tp_hit) else 10**9
        i_sl = sl_hit[0] if len(sl_hit) else 10**9
        if i_tp == 10**9 and i_sl == 10**9:
            if HARG > 0 and end - i - 1 < H:
                continue
            r = c[end-1] / entry - 1.0; j = end - 1
            reason = 'open_end' if HARG == 0 else 'timeout'
            exit_px = c[end-1]
        elif i_tp <= i_sl:
            r = TP; j = i + 1 + i_tp; reason = 'tp'
            exit_px = entry * (1 + TP)
        else:
            r = -SL; j = i + 1 + i_sl; reason = 'sl'
            exit_px = entry * (1 - SL)
        rows.append(dict(
            symbol=sym, entry_ms=int(ts[i]), entry_px=float(entry),
            exit_ms=int(ts[j]), exit_px=float(exit_px), reason=reason,
            hold_min=int((ts[j] - ts[i]) / 60000),
            gross=r, net=float(r - COST)))

T = pd.DataFrame(rows).sort_values('entry_ms').reset_index(drop=True)
p('raw signals = %d' % len(T))

# 池 N：按时间串行占位
slots, keep = [], []
for _, row in T.iterrows():
    slots = [s for s in slots if s > row.entry_ms]
    if len(slots) >= POOL:
        continue
    slots.append(row.exit_ms)
    keep.append(row)
K = pd.DataFrame(keep).reset_index(drop=True)
p('pool took = %d  over %d days' % (len(K), pd.to_datetime(K.entry_ms, unit='ms').dt.date.nunique()))

p('')
p('=' * 100)
p('按出场原因分组（pool 内的真实成交笔）')
p('=' * 100)
hdr2 = '%-10s %7s %8s %9s %9s %9s %9s %9s %9s'
p(hdr2 % ('reason', 'n', '占比', '均值', '中位', 'p10持', '中位持', 'p90持', '最大持'))
p('-' * 100)
for reason, g in K.groupby('reason'):
    hm = g.hold_min.to_numpy()
    p(hdr2 % (reason, len(g), '%.1f%%' % (100.0*len(g)/len(K)),
              '%+.2f%%' % (g.net.mean()*100), '%+.2f%%' % (g.net.median()*100),
              '%.0fm' % np.percentile(hm, 10), '%.0fm' % np.median(hm),
              '%.0fm' % np.percentile(hm, 90), '%.0fm' % hm.max()))
p('-' * 100)
p('%-10s %7d %8s %9s %9s' % ('ALL', len(K), '100%',
                             '%+.2f%%' % (K.net.mean()*100), '%+.2f%%' % (K.net.median()*100)))

# 持有小时分布（按天看更直观）
p('')
p('持有时间（小时）分位：')
for reason, g in K.groupby('reason'):
    hh = g.hold_min.to_numpy() / 60.0
    p('  %-9s p10=%6.2fh  p25=%6.2fh  中位=%6.2fh  p75=%6.2fh  p90=%6.2fh'
      % (reason, np.percentile(hh, 10), np.percentile(hh, 25), np.median(hh),
         np.percentile(hh, 75), np.percentile(hh, 90)))

# 止损单里：亏了多少、有没有"刚买就被打"
slg = K[K.reason == 'sl']
if len(slg):
    p('')
    p('止损单细分（n=%d）：' % len(slg))
    for lo_m, hi_m, lab in [(0, 15, '<=15分钟'), (15, 60, '15-60分钟'),
                            (60, 240, '1-4小时'), (240, 1440, '4-24小时'),
                            (1440, 10**9, '>24小时')]:
        sub = slg[(slg.hold_min >= lo_m) & (slg.hold_min < hi_m)]
        if len(sub):
            p('  %-12s n=%5d (%5.1f%%)   均值 %+.2f%%'
              % (lab, len(sub), 100.0*len(sub)/len(slg), sub.net.mean()*100))

tpg = K[K.reason == 'tp']
if len(tpg):
    p('')
    p('止盈单细分（n=%d）：' % len(tpg))
    for lo_m, hi_m, lab in [(0, 60, '<=1小时'), (60, 240, '1-4小时'),
                            (240, 1440, '4-24小时'), (1440, 4320, '1-3天'),
                            (4320, 10**9, '>3天')]:
        sub = tpg[(tpg.hold_min >= lo_m) & (tpg.hold_min < hi_m)]
        if len(sub):
            p('  %-12s n=%5d (%5.1f%%)   均值 %+.2f%%'
              % (lab, len(sub), 100.0*len(sub)/len(tpg), sub.net.mean()*100))

tag = '%s_tp%d_sl%d_p%d' % (tf, int(TP*1000), int(SL*1000), POOL)
csv_out = os.path.join(EVI, 'trades_%s.csv' % tag)
K.to_csv(csv_out, index=False, encoding='utf-8-sig')
p('')
p('明细 CSV -> %s  (%d 行)' % (csv_out, len(K)))

txt_out = os.path.join(EVI, 'trace_%s.txt' % tag)
with open(txt_out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('汇总 txt -> %s' % txt_out)
