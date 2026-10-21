# -*- coding: utf-8 -*-
"""_probe_stophunt_fast.py — 抄近路版止损猎杀检测（只读前 120 万行 ≈ 5 个币，秒级）
定义：局启动前 6h 窗口内，是否出现「low 跌破前 2h 最低 ×0.99 且 1 根内收回」
"""
import os, time
import numpy as np, pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
t0 = time.time()
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'),
                 usecols=['symbol', 'open_time', 'high', 'low', 'close'], nrows=1200000)
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %s  %.1fs' % (len(df), list(df.symbol.unique()), time.time() - t0))

ev = pd.read_csv(os.path.join(E, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)

def to_ms(s):
    v = pd.to_datetime(s).astype('int64').values.astype('int64')
    d = int(v[0])
    if d > 10 ** 17: v = v // 10 ** 6
    elif d > 10 ** 14: v = v // 10 ** 3
    elif d > 10 ** 11: pass
    else: v = v * 1000
    return v

ev['t0'] = to_ms(ev['start_time']); ev['t1'] = to_ms(ev['peak_time'])
BK, REF, DEP, HOLD = 72, 24, 0.99, 1

def scan(lo, cl, a):
    hits = []
    for k in range(max(REF + 1, a - BK), a - HOLD):
        r = np.nanmin(lo[k - REF:k])
        if not np.isfinite(r) or r <= 0:
            continue
        if lo[k] < r * DEP and cl[k + HOLD] > r:
            hits.append((a - k, lo[k] / r - 1.0))
    return hits

rows, rnd = [], []
rs = np.random.RandomState(7)
for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    ot = gg['open_time'].values.astype('int64')
    lo = gg['low'].values.astype('float64'); cl = gg['close'].values.astype('float64')
    n = len(gg)
    if n < 1000:
        continue
    sub = ev[ev['symbol'] == s]
    for _, e in sub.iterrows():
        a = int(np.searchsorted(ot, e['t0']))
        if a < BK + 2 or a >= n - 2:
            continue
        h = scan(lo, cl, a)
        rows.append((s, float(e['gain']), 1 if h else 0,
                     (h[0][0] if h else np.nan), (h[0][1] if h else np.nan), len(h)))
    for a in rs.randint(BK + 2, n - 2, size=min(300, max(1, n // 3000))):
        h = scan(lo, cl, int(a))
        rnd.append((s, 1 if h else 0))

R = pd.DataFrame(rows, columns=['symbol', 'gain', 'has', 'ahead', 'depth', 'nhits'])
Rr = pd.DataFrame(rnd, columns=['symbol', 'has'])
print('\n事件可对齐 %d ｜ 随机对照 %d' % (len(R), len(Rr)))
if len(R):
    print('=' * 88)
    print('【A 拉升前是否出现「假跌破再收回」】')
    print('=' * 88)
    print('  事件组 有假跌破 %4d/%4d = %.1f%%' % (R['has'].sum(), len(R), 100 * R['has'].mean()))
    print('  随机组 有假跌破 %4d/%4d = %.1f%%' % (Rr['has'].sum(), len(Rr), 100 * Rr['has'].mean()))
    lift = R['has'].mean() / max(Rr['has'].mean(), 1e-9)
    print('  ⇒ 提升 %.2f×  %s' % (lift, '★显著' if lift > 1.3 else '接近随机'))
    h = R[R['has'] == 1]
    if len(h):
        print('\n【B 假跌破距启动】中位 %.1f 根（%.1fh）｜深度中位 %+.2f%%'
              % (h['ahead'].median(), h['ahead'].median() * 5 / 60, 100 * h['depth'].median()))
        for lo_, hi_ in [(0, 6), (6, 12), (12, 24), (24, 48), (48, 73)]:
            m = (h['ahead'] > lo_) & (h['ahead'] <= hi_)
            if m.sum():
                print('    %2d~%2d 根前: %3d（%.0f%%）' % (lo_, hi_, m.sum(), 100 * m.sum() / len(h)))
        print('\n【C 有/无两组后续】')
        for lbl, su in [('有假跌破', R[R['has'] == 1]), ('无假跌破', R[R['has'] == 0])]:
            if len(su):
                print('    %-6s n=%3d  涨幅中位 %.2f 倍' % (lbl, len(su), su['gain'].median()))
print('\nTOTAL %.1fs' % (time.time() - t0))
