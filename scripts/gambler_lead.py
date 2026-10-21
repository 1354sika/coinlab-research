# -*- coding: utf-8 -*-
"""gambler_lead.py — 我们的信号到底「提前」还是「滞后」？
对每个无前视信号触发点，算：
  · 触发时价格在过去 1/6/12 根已经涨了多少   → 判断滞后量
  · 触发后 1/6/12 根、24h 的收益             → 判断还剩多少空间
判读：触发时过去涨幅中位 ≈ 0% ⇒ 真提前；> 5% ⇒ 只是在追已动过的。
"""
import os, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'
NR = int(os.environ.get('NR', '5000000'))
t0 = time.time()
COLS = ['symbol', 'open_time', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_base']
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=COLS, nrows=NR)
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %d  %.1fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

WS = [1, 6, 12]
FWD = [1, 6, 12, 288]

def stats(cl, hi, lo, mask, stride=9):
    rec = []
    for t in np.where(mask)[0][::stride]:
        if t < 300 or t + max(FWD) + 1 >= len(cl):
            continue
        e = cl[t]
        if not np.isfinite(e) or e <= 0:
            continue
        row = []
        for w in WS:                                   # 过去涨幅（已知）
            row.append(cl[t] / cl[t - w] - 1.0 if cl[t - w] > 0 else np.nan)
        for f in FWD:                                  # 未来收益（衡量空间）
            row.append(cl[t + f] / e - 1.0)
        row.append(np.nanmax(hi[t + 1:t + 289]) / e - 1.0 if e > 0 else np.nan)   # 未来 24h 最高
        rec.append(row)
    return np.array(rec, dtype='float64') if rec else np.empty((0, len(WS) + len(FWD) + 1))

agg = {}
for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    cl = gg['close'].values.astype('float64'); hi = gg['high'].values.astype('float64')
    lo = gg['low'].values.astype('float64')
    vol = gg['volume'].values.astype('float64'); trd = gg['trades'].values.astype('float64')
    n = len(gg)
    if n < 1500:
        continue
    mv = pd.Series(vol).rolling(288, min_periods=60).median().shift(1).values
    mt = pd.Series(trd).rolling(288, min_periods=60).median().shift(1).values
    vol_r6 = pd.Series(vol).rolling(6, min_periods=3).median().values / np.where(mv > 0, mv, np.nan)
    trd_r6 = pd.Series(trd).rolling(6, min_periods=3).median().values / np.where(mt > 0, mt, np.nan)
    r12 = np.full(n, np.nan); r12[12:] = cl[12:] / cl[:-12] - 1.0
    stoph = np.zeros(n, dtype=bool)
    for k in range(25, n - 1):
        r = np.nanmin(lo[k - 24:k])
        if not np.isfinite(r) or r <= 0:
            continue
        if lo[k] < r * 0.99 and cl[k + 1] > r:
            stoph[k + 1:min(k + 73, n)] = True
    fin = np.isfinite(r12) & np.isfinite(trd_r6) & np.isfinite(vol_r6)
    sigs = {
        '吸筹信号(笔数×5&量×3)': fin & (trd_r6 > 5) & (vol_r6 > 3),
        '动量信号(1h涨≥5%)': fin & (r12 >= 0.05),
        '吸筹+假跌破': fin & (trd_r6 > 5) & (vol_r6 > 3) & stoph,
        '吸筹信号&未拉完(r12≤30%)': fin & (trd_r6 > 5) & (vol_r6 > 3) & (r12 > 0) & (r12 <= 0.30),
    }
    for lbl, m in sigs.items():
        r = stats(cl, hi, lo, m)
        if len(r):
            agg.setdefault(lbl, []).append(r)

print('\n' + '=' * 108)
print('【信号触发时：价格已经涨了多少（滞后量） vs 之后还剩多少空间】')
print('=' * 108)
hdr = '  %-28s %7s | %s | %s' % ('策略', '笔数',
                                 ' '.join('%8s' % ('已涨%dh' % (w * 5 / 60)) for w in WS),
                                 ' '.join('%8s' % ('+%dmin' % (f * 5)) for f in FWD) + ' %10s' % '未来24h最高')
print(hdr)
for lbl in sorted(agg):
    r = np.concatenate(agg[lbl])
    med = np.nanmedian(r, axis=0)
    print('  %-28s %7d | %s | %s %+10.2f%%'
          % (lbl, len(r),
             ' '.join('%+7.2f%%' % (100 * med[i]) for i in range(len(WS))),
             ' '.join('%+7.2f%%' % (100 * med[len(WS) + i]) for i in range(len(FWD))),
             100 * med[-1]))
print('\n  判读：左半「已涨」中位 ≈0% ⇒ 真提前；>5% ⇒ 滞后')
print('TOTAL %.0fs' % (time.time() - t0))
