# -*- coding: utf-8 -*-
"""_probe_nowin_fast.py — 抄近路：只读前 120 万行（约 6 个币），秒级看「窗口改一改，事件数怎么变」
若事件数随窗口一路单调涨到 96h ⇒ 「4h」是我们设的上限，不是市场的尺度
"""
import os, time
import numpy as np, pandas as pd

D = r'D:\中信证券\coinlab\data'
t0 = time.time()
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'),
                 usecols=['symbol', 'open_time', 'high', 'low', 'close'], nrows=1200000)
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %d  %s  %.1fs' % (len(df), df.symbol.nunique(), list(df.symbol.unique()), time.time() - t0))

WINS = [('1h', 12), ('2h', 24), ('4h', 48), ('8h', 96), ('16h', 192), ('24h', 288), ('48h', 576), ('96h', 1152)]
print('\n%-14s %8s | %s' % ('币', 'bars', ' '.join('%7s' % w[0] for w in WINS)))
tot = np.zeros(len(WINS), dtype=int)
for s, g in df.groupby('symbol', sort=False):
    hv = g['high'].values.astype('float64'); cl = g['close'].values.astype('float64'); n = len(g)
    if n < 3000:
        continue
    cnt = []
    for lbl, W in WINS:
        rm = pd.Series(hv).rolling(W, min_periods=1).max().shift(-(W - 1)).values
        hit = np.nan_to_num((rm / np.where(cl > 0, cl, np.nan) - 1.0) >= 0.5, nan=0).astype(bool)
        d = np.diff(hit.astype(np.int8))
        starts = (np.where(d == 1)[0] + 1).tolist()
        if len(hit) and hit[0]:
            starts = [0] + starts
        cnt.append(len(starts))
    cnt = np.array(cnt); tot += cnt
    print('%-14s %8d | %s' % (s, n, ' '.join('%7d' % c for c in cnt)))
print('%-14s %8s | %s' % ('【合计】', '', ' '.join('%7d' % c for c in tot)))
print('%-14s %8s | %s' % ('相对 1h', '', ' '.join('%6.2fx' % (c / max(tot[0], 1)) for c in tot)))

# 无窗口：首次达 +50% 的等待时长
print('\n【无窗口】从每根 bar 起算，首次达到 +50% 需要多久（小时）')
allv = []
for s, g in df.groupby('symbol', sort=False):
    hv = g['high'].values.astype('float64'); cl = g['close'].values.astype('float64'); n = len(g)
    if n < 3000:
        continue
    lim = min(n, 1152 + 12)          # 只看未来 96h 内能否达标
    for i in range(0, n - lim, 24):  # 每 2h 取一个起点，省时间
        seg = hv[i:i + lim] / (cl[i] if cl[i] > 0 else np.nan) - 1.0
        ok = np.where(seg >= 0.5)[0]
        if len(ok):
            allv.append((ok[0] + 1) * 5 / 60.0)
v = np.array(allv)
if len(v):
    print('  n=%d  中位 %.2f h | p25 %.2f | p75 %.2f | p90 %.2f | 最大 %.1f h'
          % (len(v), np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.percentile(v, 90), v.max()))
    for lo_, hi_ in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 24), (24, 96)]:
        m = (v > lo_) & (v <= hi_)
        if m.sum():
            print('     %4.1f~%4.1f h: %6d (%.1f%%) %s' % (lo_, hi_, m.sum(), 100 * m.sum() / len(v), '#' * int(45 * m.sum() / len(v))))
print('\nTOTAL %.1fs' % (time.time() - t0))
