# -*- coding: utf-8 -*-
"""gambler_nowin.py — 验证「4h 窗口是我们自己捏出来的，不是市场的性质」
三个检验：
  A 多窗口对照：窗口限 W ∈ {1h,2h,4h,8h,16h,24h,48h,96h} 下，能扫出多少个「涨≥+50%」的局
                （若事件数在 4h 附近饱和 → 4h 是真的；若一直单调涨 → 4h 是人为上限）
  B 无窗口版：不设时长上限，对每个点找「未来首次达到 +50%」需要多久 → 真实时长分布
  C 达标的等待时长分位（用来看"真正的时间尺度"是几小时还是几天）
"""
import os, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'
t0 = time.time()
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=['symbol', 'open_time', 'high', 'low', 'close'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %d   %.0fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

WINS = [('1h', 12), ('2h', 24), ('4h', 48), ('8h', 96), ('16h', 192), ('24h', 288), ('48h', 576), ('96h', 1152), ('7d', 2016)]
rowsA = []
lens_all = []
uniq_intervals = []

for s, g in df.groupby('symbol', sort=False):
    hv = g['high'].values.astype('float64')
    cl = g['close'].values.astype('float64')
    n = len(g)
    if n < 2000:
        continue
    for lbl, W in WINS:
        # 未来 W 根内的最高价（含当根）
        rm = pd.Series(hv).rolling(W, min_periods=1).max().shift(-(W - 1)).values
        hit = (rm / np.where(cl > 0, cl, np.nan) - 1.0) >= 0.5
        hit = np.nan_to_num(hit, nan=0).astype(bool)
        # 去重叠：连续 True 段只算 1 个事件，并记录段内"最快达标"的时长
        d = np.diff(hit.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1) + ([0] if hit[0] else [])
        ev_n = len(starts)
        rowsA.append((s, lbl, W, ev_n))
        if lbl in ('4h', '24h'):
            for a in starts[:40000]:
                # 段内找最早达标的那一根
                end = min(a + W, n)
                seg = hv[a:end] / (cl[a] if cl[a] > 0 else np.nan) - 1.0
                ok = np.where(seg >= 0.5)[0]
                if len(ok):
                    uniq_intervals.append((lbl, (ok[0] + 1) * 5))

A = pd.DataFrame(rowsA, columns=['symbol', 'win', 'W', 'events'])
print('\n' + '=' * 96)
print('【A 多窗口对照】同一份数据，只改「允许的最大时长」，能扫出多少个 +50% 的局')
print('=' * 96)
print('  %-8s %10s %12s %14s' % ('窗口限', '事件总数', '有局的币数', '相对 1h 的倍数'))
base = A[A['win'] == '1h']['events'].sum()
for lbl, W in WINS:
    sub = A[A['win'] == lbl]
    tot = int(sub['events'].sum()); nz = int((sub['events'] > 0).sum())
    print('  %-8s %10d %12d %13.2fx' % (lbl, tot, nz, tot / max(base, 1)))
print('  ⇒ 若事件数随窗口一路单调涨到 96h，说明「4h」不是市场的自然尺度，是我们设的上限')

print('\n' + '=' * 96)
print('【B 无窗口：首次达到 +50% 需要多久】（每根 bar 起算，去重叠后的事件）')
print('=' * 96)
for lbl in ('4h', '24h'):
    v = np.array([x[1] for x in uniq_intervals if x[0] == lbl], dtype='float64')
    if not len(v):
        continue
    v = v / 60.0   # 分钟 → 小时
    print('  [%s 窗口内的事件] n=%d' % (lbl, len(v)))
    print('     中位 %6.2f h ｜ 均值 %6.2f h ｜ p25 %6.2f ｜ p75 %6.2f ｜ p90 %6.2f ｜ 最大 %6.2f h'
          % (np.median(v), v.mean(), np.percentile(v, 25), np.percentile(v, 75), np.percentile(v, 90), v.max()))
    for lo_, hi_ in [(0, .5), (.5, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 24), (24, 999)]:
        m = (v > lo_) & (v <= hi_)
        if m.sum():
            print('     %4.1f~%4.1f h: %6d (%.1f%%) %s' % (lo_, hi_, m.sum(), 100 * m.sum() / len(v), '#' * int(40 * m.sum() / len(v))))
print('\nTOTAL %.0fs' % (time.time() - t0))
