# -*- coding: utf-8 -*-
"""gambler_stophunt.py — 止损猎杀（假跌破再拉）检测
定义：在局启动前的 6 小时（72 根）窗口内，是否出现
      「low[k] 跌破前 2 小时最低点 × 0.99，且 1 根内收盘收回该低点之上」
      —— 这就是经典的"扫止损 / 洗盘"，庄家拉之前先把浮筹洗掉。
对照：同币随机时刻做同样检测（同数量级）。
输出：有/无假跌破的比例差、假跌破距启动的根数、深度、以及有/无两组后续表现。
"""
import os, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
t0 = time.time()

df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=['symbol', 'open_time', 'high', 'low', 'close'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %d  %.0fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

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

rows, rnd_rows = [], []
rs = np.random.RandomState(7)

for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    ot = gg['open_time'].values.astype('int64')
    lo = gg['low'].values.astype('float64'); cl = gg['close'].values.astype('float64')
    hi = gg['high'].values.astype('float64')
    n = len(gg)
    if n < 1000:
        continue
    sub = ev[ev['symbol'] == s]
    for _, e in sub.iterrows():
        a = int(np.searchsorted(ot, e['t0'])); b = int(np.searchsorted(ot, e['t1']))
        if a < BK + 2 or b >= n - 2 or b <= a:
            continue
        h = scan(lo, cl, a)
        pk = hi[b]
        e2 = min(b + 289, n)
        post = (np.nanmax(hi[b + 1:e2]) / pk - 1.0) if e2 > b + 1 and pk > 0 else np.nan
        dd = (np.nanmin(cl[b:e2]) / pk - 1.0) if pk > 0 else np.nan
        rows.append((s, float(e['gain']), 1 if h else 0, (h[0][0] if h else np.nan),
                     (h[0][1] if h else np.nan), len(h), post, dd))
    for a in rs.randint(BK + 2, n - 2, size=min(80, max(1, n // 4000))):
        h = scan(lo, cl, int(a))
        rnd_rows.append((s, 1 if h else 0, (h[0][0] if h else np.nan),
                         (h[0][1] if h else np.nan), len(h)))

R = pd.DataFrame(rows, columns=['symbol', 'gain', 'has', 'ahead', 'depth', 'nhits', 'post_max', 'dd24'])
Rr = pd.DataFrame(rnd_rows, columns=['symbol', 'has', 'ahead', 'depth', 'nhits'])
R.to_csv(os.path.join(E, 'gambler_stophunt.csv'), index=False, encoding='utf-8-sig')
print('事件可对齐 %d 个 ｜ 随机对照 %d 个 ｜ %.0fs' % (len(R), len(Rr), time.time() - t0), flush=True)

print('\n' + '=' * 96)
print('【A 拉升前是否出现「假跌破再收回」（扫止损）】')
print('=' * 96)
print('  事件组 n=%5d  有假跌破 %4d（%.1f%%）' % (len(R), R['has'].sum(), 100 * R['has'].mean()))
print('  随机组 n=%5d  有假跌破 %4d（%.1f%%）' % (len(Rr), Rr['has'].sum(), 100 * Rr['has'].mean()))
lift = R['has'].mean() / max(Rr['has'].mean(), 1e-9)
print('  ⇒ 提升倍数 = %.2f×  %s' % (lift, '★显著' if lift > 1.3 else ('接近随机' if lift > 0.85 else '反而更少')))

h = R[R['has'] == 1]
if len(h):
    print('\n【B 假跌破距启动多远】')
    print('  中位 %5.1f 根前（%.1f 小时）｜ 均值 %.1f 根｜ p25 %.0f｜ p75 %.0f'
          % (h['ahead'].median(), h['ahead'].median() * 5 / 60, h['ahead'].mean(),
             h['ahead'].quantile(.25), h['ahead'].quantile(.75)))
    for lo_, hi_ in [(0, 6), (6, 12), (12, 24), (24, 48), (48, 73)]:
        m = (h['ahead'] > lo_) & (h['ahead'] <= hi_)
        if m.sum():
            print('    %2d~%2d 根前（%.1f~%.1f h）: %4d（%.1f%%）%s'
                  % (lo_, hi_, lo_ * 5 / 60, hi_ * 5 / 60, m.sum(), 100 * m.sum() / len(h),
                     '#' * int(40 * m.sum() / len(h))))
    print('\n【C 假跌破深度】中位 %+.2f%%｜均值 %+.2f%%｜最深 %+.2f%%'
          % (100 * h['depth'].median(), 100 * h['depth'].mean(), 100 * h['depth'].min()))

print('\n' + '=' * 96)
print('【D 有/无假跌破两组的后续表现】')
print('=' * 96)
for lbl, sub in [('有假跌破', R[R['has'] == 1]), ('无假跌破', R[R['has'] == 0])]:
    if not len(sub):
        continue
    print('  %-8s n=%4d  峰值涨幅中位 %5.2f 倍  峰后 24h 最高 %+6.2f%%  峰后 24h 最大回撤 %+7.2f%%'
          % (lbl, len(sub), sub['gain'].median(), 100 * sub['post_max'].median(), 100 * sub['dd24'].median()))

print('\nTOTAL %.0fs' % (time.time() - t0))
