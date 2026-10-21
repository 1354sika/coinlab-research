# -*- coding: utf-8 -*-
"""_probe_two.py — 两件事一起跑（中等样本 ~24 币）：
A 止损猎杀：局启动前 6h 内是否出现「假跌破再收回」
B 涨前微观异动：启动前 1/3/6/12 根（5~60 分钟）的
     · 主动买占比 taker_base/volume   （庄家在吃卖单吗）
     · 单笔均额 quote_volume/trades   （是不是大单）
     · 成交量 / 笔数                  （有没有突然放量）
   用来检验「吸筹存在、但在极短时间内结束」这个假设 —— 24h 窗口会把它平均掉。
"""
import os, time
import numpy as np, pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
t0 = time.time()
COLS = ['symbol', 'open_time', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_base']
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=COLS, nrows=5000000)
df = df.sort_values(['symbol', 'open_time'])
print('rows %d  coins %d  %.1fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

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
df['tbr'] = (df['taker_base'] / df['volume'].replace(0, np.nan)).astype('float32')
df['ats'] = (df['quote_volume'] / df['trades'].replace(0, np.nan)).astype('float32')
ev_by = {s: g for s, g in ev.groupby('symbol')}

BK, REF, DEP, HOLD = 72, 24, 0.99, 1
WS = [1, 3, 6, 12]

def scan(lo, cl, a):
    hits = []
    for k in range(max(REF + 1, a - BK), a - HOLD):
        r = np.nanmin(lo[k - REF:k])
        if not np.isfinite(r) or r <= 0:
            continue
        if lo[k] < r * DEP and cl[k + HOLD] > r:
            hits.append((a - k, lo[k] / r - 1.0))
    return hits

rows, rnds, mev, mrnd = [], [], [], []
rs = np.random.RandomState(7)
for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    ot = gg['open_time'].values.astype('int64')
    lo = gg['low'].values.astype('float64'); cl = gg['close'].values.astype('float64')
    tbr = gg['tbr'].values.astype('float64'); ats = gg['ats'].values.astype('float64')
    vol = gg['volume'].values.astype('float64'); trd = gg['trades'].values.astype('float64')
    n = len(gg)
    if n < 1000:
        continue
    b_tbr = np.nanmedian(tbr); b_ats = np.nanmedian(ats); b_vol = np.nanmedian(vol); b_trd = np.nanmedian(trd)
    if not (b_ats > 0):
        continue
    sub = ev_by.get(s)
    if sub is not None:
        for _, e in sub.iterrows():
            a = int(np.searchsorted(ot, e['t0'])); b = int(np.searchsorted(ot, e['t1']))
            if a < BK + 80 or b >= n - 2 or b <= a:
                continue
            h = scan(lo, cl, a)
            rows.append((s, float(e['gain']), 1 if h else 0, (h[0][0] if h else np.nan), (h[0][1] if h else np.nan)))
            for W in WS:
                sl = slice(a - W, a)
                mev.append((W, np.nanmean(tbr[sl]), np.nanmedian(ats[sl]) / b_ats,
                            np.nanmedian(vol[sl]) / b_vol, np.nanmedian(trd[sl]) / b_trd))
    for a in rs.randint(BK + 80, n - 2, size=min(300, max(1, n // 3000))):
        h = scan(lo, cl, int(a)); rnds.append((s, 1 if h else 0))
    for a in rs.randint(BK + 80, n - 2, size=min(200, max(1, n // 4000))):
        for W in WS:
            sl = slice(a - W, a)
            if a - W < 0: continue
            mrnd.append((W, np.nanmean(tbr[sl]), np.nanmedian(ats[sl]) / b_ats,
                         np.nanmedian(vol[sl]) / b_vol, np.nanmedian(trd[sl]) / b_trd))

R = pd.DataFrame(rows, columns=['symbol', 'gain', 'has', 'ahead', 'depth'])
Rr = pd.DataFrame(rnds, columns=['symbol', 'has'])
ME = pd.DataFrame(mev, columns=['W', 'tbr', 'ats_ratio', 'vol_ratio', 'trd_ratio'])
MR = pd.DataFrame(mrnd, columns=['W', 'tbr', 'ats_ratio', 'vol_ratio', 'trd_ratio'])
R.to_csv(os.path.join(E, 'gambler_stophunt_mid.csv'), index=False, encoding='utf-8-sig')
print('事件 %d ｜ 随机 %d ｜ %.0fs' % (len(R), len(Rr), time.time() - t0), flush=True)

print('\n' + '=' * 96)
print('【A 止损猎杀】局启动前 6h 内的「假跌破再收回」')
print('=' * 96)
if len(R):
    print('  事件组 有假跌破 %4d/%4d = %.1f%%' % (R['has'].sum(), len(R), 100 * R['has'].mean()))
    print('  随机组 有假跌破 %4d/%4d = %.1f%%' % (Rr['has'].sum(), len(Rr), 100 * Rr['has'].mean()))
    lift = R['has'].mean() / max(Rr['has'].mean(), 1e-9)
    print('  ⇒ 提升 %.2f×  %s' % (lift, '★显著' if lift > 1.3 else ('接近随机' if lift > 0.8 else '更少')))
    h = R[R['has'] == 1]
    if len(h):
        print('  距启动：中位 %.1f 根（%.1fh）｜ 深度中位 %+.2f%%'
              % (h['ahead'].median(), h['ahead'].median() * 5 / 60, 100 * h['depth'].median()))

print('\n' + '=' * 96)
print('【B 涨前微观异动】启动前 W 根 vs 随机时点（检验「吸筹极短」假设）')
print('=' * 96)
print('  %-5s %-9s | %-11s %-11s %-11s %-11s' % ('W(根)', '时长', '主动买占比', '单笔均额比', '量比', '笔数比'))
for W in WS:
    a = ME[ME['W'] == W]; b = MR[MR['W'] == W]
    if not len(a) or not len(b):
        continue
    print('  事件 W=%-2d %-7s | %-11.4f %-11.2f %-11.2f %-11.2f'
          % (W, '%d分' % (W * 5), a['tbr'].mean(), a['ats_ratio'].median(), a['vol_ratio'].median(), a['trd_ratio'].median()))
    print('  随机 W=%-2d %-7s | %-11.4f %-11.2f %-11.2f %-11.2f'
          % (W, '%d分' % (W * 5), b['tbr'].mean(), b['ats_ratio'].median(), b['vol_ratio'].median(), b['trd_ratio'].median()))
print('  （主动买占比 = taker_base/volume；0.5 = 买卖均衡；>0.5 = 主动买占优）')
print('\nTOTAL %.0fs' % (time.time() - t0))
