# -*- coding: utf-8 -*-
"""gambler_slip.py — 滑点敏感性：正期望扛不扛得住真实滑点
同一组信号，测「单边滑点」0 / 0.2% / 0.5% / 1% / 2% / 3% / 5% 下的期望净收益。
（扣 maker 双边 0.04% 之外的额外滑点，买入按 +slip、卖出按 −slip）
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
df['tbr'] = df['taker_base'] / df['volume'].replace(0, np.nan)
print('rows %d  coins %d  %.1fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

FEE = 0.0004
K = 288
SLIPS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]

def gross(cl, hi, lo, mask):
    rs = []
    for t in np.where(mask)[0][::6]:
        if t + K >= len(cl):
            continue
        e = cl[t]
        if not np.isfinite(e) or e <= 0:
            continue
        sh = hi[t + 1:t + 1 + K]; sl = lo[t + 1:t + 1 + K]
        tp = np.where(sh >= e * 1.5)[0]; sx = np.where(sl <= e * 0.85)[0]
        it = tp[0] if len(tp) else 10 ** 9
        isx = sx[0] if len(sx) else 10 ** 9
        if it == isx == 10 ** 9:
            r = cl[t + K] / e - 1.0
        elif it < isx:
            r = 0.50
        else:
            r = -0.15
        rs.append(r)
    return np.array(rs)

agg = {}
for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    cl = gg['close'].values.astype('float64'); hi = gg['high'].values.astype('float64'); lo = gg['low'].values.astype('float64')
    vol = gg['volume'].values.astype('float64'); trd = gg['trades'].values.astype('float64')
    n = len(gg)
    if n < 1200:
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
        'A 新信号(笔数×5&量×3)': fin & (trd_r6 > 5) & (vol_r6 > 3) & (r12 > 0) & (r12 <= 0.30),
        'B 旧规则(1h涨≥5%)': fin & (r12 >= 0.05),
        'C 新信号+假跌破': fin & (trd_r6 > 5) & (vol_r6 > 3) & (r12 > 0) & (r12 <= 0.30) & stoph,
    }
    for lbl, m in sigs.items():
        r = gross(cl, hi, lo, m)
        if len(r):
            agg.setdefault(lbl, []).append(r)

print('\n' + '=' * 104)
print('【滑点敏感性】同一组信号，不同单边滑点下的期望净收益（%%/笔，已扣 maker 0.04%%）')
print('=' * 104)
print('  %-24s %7s | %s' % ('策略', '笔数', ' '.join('%8s' % ('滑点%.1f%%' % (100 * s)) for s in SLIPS)))
for lbl in sorted(agg):
    r = np.concatenate(agg[lbl])
    cells = []
    for s in SLIPS:
        net = r - FEE - 2 * s
        cells.append('%+8.3f' % (100 * net.mean()))
    print('  %-24s %7d | %s' % (lbl, len(r), ' '.join(cells)))
print('\n  正期望能否存活：看哪一档滑点后期望翻负')
print('\nTOTAL %.0fs' % (time.time() - t0))
