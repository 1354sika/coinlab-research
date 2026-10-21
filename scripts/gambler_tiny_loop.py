# -*- coding: utf-8 -*-
"""gambler_tiny_loop.py — 把「极短吸筹」信号做成特征，进【无前视闭环】
特征（每一刻只用「过去」）：
  trd_r6 = 过去 6 根(30min) 笔数中位 / 该币 288 根中位笔数  ← 吸筹的核心证据（笔数暴增）
  vol_r6 = 过去 6 根量中位 / 288 根中位
  tbr6   = 过去 6 根主动买占比均值
  stoph  = 过去 6h(72根) 内出现过「假跌破再收回」            ← 洗盘标记
  r12    = 过去 12 根(1h) 涨幅                                ← 要求「刚启动、还没拉完」
入场（无前视）：trd_r6 > 5 且 vol_r6 > 3 且 0 < r12 ≤ 30%
出场：+50% 止盈 / −15% 止损 / 24h 超时（扣 maker 双边 0.04%）
对照：旧规则（仅 1h 涨≥5%）｜ 新信号+假跌破 ｜ 随机
"""
import os, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
NR = int(os.environ.get('NR', '5000000'))
t0 = time.time()
COLS = ['symbol', 'open_time', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_base']
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=COLS, nrows=NR)
df = df.sort_values(['symbol', 'open_time'])
df['tbr'] = df['taker_base'] / df['volume'].replace(0, np.nan)
print('rows %d  coins %d  %.1fs' % (len(df), df.symbol.nunique(), time.time() - t0), flush=True)

FS = 0.0004      # maker 双边 0.02% × 2
K = 288          # 24h 持有上限

def sim(cl, hi, lo, mask):
    rs = []
    idx = np.where(mask)[0][::6]
    for t in idx:
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
        rs.append(r - FS)
    return np.array(rs)

agg = {}
rnd_all = []
for s, g in df.groupby('symbol', sort=False):
    gg = g.reset_index(drop=True)
    cl = gg['close'].values.astype('float64'); hi = gg['high'].values.astype('float64'); lo = gg['low'].values.astype('float64')
    vol = gg['volume'].values.astype('float64'); trd = gg['trades'].values.astype('float64'); tbr = gg['tbr'].values.astype('float64')
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
    sig_new = fin & (trd_r6 > 5) & (vol_r6 > 3) & (r12 > 0) & (r12 <= 0.30)
    sig_old = fin & (r12 >= 0.05)
    sig_stp = sig_new & stoph
    for lbl, m in [('A 新信号(笔数×5&量×3&未拉完)', sig_new), ('B 旧规则(1h涨≥5%)', sig_old), ('C 新信号+6h内假跌破', sig_stp)]:
        r = sim(cl, hi, lo, m)
        if len(r):
            agg.setdefault(lbl, []).append(r)
    rnd = np.zeros(n, dtype=bool)
    rs = np.random.RandomState(abs(hash(s)) % 2 ** 31)
    rnd[rs.randint(300, n - K, size=min(300, n // 3000))] = True
    rr = sim(cl, hi, lo, rnd)
    if len(rr):
        rnd_all.append(rr)

print('\n' + '=' * 104)
print('【无前视闭环：期望净收益（已扣 maker 双边 0.04%%）】  NR=%d' % NR)
print('=' * 104)
print('  %-30s %8s %10s %10s %9s %9s %9s' % ('策略', '笔数', '期望净%', '中位%', '胜率%', '止盈%', '止损%'))
for lbl in sorted(agg):
    r = np.concatenate(agg[lbl])
    print('  %-30s %8d %+10.3f %+10.2f %9.1f %9.1f %9.1f'
          % (lbl, len(r), 100 * r.mean(), 100 * np.median(r), 100 * (r > 0).mean(),
             100 * (r >= 0.5).mean(), 100 * (r <= -0.15).mean()))
if rnd_all:
    r = np.concatenate(rnd_all)
    print('  %-30s %8d %+10.3f %+10.2f %9.1f %9.1f %9.1f'
          % ('D 随机入场', len(r), 100 * r.mean(), 100 * np.median(r), 100 * (r > 0).mean(),
             100 * (r >= 0.5).mean(), 100 * (r <= -0.15).mean()))
print('\nTOTAL %.0fs' % (time.time() - t0))
