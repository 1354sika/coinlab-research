# -*- coding: utf-8 -*-
"""analyze_gap_dedup.py — 每 5 分钟扫一遍时，去掉「同一波行情重复计数」后还剩多少真实机会

gap 实验发现：GAP=1 时 60,294 笔、单笔 +1.0761%，与 GAP=24 的 +1.0822% 几乎一样
——信号质量不随扫描频率下降。但 60,294 笔里有大量是同一波行情被连续 5 分钟各记一次。

本脚本：GAP=1 全扫，然后按「同币 N 小时内只留第一个」去重，看真实独立机会数。
去重窗：1h / 4h / 12h / 24h / 72h。

诚实边界：零滑点、成本 0.04% 单边、TP 用 close 触价、成交率未计。
"""
import numpy as np
import pandas as pd
import time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/gap_dedup_REAL.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR = -0.02, 3.0
WINS = [1, 4, 12, 24, 72]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

t0 = time.time()
log('=' * 96)
log(f'gap_dedup · 每 5 分钟全扫 + 同币去重   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 96)

df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
recs = []
tmin, tmax = None, None

for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H:
        continue
    if tmin is None:
        tmin, tmax = int(t[0]), int(t[-1])
    else:
        tmin = min(tmin, int(t[0])); tmax = max(tmax, int(t[-1]))
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    sig = (ret24 <= DIP) & (volr >= VOLR)
    sig[: W + 300] = False
    sig[n - H - 1:] = False
    idx = np.where(sig)[0]
    for i in idx:
        r = c[i:i + H] / c[i] - 1.0
        hm = r >= 0.02
        if hm.any():
            j = int(np.argmax(hm)); net = float(r[j]) - COST
        else:
            eh = h[i:i + H] / c[i] - 1.0
            el = l[i:i + H] / c[i] - 1.0
            hi = 0.0; k = H - 1
            for jj in range(H):
                if eh[jj] > hi:
                    hi = eh[jj]
                if jj >= 1 and (1.0 + el[jj]) / (1.0 + hi) - 1.0 <= -0.01:
                    k = jj; break
            net = float(r[k]) - COST
        recs.append((int(t[i]), sym, net))

del df
R = pd.DataFrame(recs, columns=['t', 'sym', 'net']).sort_values('t').reset_index(drop=True)
span_y = (tmax - tmin) / (365.25 * 86400 * 1000)
log(f'全扫信号 {len(R)}  币 {R.sym.nunique()}  跨度 {span_y:.2f} 年  耗时 {time.time()-t0:.0f}s')
log('')

log('=' * 96)
log('去重：同一币 N 小时内只留第一个入场')
log('=' * 96)
log(f'{"去重窗":>8} {"笔数":>9} {"笔/天":>8} {"单笔均值":>10} {"中位":>9} {"胜率":>8} {"不复利年化":>11}')
for win in [0] + WINS:
    if win == 0:
        S = R
        tag = '不去重'
    else:
        last = {}
        keep = []
        for row in R.itertuples(index=False):
            if row.sym in last and row.t - last[row.sym] < win * 3600_000:
                continue
            last[row.sym] = row.t
            keep.append((row.t, row.sym, row.net))
        S = pd.DataFrame(keep, columns=['t', 'sym', 'net'])
        tag = f'{win}h'
    a = S.net.to_numpy()
    per_day = len(a) / (span_y * 365.25)
    ann = a.mean() * per_day * 365.25
    log(f'{tag:>8} {len(a):>9} {per_day:>7.2f} {a.mean():>9.4%} '
        f'{np.median(a):>8.4%} {(a>0).mean():>7.2%} {ann*100:>10.1f}%')
log('')
log('读法：去重后的笔数 = 真实独立机会数的上界（仍需按持仓时间再打一次折）。')
log('')
log('诚实边界：')
log('  1. 零滑点；成本 0.04% 单边；TP 用 close 触价，成交率未计。')
log('  2. 去重只按「同币 + 时间窗」，不同币的同步行情仍算独立。')
log('  3. 「不复利年化」未考虑资金分份；实际同时持仓 = 笔/天 × 持仓小时 / 24。')
log('  4. 全样本 3 年，含样本内，未做时间切分外推。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {OUT}  TOTAL {time.time()-t0:.0f}s')
