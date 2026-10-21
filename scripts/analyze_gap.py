# -*- coding: utf-8 -*-
"""analyze_gap.py — 扫描间隔决定笔数：一天 100 笔时，每笔还能赚 1.08% 吗？

背景：全项目最强规则在「每 2 小时扫一次」口径下是 972 笔/年、单笔 +1.08%。
但实盘 live 数据（每 5 分钟扫）显示约 100 笔/天 = 36,500 笔/年。
多出来的信号是否同样赚钱？本脚本用同一批历史 5m 数据，只改扫描间隔 GAP 来测。

GAP（5m 根数）：1=每 5 分钟、4=每 20 分钟、12=每 1 小时、24=每 2 小时、48=每 4 小时。

规则不变：入场 ret24<=-2% 且 volr>=3.0；出场 首次>=+2% 卖，否则 θ=1% 兜底；成本 0.04% 单边。

诚实边界：GAP 越小样本重叠越严重（同一波行情重复计数）；
          零滑点、TP 用 close 触价、成交率未计。
"""
import numpy as np
import pandas as pd
import time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/gap_REAL.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR = -0.02, 3.0
GAPS = [1, 4, 12, 24, 48]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

t0 = time.time()
log('=' * 96)
log(f'gap · 扫描间隔 vs 笔数 vs 单笔期望   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 96)
log('入场 ret24<=-2% 且 volr>=3.0；出场 首次>=+2% 卖，否则 θ=1% 兜底；成本 0.04% 单边')
log('')

df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])

buckets = {g: [] for g in GAPS}
tmin = None; tmax = None

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
    for GAP in GAPS:
        pos = np.arange(W + 300, n - H - 1, GAP)
        sel = pos[sig[pos]]
        for i in sel:
            r = c[i:i + H] / c[i] - 1.0
            hm = r >= 0.02
            if hm.any():
                j = int(np.argmax(hm))
                net = float(r[j]) - COST
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
            buckets[GAP].append(net)

del df
span_y = (tmax - tmin) / (365.25 * 86400 * 1000)
log(f'计算完成  耗时 {time.time()-t0:.0f}s   数据跨度 {span_y:.2f} 年')
log('')

log('=' * 96)
log('结果：同一规则、只改扫描间隔')
log('=' * 96)
log(f'{"GAP":>6} {"扫描频率":>10} {"笔数":>9} {"笔/天":>8} {"单笔均值":>10} {"中位":>9} {"胜率":>8} {"不复利年化":>11}')
rows = []
for GAP in GAPS:
    a = np.array(buckets[GAP], dtype=np.float64)
    if len(a) == 0:
        continue
    per_day = len(a) / (span_y * 365.25)
    ann = a.mean() * per_day * 365.25
    freq = f'每{GAP*5}分' if GAP * 5 < 60 else f'每{GAP*5//60}时'
    log(f'{GAP:>6} {freq:>10} {len(a):>9} {per_day:>7.2f} {a.mean():>9.4%} '
        f'{np.median(a):>8.4%} {(a>0).mean():>7.2%} {ann*100:>10.1f}%')
    rows.append((GAP, len(a), per_day, a.mean(), np.median(a), (a > 0).mean(), ann))
log('')

log('=' * 96)
log('读法')
log('=' * 96)
log('  若「单笔均值」随 GAP 变小而明显下降 => 多出来的信号是弱信号，不能全做；')
log('  若基本不变 => 信号密度可自由提高，机器可全量执行。')
log('')
log('诚实边界：')
log('  1. GAP=1 时同一波行情被重复计数多次（样本重叠最严重），笔数不代表独立机会数。')
log('  2. 零滑点；成本 0.04% 单边；TP 用 close 触价，成交率未计。')
log('  3. 「不复利年化」= 单笔均值 × 笔/年，未考虑资金分份（同时持仓占用）。')
log('  4. 全样本 3 年，含样本内，未做时间切分外推。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {OUT}  TOTAL {time.time()-t0:.0f}s')
