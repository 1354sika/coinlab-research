# -*- coding: utf-8 -*-
"""gambler_type1_cycle · 统计「1型小周期」有多少个

用户定义（2026-10-18）：买入卖出能赚本金 5% 以上、且周期时长 ≤ 24h 的行情段。

两种口径：
  C 理论口径：买在局部低点、卖在之后 24h 内最高点，涨幅 >= 5%
              （买不到，只作上界参考）
  A 可交易口径：低点之后涨 3% 才入场，之后 24h 内最高价 >= 入场价 x 1.05
              （与 tpsl_grid 的 TP=5% TP率一致）

低点定义：过去 240 分钟（V05=4h）最低；相邻低点至少间隔 30 根。
周期去重：找到一个周期后，从它的高点之后继续找，不重叠。
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
VW = 240          # 低点窗口 4h
HOLD = 1440       # 24h
MIN_GAIN = 0.05   # 5%
CONF = 2880       # 入场确认窗口（同大周期）
MAXSYM = int(os.environ.get('SYMS', '60'))

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_type1_cycle · 1型小周期计数（+5% 以上、≤24h）')
log('=' * 100)
log(f'低点窗口 {VW}min  持有上限 {HOLD}min  最小涨幅 {MIN_GAIN:.0%}')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['high'].values.astype(float), g['low'].values.astype(float),
                 g['close'].values.astype(float), g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')

# ---------- 口径 C：买低点、卖高点 ----------
C = []   # (sym, t_lo, t_hi, gain, dur_min)
# ---------- 口径 A：低点+3% 入场，24h 内摸到 +5% ----------
A = []   # (sym, t_entry, t_hit, gain_from_entry, dur_min)

for sym, (h, l, c, t) in RAWS.items():
    n = len(c)
    lo_w = pd.Series(l).rolling(VW, min_periods=1).min().values
    i = VW - 1
    while i < n - 1:
        if l[i] <= lo_w[i]:
            # ---- 口径 C ----
            end = min(i + HOLD + 1, n)
            seg = h[i:end]
            k = int(np.argmax(seg))
            hi = seg[k]
            gain_c = hi / l[i] - 1.0
            if gain_c >= MIN_GAIN:
                C.append((sym, int(t[i]), int(t[i + k]), float(gain_c), int(k)))
                i = i + k + 1
                continue

            # ---- 口径 A ----
            entry_j = -1
            for j in range(i + 1, min(i + CONF + 1, n)):
                if c[j] >= l[i] * 1.03:
                    entry_j = j; break
            if entry_j > 0 and entry_j + 1 < n:
                end2 = min(entry_j + HOLD + 1, n)
                seg2 = h[entry_j:end2]
                k2 = int(np.argmax(seg2))
                hi2 = seg2[k2]
                gain_a = hi2 / c[entry_j] - 1.0
                if gain_a >= MIN_GAIN:
                    A.append((sym, int(t[entry_j]), int(t[entry_j + k2]), float(gain_a), int(k2)))
            i += 30
        else:
            i += 1
    if len(RAWS) and sym == list(RAWS)[-1]:
        log(f'  已处理 {sym}  累计 C={len(C):,}  A={len(A):,}  {time.time()-t0:.0f}s')

all_t = np.concatenate([np.array([x[1] for x in C]), np.array([x[1] for x in A])])
t_min = int(all_t.min()); t_max = int(all_t.max())
days = (pd.Timestamp(t_max // 1000, unit='s', tz='UTC')
        - pd.Timestamp(t_min // 1000, unit='s', tz='UTC')).days + 1
log(f'数据跨度 {days} 天（{pd.Timestamp(t_min//1000, unit="s", tz="UTC").date()} ~ {pd.Timestamp(t_max//1000, unit="s", tz="UTC").date()}）')


def report(name, rows):
    log('')
    log('=' * 100)
    log(f'【{name}】')
    n = len(rows)
    if n == 0:
        log('  0 个'); return
    log(f'  周期总数 {n:,}')
    log(f'  每天 {n/days:.1f}   每周 {n/days*7:.0f}   每年 {n/days*365:,.0f}')
    gains = np.array([r[3] for r in rows])
    durs = np.array([r[4] for r in rows])
    qs = [0, 25, 50, 75, 90, 99, 100]
    log(f'  涨幅分位 %: ' + '  '.join(f'p{q}={np.percentile(gains, q)*100:.1f}' for q in qs))
    log(f'  时长分位 min: ' + '  '.join(f'p{q}={np.percentile(durs, q):.0f}' for q in qs))
    log('  --- 按涨幅分档 ---')
    for lo, hi, label in [(0.05, 0.10, '5~10%'), (0.10, 0.20, '10~20%'),
                          (0.20, 0.30, '20~30%'), (0.30, 0.50, '30~50%'),
                          (0.50, 1e9, '>50%')]:
        m = (gains >= lo) & (gains < hi)
        log(f'    {label:<8} {int(m.sum()):>8,}  ({m.mean()*100:>5.1f}%)  每天 {m.sum()/days:>7.1f}')
    log('  --- 按时长分档 ---')
    for lo, hi, label in [(0, 15, '<=15min'), (15, 60, '15~60min'),
                          (60, 240, '1~4h'), (240, 720, '4~12h'), (720, 9999, '12~24h')]:
        m = (durs >= lo) & (durs < hi)
        log(f'    {label:<10} {int(m.sum()):>8,}  ({m.mean()*100:>5.1f}%)  每天 {m.sum()/days:>7.1f}')


report('口径 C：买在低点、卖在 24h 内最高点（理论上界）', C)
report('口径 A：低点+3% 入场，24h 内摸到 +5%（可交易）', A)

log('')
log('=' * 100)
log('【按币统计 Top10（口径 C）】')
dfc = pd.DataFrame(C, columns=['sym', 't_lo', 't_hi', 'gain', 'dur'])
if len(dfc):
    g = dfc.groupby('sym').size().sort_values(ascending=False)
    log('  ' + '  '.join(f'{k}={v}' for k, v in g.head(10).items()))
    log(f'  币数 {len(g)}  每币中位 {g.median():.0f}  最少 {g.min()}  最多 {g.max()}')

log('')
log('【判读】')
log('  · 口径 C 是理论最大机会数；口径 A 是实际能抓到的')
log('  · 两者差距 = 「买不到最低点」的代价')
log('  · 这些周期里有多少能被现有信号（h3 等）提前识别，是下一个问题')

out = os.path.join(OUT, 'type1_cycle_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
pd.DataFrame(C, columns=['sym', 't_lo', 't_hi', 'gain', 'dur_min']).to_csv(
    os.path.join(OUT, 'type1_cycle_C.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(A, columns=['sym', 't_entry', 't_hit', 'gain', 'dur_min']).to_csv(
    os.path.join(OUT, 'type1_cycle_A.csv'), index=False, encoding='utf-8-sig')
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
