# -*- coding: utf-8 -*-
"""gambler_type1_catch · 1型周期（+5% / ≤24h）我们抓得住吗

输入：evidence/cycle/type1_cycle_A.csv（8,528 个可交易周期：低点+3% 入场，24h 内摸到 +5%）
做法：对每个止损档 SL，检查「从入场到首次触及 +5%」之间，最低价有没有先跌破 entry×(1−SL)
      —— 跌破了 = 这个周期被止损打掉，抓不到
输出：每档止损能抓住多少 / 打掉多少 / 净期望
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
TP = 0.05
MAXSYM = int(os.environ.get('SYMS', '60'))

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_type1_catch · 1型周期抓得住吗')
log('=' * 100)

dfa = pd.read_csv(os.path.join(OUT, 'type1_cycle_A.csv'))
log(f'周期数 {len(dfa):,}  币 {dfa["sym"].nunique()}')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close']
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
log(f'K线读取 {time.time()-t0:.0f}s')

SL_GRID = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
HOLD = 1440

stats = {sl: dict(n_catch=0, n_sl=0, n_timeout=0, ret_sum=0.0) for sl in SL_GRID}
no_kline = 0

for _, r in dfa.iterrows():
    sym = r['sym']
    if sym not in RAWS:
        no_kline += 1; continue
    h, l, c, t = RAWS[sym]
    n = len(c)
    i0 = int(np.searchsorted(t, int(r['t_entry'])))
    if i0 <= 0 or i0 >= n: no_kline += 1; continue
    entry = c[i0]
    tp_px = entry * (1.0 + TP)
    end = min(i0 + HOLD, n)

    # 首次触及 +5% 的位置
    seg_h = h[i0 + 1:end]
    hit = seg_h >= tp_px
    if not hit.any():
        continue
    i_tp = int(np.argmax(hit))  # 相对 i0+1
    i_tp_abs = i0 + 1 + i_tp

    for sl in SL_GRID:
        sl_px = entry * (1.0 - sl)
        seg_l = l[i0 + 1:i_tp_abs + 1]      # 到首次触及止盈之前（含当根，保守）
        if (seg_l <= sl_px).any():
            stats[sl]['n_sl'] += 1
            stats[sl]['ret_sum'] += -sl
        else:
            stats[sl]['n_catch'] += 1
            stats[sl]['ret_sum'] += TP

log(f'可用 K 线周期 {len(dfa)-no_kline:,}  无 K 线 {no_kline:,}  {time.time()-t0:.0f}s')

log('')
log('=' * 100)
log('【每档止损：抓住多少 / 打掉多少 / 扣费后每笔期望】')
log(f"{'止损':>7}{'抓住':>8}{'抓中率':>9}{'打掉':>8}{'净期望%':>10}")
best = None
for sl in SL_GRID:
    s = stats[sl]
    tot = s['n_catch'] + s['n_sl']
    if tot == 0: continue
    gross = s['ret_sum'] / tot
    net = gross - FEE
    log(f'{-sl:>7.1%}{s["n_catch"]:>8,}{s["n_catch"]/tot*100:>8.1f}%{s["n_sl"]:>8,}{net*100:>10.3f}')
    if best is None or net > best[1]:
        best = (sl, net, s['n_catch'], tot)

log('')
log(f'最好档：止损 -{best[0]:.1%}  净期望 {best[1]*100:+.3f}%/笔  抓住 {best[2]:,}/{best[3]:,} ({best[2]/best[3]*100:.1f}%)')

log('')
log('【关键对照】')
log(f'  理论周期数（口径C）9,897   可买到的（口径A）8,528')
log(f'  实际能抓住（最好止损档）{best[2]:,}')
log('')
log('【判读】')
log('  · 抓中率 = 在到达 +5% 之前没跌破止损的比例')
log('  · 净期望 = (抓中×5% − 打掉×SL)/总数 − 0.5%')
log('  · 若所有档净期望都为负 ⇒ 这些机会看得见、吃不到')

out = os.path.join(OUT, 'type1_catch_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
