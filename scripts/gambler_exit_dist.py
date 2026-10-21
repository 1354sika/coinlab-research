# -*- coding: utf-8 -*-
"""
gambler_exit_dist · TP=+5% / SL=-12% / 1440min 的真实出口分布
目的：核对之前反推的「先触止盈约 25%」是否成立
口径：低点(过去240min) +3% 入场；同一根内双触保守算止损；扣费 0.50%/笔
"""
import os, sys, time, io
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FN = 'klines_1m_1y.csv'
LOW_WIN = 240
HOLD = 1440
CONFIRM = 0.03
TP = 0.05
SL = 0.12
FEE = 0.005

logbuf = []
def log(s=''):
    print(s); logbuf.append(str(s))

def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_exit_dist · TP=+5% / SL=-12% / 1440min 真实出口分布')
    log('=' * 100)
    p = os.path.join(DATA, FN)
    lst = []
    for chunk in pd.read_csv(p, chunksize=2_000_000,
                             usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close']):
        chunk['symbol'] = chunk['symbol'].astype(str)
        lst.append(chunk)
    df = pd.concat(lst, ignore_index=True)
    syms = sorted(df['symbol'].unique().tolist())
    log(f'币 {len(syms)}  行 {len(df):,}  读取 {time.time()-t0:.0f}s')

    n_tp = n_sl = n_to = 0
    rets = []
    ret_tp, ret_sl, ret_to = [], [], []
    hold_tp, hold_sl = [], []
    tot = 0

    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        ot = g['open_time'].values.astype(np.int64)
        o = g['open'].values.astype(np.float64)
        h = g['high'].values.astype(np.float64)
        l = g['low'].values.astype(np.float64)
        c = g['close'].values.astype(np.float64)
        n = len(c)
        if n < LOW_WIN + HOLD + 10:
            continue
        lo_win = pd.Series(l).rolling(LOW_WIN, min_periods=60).min().values
        entry = lo_win * (1 + CONFIRM)
        m = (c >= entry) & (o <= entry * 1.03)
        m[0] = False
        m &= np.isfinite(lo_win)
        idxs = np.where(m)[0]
        tot += len(idxs)
        for idx in idxs:
            e = entry[idx]
            tp_p = e * (1 + TP)
            sl_p = e * (1 - SL)
            end = min(idx + HOLD, n - 1)
            seg_h = h[idx+1:end+1]
            seg_l = l[idx+1:end+1]
            hit_tp = np.where(seg_h >= tp_p)[0]
            hit_sl = np.where(seg_l <= sl_p)[0]
            i_tp = int(hit_tp[0]) if len(hit_tp) else 10**9
            i_sl = int(hit_sl[0]) if len(hit_sl) else 10**9
            if i_sl <= i_tp and i_sl < 10**9:
                n_sl += 1
                r = -SL - FEE
                ret_sl.append(r); hold_sl.append(i_sl + 1)
            elif i_tp < 10**9:
                n_tp += 1
                r = TP - FEE
                ret_tp.append(r); hold_tp.append(i_tp + 1)
            else:
                n_to += 1
                r = c[end] / e - 1 - FEE
                ret_to.append(r)
            rets.append(r)
        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    N = len(rets)
    arr = np.array(rets)
    log(f'\n入场点 {tot:,}  有效 {N:,}  ({time.time()-t0:.0f}s)')
    log('\n' + '=' * 100)
    log('【出口分布】')
    log('=' * 100)
    log(f'  先触止盈 +{TP:.1%}   {n_tp:>7,}  {n_tp/N:>6.2%}   均值 {np.mean(ret_tp) if ret_tp else 0:+.3f}%')
    log(f'  先触止损 -{SL:.1%}   {n_sl:>7,}  {n_sl/N:>6.2%}   均值 {np.mean(ret_sl) if ret_sl else 0:+.3f}%')
    log(f'  超时按收盘价      {n_to:>7,}  {n_to/N:>6.2%}   均值 {np.mean(ret_to) if ret_to else 0:+.3f}%  中位 {np.median(ret_to) if ret_to else 0:+.3f}%')
    log(f'\n  合计均值 {arr.mean():+.4f}%   中位 {np.median(arr):+.4f}%   胜率(w/n) {np.mean(arr>0):.2%}')
    log(f'  p1 {np.percentile(arr,1):+.2f}%  p10 {np.percentile(arr,10):+.2f}%  p90 {np.percentile(arr,90):+.2f}%  p99 {np.percentile(arr,99):+.2f}%')
    log(f'  最差 {arr.min():+.2f}%   最好 {arr.max():+.2f}%')
    if hold_tp:
        log(f'  止盈笔持有分钟 中位 {np.median(hold_tp):.0f}  p90 {np.percentile(hold_tp,90):.0f}')
    if hold_sl:
        log(f'  止损笔持有分钟 中位 {np.median(hold_sl):.0f}  p90 {np.percentile(hold_sl,90):.0f}')
    log('\n' + '=' * 100)
    log('【核对】之前反推：先触止盈约 25% / 止损约 7% / 超时约 68%')
    log(f'        本次实测：止盈 {n_tp/N:.2%} / 止损 {n_sl/N:.2%} / 超时 {n_to/N:.2%}')
    log('=' * 100)

    out = os.path.join(OUT, 'exit_dist_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
