# -*- coding: utf-8 -*-
r"""
gambler_bench.py — 基准对照：后 50% 时间，等权买入持有

full2 的 0.9975 只有和基准比才有意义：
  残酷市场里"不亏"本身就是超额。

算三件事：
  1. 每币后 50% 的买入持有收益（等权平均）
  2. 等权买入持有的组合曲线（最大回撤）
  3. 与策略 0.9975 的超额
"""
import sys, io, os
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\bench.txt'
NSYM = 60
OBSERVE = 0.50

def main():
    print('='*100, flush=True)
    print('gambler_bench · 等权买入持有基准（后 50% 时间）', flush=True)
    print('='*100, flush=True)

    df = pd.read_csv(DATA, usecols=['symbol','open_time','close'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(NSYM).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)

    rows = []
    for sym, g in df.groupby('symbol', observed=True):
        c = g['close'].values.astype(np.float64)
        n = len(c)
        if n < 1000: continue
        cut0 = int(n*OBSERVE)
        seg = c[cut0:]
        bh = seg[-1]/seg[0] - 1.0
        peak = np.maximum.accumulate(seg)
        dd = (seg/peak - 1.0).min()
        rows.append((str(sym), bh, dd, seg[0], seg[-1]))

    rows.sort(key=lambda r: -r[1])
    bhs = np.array([r[1] for r in rows])
    dds = np.array([r[2] for r in rows])
    print(f'  币数 {len(rows)}', flush=True)
    t0 = pd.to_datetime(df[df.symbol==rows[0][0]]['open_time'].values[0], unit='ms')
    print(f'  每币后 50% 时段起点 ~{t0}', flush=True)
    print(flush=True)
    print('='*100, flush=True)
    print('【一】每币买入持有（后 50% 时间）', flush=True)
    print('='*100, flush=True)
    print(f'  收益：P10 {np.percentile(bhs,10)*100:+8.2f}%  P25 {np.percentile(bhs,25)*100:+8.2f}%  '
          f'P50 {np.percentile(bhs,50)*100:+8.2f}%  P75 {np.percentile(bhs,75)*100:+8.2f}%  P90 {np.percentile(bhs,90)*100:+8.2f}%', flush=True)
    print(f'  等权平均 {bhs.mean()*100:+.2f}%   盈利 {int((bhs>0).sum())}/{len(bhs)}', flush=True)
    print(f'  币内最大回撤：P50 {np.percentile(dds,50)*100:.1f}%  最差 {dds.min()*100:.1f}%', flush=True)
    print(flush=True)
    print('  最好 5:', flush=True)
    for s, b, d, a, z in rows[:5]:
        print(f'    {s:<14} {b*100:+8.2f}%  回撤 {d*100:6.1f}%', flush=True)
    print('  最差 5:', flush=True)
    for s, b, d, a, z in rows[-5:]:
        print(f'    {s:<14} {b*100:+8.2f}%  回撤 {d*100:6.1f}%', flush=True)

    # 等权组合曲线（按全局时间对齐、按币内后 50% 归一）
    print(flush=True)
    print('='*100, flush=True)
    print('【二】等权买入持有组合 vs 策略', flush=True)
    print('='*100, flush=True)
    # 等权组合期末 = 各币 (1+bh) 的平均
    combo = (1.0+bhs).mean()
    print(f'  等权买入持有组合期末 {combo:.4f}   （收益 {(combo-1)*100:+.2f}%）', flush=True)
    print(f'  策略（full2）        0.9975   （收益 -0.25%）', flush=True)
    print(f'  超额                 {(0.9975-combo)*100:+.2f} 个百分点', flush=True)
    print(flush=True)
    print('='*100, flush=True)
    print('【三】判据', flush=True)
    print('='*100, flush=True)
    if combo < 1.0:
        print('  · 基准为负（残酷市场）⇒ 策略"不亏"本身就是超额', flush=True)
    else:
        print('  · 基准为正 ⇒ 策略必须跑赢它才算本事', flush=True)
    print(f'  · 策略 {"跑赢" if 0.9975 > combo else "跑输"}基准 {(0.9975-combo)*100:+.2f} 个百分点', flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'等权基准 {combo:.6f}\n策略 0.9975\n超额 {(0.9975-combo)*100:.4f}pp\n')

if __name__ == '__main__':
    main()
