# -*- coding: utf-8 -*-
"""analyze_live_freq.py — 用最新 live 数据，数「跌2%+放量3倍」一天有几笔

口径与全项目最强规则一致：
  ret24 = close / close[24根前] - 1   （24 根 5m = 2 小时）
  volr  = 当前 5m 量 / 前 24 根均量
  入场  = ret24 <= -2%  且  volr >= 3.0

数据：coinlab/data/live/<日期>/klines_1s.csv（实时录制，1 秒 K 线）
处理：1s 聚合成 5m，再算指标。

诚实边界：live 只有部分时段（2026-10-18 约 8 小时）、1s 覆盖约 65% 的秒、
          聚合后每币约 96 根 5m；只能回答「这 8 小时几笔」，全天需外推。
"""
import os
import numpy as np
import pandas as pd

LIVE = r'D:\中信证券\coinlab\data\live'
OUT = r'D:\中信证券\coinlab\evidence\live_freq_REAL.txt'

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

log('=' * 96)
log('live_freq · 用最新实时数据数「跌2%+放量3倍」一天几笔')
log('=' * 96)

dates = sorted([d for d in os.listdir(LIVE) if os.path.isdir(os.path.join(LIVE, d))])
log(f'可用日期目录: {dates}')
log('')

for d in dates:
    SRC = os.path.join(LIVE, d, 'klines_1s.csv')
    if not os.path.exists(SRC):
        log(f'{d}: 无 klines_1s.csv')
        continue
    size_mb = os.path.getsize(SRC) / 1024 / 1024
    log('=' * 96)
    log(f'【{d}】klines_1s.csv  {size_mb:.1f} MB')
    log('=' * 96)

    try:
        df = pd.read_csv(SRC, skiprows=1, header=None)
    except Exception as e:
        log(f'  读取失败: {e}')
        continue

    ncol = df.shape[1]
    names = ['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_time'][:ncol]
    df.columns = names
    for col in ['open_time', 'open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open_time', 'close', 'volume'])
    df['open_time'] = df['open_time'].astype('int64')

    log(f'  原始行 {len(df)}  币 {df.symbol.nunique()}  '
        f'时间 {pd.to_datetime(df.open_time.min(), unit="ms")} ~ '
        f'{pd.to_datetime(df.open_time.max(), unit="ms")}')

    # 1s -> 5m
    df['bucket'] = df['open_time'] // 300_000
    agg = (df.groupby(['symbol', 'bucket'])
             .agg(close=('close', 'last'), volume=('volume', 'sum'), high=('high', 'max'), low=('low', 'min'))
             .reset_index())
    bars = agg.groupby('symbol').size()
    log(f'  聚合 5m 后：每币根数 中位 {bars.median():.0f}  最少 {bars.min()}  最多 {bars.max()}  '
        f'(需 24 根预热，有效根数中位 {max(0, bars.median()-24):.0f})')
    nb = int(agg.bucket.nunique())
    span_h = nb * 5 / 60.0
    log(f'  5m 桶数 {nb} => 实际覆盖 {span_h:.1f} 小时'
        f'（原始时间戳跨 {(df.open_time.max()-df.open_time.min())/3600_000:.0f} 小时 = 文件混有历史行，不代表覆盖）')
    log('')

    rows = []
    for sym, g in agg.groupby('symbol', sort=True):
        g = g.sort_values('bucket')
        c = g.close.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
        n = len(c)
        if n < 30:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n)
        ret24[24:] = c[24:] / c[:-24] - 1.0
        sig = (ret24 <= -0.02) & (volr >= 3.0)
        sig[:24] = False
        cnt = int(sig.sum())
        rows.append((sym, n, cnt, float(volr[sig].max()) if cnt else float('nan')))

    R = pd.DataFrame(rows, columns=['sym', 'bars', 'n_entries', 'max_volr'])
    tot = int(R.n_entries.sum())
    log(f'{"币":<14} {"5m根":>6} {"入场次数":>8} {"最大量比":>9}')
    for _, r in R.sort_values('n_entries', ascending=False).head(15).iterrows():
        log(f'{r.sym:<14} {int(r.bars):>6} {int(r.n_entries):>8} {r.max_volr:>8.1f}')
    log('')
    log(f'  合计入场 {tot} 笔 / {len(R)} 币 / {span_h:.1f} 小时')
    if span_h > 0:
        log(f'  折合每小时 {tot/span_h:.2f} 笔；若全天 24h 同强度 => 约 {tot/span_h*24:.0f} 笔/天')
    log(f'  有信号的币 {int((R.n_entries>0).sum())}/{len(R)}')
    log('')

log('诚实边界：')
log('  1. live 只覆盖部分时段（2026-10-18 约 8 小时），1s 数据约 65% 的秒有值。')
log('  2. 5m 聚合后每币约 96 根，ret24 需要 24 根（2 小时）预热，前 2 小时样本不可用。')
log('  3. 「外推全天」假设波动强度不变，实际早晚差异大。')
log('  4. 这是「信号出现次数」，不是「能成交的次数」——成交率未测。')
log('  5. 入场后是否真能涨 2% 未验证（本脚本只数信号）。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {OUT}')
