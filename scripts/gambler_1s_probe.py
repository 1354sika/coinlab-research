# -*- coding: utf-8 -*-
"""1s 粒度探查：live 录到的 1 秒 K 线能不能建基元。

现有 1s 数据只有 live/2026-10-16 和 live/2026-10-18 两天（12MB / 21MB）。
本脚本先只做体检，不下结论：
  ① 列名、币数、时间跨度、每秒成交笔数
  ② 1s 粒度上的短期反转/动量：上一秒涨跌 → 下一秒收益
  ③ 聚到 1 分钟后再看一次（和现有 1m 数据对比量级）

样本只有两天，任何"显著"都不可信——这里只回答"有没有值得继续看的结构"。
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

ROOT = r'D:\中信证券\coinlab\data\live'
OUT = r'D:\中信证券\coinlab\evidence\probe_1s.txt'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('1s 粒度体检')
say('=' * 100)

files = sorted(glob.glob(os.path.join(ROOT, '*', 'klines_1s.csv')))
say(f'  找到 {len(files)} 个文件')
allp = []
for f in files:
    day = os.path.basename(os.path.dirname(f))
    gb = os.path.getsize(f) / 1e6
    head = pd.read_csv(f, nrows=3, header=None, skiprows=1,
                       names=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
    say(f'\n  [{day}]  {gb:.1f} MB')
    say(f'    列: {list(head.columns)}')
    say(f'    首行: {head.iloc[0].to_dict()}')

    df = pd.read_csv(f, header=None, skiprows=1,
                     names=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
    say(f'    行数 {len(df):,}')
    if 'symbol' in df.columns:
        say(f'    币数 {df.symbol.nunique()}: {sorted(df.symbol.unique())[:10]}')
    # 时间列猜测
    tcol = None
    for c in df.columns:
        if 'time' in c.lower() or c.lower() in ('t', 'ts'):
            tcol = c
            break
    if tcol:
        tv = df[tcol].values
        if tv.max() > 1e12:
            ts = pd.to_datetime(tv, unit='ms')
        elif tv.max() > 1e9:
            ts = pd.to_datetime(tv, unit='s')
        else:
            ts = pd.Series(tv)
        say(f'    时间列 {tcol}: {ts.min()} ~ {ts.max()}  跨度 {len(df):,} 行')
    allp.append((day, df))

say('')
say('=' * 100)
say('【二】1s 上的短期反转/动量（上一秒涨跌 → 下一秒收益）')
say('=' * 100)
say('  注：只有两天数据，样本不独立（同一天内高度重叠），只作方向参考')
for day, df in allp:
    if 'symbol' not in df.columns or 'close' not in df.columns:
        say(f'  [{day}] 缺 symbol/close 列，跳过')
        continue
    say(f'\n  [{day}]')
    for sym, g in list(df.groupby('symbol'))[:5]:
        g = g.sort_values(g.columns[0])
        C = g['close'].values.astype(float)
        if len(C) < 1000:
            continue
        r = np.zeros(len(C)); r[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
        # 上一秒涨 → 下一秒
        prev = r[:-1]; nxt = r[1:]
        ok = np.isfinite(prev) & np.isfinite(nxt)
        if ok.sum() < 500:
            continue
        # 按上一秒涨跌分 4 档
        q = np.quantile(prev[ok], [0.25, 0.5, 0.75])
        bins = np.digitize(prev[ok], q)
        line = f'    {sym:12s} n={ok.sum():>7,}  '
        for k in range(4):
            m = bins == k
            if m.sum() > 50:
                line += f'Q{k+1}:{nxt[ok][m].mean()*1e4:+.3f}bps  '
        say(line)

say('')
say('=' * 100)
say('【三】数据量判断')
say('=' * 100)
for day, df in allp:
    say(f'  {day}: {len(df):,} 行，{df.symbol.nunique() if "symbol" in df.columns else "?"} 币')
say('  · 两天、约 20 币 → 独立事件远不足以判别任何规律')
say('  · 要真正用 1s，需要 live-recorder 连续录 30 天以上')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
