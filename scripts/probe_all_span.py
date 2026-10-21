# -*- coding: utf-8 -*-
"""probe_all_span · 盘点 data 下全部 klines_*.csv 的跨度/币数/行数（只读，不写数据）

回答「我们到底有哪些历史资料」。
仅读 symbol + open_time 两列，按 300 万行分块累计 min/max 与去重币种。
"""
import os, glob, time
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
t0 = time.time()
buf = []


def log(s):
    print(s, flush=True); buf.append(s)


log('=' * 116)
log('全部 K 线数据盘点（只读）—— data\\klines_*.csv')
log('=' * 116)

files = sorted(glob.glob(os.path.join(DATA, 'klines_*.csv')))
log(f'共 {len(files)} 个文件\n')

for fn in files:
    name = os.path.basename(fn)
    size = os.path.getsize(fn) / 1e9
    lo = None; hi = None; syms = set(); rows = 0
    try:
        for ch in pd.read_csv(fn, usecols=['symbol', 'open_time'], chunksize=3_000_000):
            rows += len(ch)
            t = ch['open_time']
            a, b = int(t.min()), int(t.max())
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
            syms.update(ch['symbol'].drop_duplicates().tolist())
    except Exception as e:
        log(f'{name:32s} {size:7.2f}GB  读取失败: {type(e).__name__}: {e}')
        continue
    if lo is None:
        log(f'{name:32s} {size:7.2f}GB  空文件')
        continue
    d0 = pd.to_datetime(lo, unit='ms', utc=True)
    d1 = pd.to_datetime(hi, unit='ms', utc=True)
    days = (hi - lo) / 86400000.0
    log(f'{name:32s} {size:7.2f}GB {rows:>14,} 行 {len(syms):>5} 币  '
        f'{d0:%Y-%m-%d %H:%M} ~ {d1:%Y-%m-%d %H:%M} {days:7.0f} 天 ({days/30.4:5.1f} 月)')

# live 目录单独看
live_root = os.path.join(DATA, 'live')
if os.path.isdir(live_root):
    log('\n--- live（实时录制）---')
    for d in sorted(os.listdir(live_root)):
        p = os.path.join(live_root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            fp = os.path.join(p, f)
            sz = os.path.getsize(fp) / 1e6
            log(f'live\\{d}\\{f:24s} {sz:8.1f} MB')

log(f'\nTOTAL {time.time()-t0:.0f}s')
out = os.path.join(OUT, 'data_all_span.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {out}')
