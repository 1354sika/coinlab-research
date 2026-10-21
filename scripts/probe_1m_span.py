# -*- coding: utf-8 -*-
"""probe_1m_span · 盘点全部 1m K 线文件的时间跨度、币数、行数（只读，不写数据）"""
import os, glob, time
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
t0 = time.time()
buf = []


def log(s):
    print(s, flush=True); buf.append(s)


log('=' * 108)
log('1m K 线数据盘点（只读）—— 回答「1m 历史有多少」')
log('=' * 108)

files = sorted(glob.glob(os.path.join(DATA, 'klines_1m_*.csv')))
if not files:
    log('未找到 klines_1m_*.csv')

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
        log(f'{name:34s} {size:6.2f}GB  读取失败: {type(e).__name__}: {e}')
        continue
    if lo is None:
        log(f'{name:34s} {size:6.2f}GB  空文件')
        continue
    d0 = pd.to_datetime(lo, unit='ms', utc=True)
    d1 = pd.to_datetime(hi, unit='ms', utc=True)
    days = (hi - lo) / 86400000.0
    log(f'{name:34s} {size:6.2f}GB {rows:>13,} 行 {len(syms):>5} 币  '
        f'{d0:%Y-%m-%d %H:%M} ~ {d1:%Y-%m-%d %H:%M}  {days:7.0f} 天 ({days/30.4:5.1f} 月)')

log(f'\nTOTAL {time.time()-t0:.0f}s')
out = os.path.join(OUT, 'data_1m_span.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {out}')
