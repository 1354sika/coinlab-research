# -*- coding: utf-8 -*-
"""体检 live 各日 klines_1s.csv 的真实覆盖：混了多少历史行、当天实际覆盖多少分钟。"""
import os
import numpy as np
import pandas as pd

LIVE = r'D:\中信证券\coinlab\data\live'
EVI = r'D:\中信证券\coinlab\evidence'
L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

def fmt(ms):
    try:
        if ms is None or not np.isfinite(ms):
            return 'NA'
        return pd.to_datetime(int(ms), unit='ms').strftime('%m-%d %H:%M')
    except Exception:
        return 'NA'

p('=' * 116)
p('LIVE 1s COVERAGE AUDIT')
p('=' * 116)
p('%-12s %10s %8s %13s %13s %9s %9s %10s' % (
    'day', 'rows', 'symbols', 'min_ts', 'max_ts', 'mins_cov', 'bars_med', 'hist_rows'))
p('-' * 116)

for d in sorted(os.listdir(LIVE)):
    f = os.path.join(LIVE, d, 'klines_1s.csv')
    if not os.path.exists(f):
        continue
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            hdr = fh.readline().strip().replace('\ufeff', '')
            first = fh.readline().strip()
        nh, nd = len(hdr.split(',')), len(first.split(','))
        if nd == nh + 1:
            names = hdr.split(',') + ['close_time']
            raw = pd.read_csv(f, skiprows=1, names=names, index_col=False)
        else:
            raw = pd.read_csv(f)
    except Exception as e:
        p('%-12s READ FAIL %s' % (d, type(e).__name__)); continue
    if not len(raw):
        p('%-12s EMPTY' % d); continue

    # 列名去 BOM / 空白
    raw.columns = [str(c).replace('\ufeff', '').strip() for c in raw.columns]
    if 'open_time' not in raw.columns or 'symbol' not in raw.columns:
        p('%-12s BAD COLS %s' % (d, list(raw.columns)[:6])); continue

    raw['open_time'] = pd.to_numeric(raw['open_time'], errors='coerce')
    raw = raw[raw['open_time'].notna()]
    if not len(raw):
        p('%-12s NO NUMERIC open_time' % d); continue

    raw['open_time'] = raw['open_time'].astype('int64')
    raw['t'] = raw['open_time'] // 60000

    try:
        day0 = int(pd.Timestamp(d + ' 00:00:00', tz='UTC').value // 10**6)
    except Exception:
        p('%-12s BAD DIRNAME' % d); continue
    day1 = day0 + 86400 * 1000

    today = raw[(raw.open_time >= day0) & (raw.open_time < day1)]
    hist = len(raw) - len(today)

    if len(today):
        mins = today.t.nunique()
        per = today.groupby('symbol').t.nunique()
        bars_med = int(per.median())
    else:
        mins = 0
        bars_med = 0

    p('%-12s %10d %8d %13s %13s %9d %9d %10d' % (
        d, len(raw), raw.symbol.nunique(),
        fmt(raw.open_time.min()), fmt(raw.open_time.max()),
        mins, bars_med, hist))

p('-' * 116)
p('mins_cov  = 当天(UTC)实际有数据的分钟数，1440=完整一天')
p('hist_rows = 不属于当天的时间戳行数（live_recorder 回填/混入的历史行）')
p('bars_med  = 当天每币 1m 根数的中位数')

out = os.path.join(EVI, 'live_audit.txt')
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
