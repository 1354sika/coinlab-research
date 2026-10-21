# -*- coding: utf-8 -*-
"""扫 E 盘 1m 数据头尾，统计每个币的时间跨度与总覆盖。"""
import os, glob, datetime as dt

D = r'E:\binance\klines_1m\1m'
files = sorted(glob.glob(os.path.join(D, '*.csv')))
print('files', len(files))

def head_tail_time(p):
    size = os.path.getsize(p)
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()                      # header
        first = f.readline()
    with open(p, 'rb') as f:
        f.seek(max(0, size - 4096))
        tail = f.read().decode('utf-8', 'ignore').strip().splitlines()
    last = tail[-1] if tail else ''
    def ts(line):
        parts = line.split(',')
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except Exception:
            return None
    return ts(first), ts(last), size

rows = []
for p in files:
    try:
        a, b, sz = head_tail_time(p)
        if a and b:
            rows.append((os.path.basename(p)[:-4], a, b, sz))
    except Exception as e:
        rows.append((os.path.basename(p)[:-4], None, None, 0))

ok = [r for r in rows if r[1]]
print('parsed', len(ok), '/', len(rows))

def d(ms):
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime('%Y-%m-%d')

ok.sort(key=lambda r: r[1])
spans = [((r[2] - r[1]) / 86400000.0, r[0], r[1], r[2]) for r in ok]
spans.sort()

tot_gb = sum(r[3] for r in ok) / (1024 ** 3)
mins = min(r[1] for r in ok)
maxs = max(r[2] for r in ok)

print('total %.2f GB' % tot_gb)
print('global range %s -> %s  (%.1f years)' % (d(mins), d(maxs), (maxs - mins) / 86400000.0 / 365.25))
print()
print('--- span distribution ---')
buckets = {'<30d': 0, '30-180d': 0, '180-365d': 0, '1-2y': 0, '2-3y': 0, '>3y': 0}
for sp, name, a, b in spans:
    if sp < 30: buckets['<30d'] += 1
    elif sp < 180: buckets['30-180d'] += 1
    elif sp < 365: buckets['180-365d'] += 1
    elif sp < 730: buckets['1-2y'] += 1
    elif sp < 1095: buckets['2-3y'] += 1
    else: buckets['>3y'] += 1
for k, v in buckets.items():
    print('  %-10s %4d' % (k, v))

print()
print('--- coins with >= 3 years (%d) ---' % buckets['>3y'])
n3 = [x for x in spans if x[0] >= 1095]
n3.sort(reverse=True)
for sp, name, a, b in n3[:15]:
    print('  %-14s %7.1f y  %s -> %s' % (name, sp / 365.25, d(a), d(b)))

print()
print('--- shortest 5 ---')
for sp, name, a, b in spans[:5]:
    print('  %-14s %7.1f d  %s -> %s' % (name, sp, d(a), d(b)))
