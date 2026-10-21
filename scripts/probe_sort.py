# -*- coding: utf-8 -*-
"""probe_sort.py — 快速探测各 CSV 是否按 symbol 连续排序 + 行数/粒度"""
import os, csv, sys

D = r'D:\中信证券\coinlab\data'
FILES = [f for f in os.listdir(D) if f.endswith('.csv') and f != 'symbols_usdt.csv']

for f in FILES:
    p = os.path.join(D, f)
    runs = 0
    prev = None
    first = []
    last = None
    n = 0
    bad_order = 0
    with open(p, 'r', encoding='utf-8', errors='ignore', newline='') as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        try:
            ix_s = hdr.index('symbol')
            ix_t = hdr.index('open_time')
        except ValueError:
            print(f, '列不合规', hdr); continue
        for row in rd:
            n += 1
            s = row[ix_s]
            if s != prev:
                runs += 1
                if len(first) < 3: first.append(s)
                prev = s
            last = s
    print(f'{f:28s} 行={n:>10,d} symbol游程={runs:>5d} 首={first} 末={last}')
