# -*- coding: utf-8 -*-
"""快速抽样：列出已落盘 1m 文件的首/尾时间、按起始年份分布（只读首末行，不全扫）"""
import glob
import os
import collections
from datetime import datetime, timezone

D = os.environ.get("KPATH", r"E:\binance\klines_1m\1m")


def fmt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


rows = []
for p in sorted(glob.glob(os.path.join(D, "*.csv"))):
    with open(p, encoding="utf-8") as f:
        f.readline()
        line = f.readline()
        if not line:
            continue
        a = int(line.split(",")[1])
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(max(0, size - 8192))
        tail = f.read().decode("utf-8", "ignore").strip().splitlines()
    lines = [x for x in tail if x and not x.startswith("symbol")]
    if not lines:
        continue
    b = int(lines[-1].split(",")[1])
    rows.append((os.path.basename(p)[:-4], a, b, size))

rows.sort(key=lambda r: r[1])
print("已落盘文件 %d 个" % len(rows))
print("\n最早起点 6 个:")
for n, a, b, s in rows[:6]:
    print("  %-14s %s -> %s  %.0f MB" % (n, fmt(a), fmt(b), s / 1e6))
print("\n最晚起点 4 个:")
for n, a, b, s in rows[-4:]:
    print("  %-14s %s -> %s  %.0f MB" % (n, fmt(a), fmt(b), s / 1e6))
c = collections.Counter(fmt(a)[:4] for n, a, b, s in rows)
print("\n按起始年份分布:", dict(sorted(c.items())))
print("尾部时间范围:", min(fmt(b) for n, a, b, s in rows), "~", max(fmt(b) for n, a, b, s in rows))
