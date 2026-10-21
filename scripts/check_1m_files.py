# -*- coding: utf-8 -*-
"""校验落盘的 1m CSV：行数 / open_time 单调 / 重复 / 间隔断档 / 首尾时间"""
import os
import sys
import time
from datetime import datetime, timezone

D = sys.argv[1] if len(sys.argv) > 1 else r"E:\binance\klines_1m\1m"
for name in sorted(os.listdir(D)):
    if not name.endswith(".csv"):
        continue
    p = os.path.join(D, name)
    n = 0
    prev = None
    dup = 0
    back = 0
    gaps = []
    first = last = None
    with open(p, encoding="utf-8") as f:
        head = f.readline().strip()
        for line in f:
            n += 1
            t = int(line.split(",", 2)[1])
            if first is None:
                first = t
            if prev is not None:
                if t == prev:
                    dup += 1
                elif t < prev:
                    back += 1
                else:
                    d = (t - prev) // 60000
                    if d > 1:
                        gaps.append(int(d - 1))
            prev = t
            last = t
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    span = (last - first) // 60000 + 1 if first else 0
    print("%-16s 行=%9d 覆盖分钟=%9d 完整率=%6.2f%% 重复=%d 逆序=%d 断档段=%d 最大断档=%d 首=%s 尾=%s" % (
        name, n, span, 100.0 * n / span if span else 0, dup, back, len(gaps),
        max(gaps) if gaps else 0, fmt(first) if first else "-", fmt(last) if last else "-"))
    print("   表头: " + head)
