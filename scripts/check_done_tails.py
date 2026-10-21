# -*- coding: utf-8 -*-
"""只看 progress 里已完成(msg=ok)的币：其文件尾部是否到今天；并区分"进行中"的文件。"""
import glob
import json
import os
from datetime import datetime, timezone

D = r"E:\binance\klines_1m\1m"
P = r"E:\binance\klines_1m\progress.json"
NOW = datetime.now(timezone.utc)


def tail_ms(p):
    with open(p, "rb") as f:
        f.seek(max(0, os.path.getsize(p) - 8192))
        lines = [x for x in f.read().decode("utf-8", "ignore").strip().splitlines()
                 if x and not x.startswith("symbol")]
    return int(lines[-1].split(",")[1]) if lines else None


def head_ms(p):
    with open(p, encoding="utf-8") as f:
        f.readline()
        ln = f.readline()
    return int(ln.split(",")[1]) if ln else None


def fmt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


prog = json.load(open(P, encoding="utf-8"))
done = [k for k, v in prog.items() if v.get("msg") == "ok"]
skip = [k for k, v in prog.items() if "最新" in str(v.get("msg"))]
files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(D, "*.csv"))}
print("progress: 完成 ok=%d 跳过=%d 其中无文件=%d" % (len(done), len(skip), len([k for k in done + skip if k not in files])))
print("\n=== 已完成币的尾部（应全部到 %s 附近）===" % fmt(NOW.timestamp() * 1000))
bad = []
for k in done + skip:
    p = files.get(k)
    if not p:
        continue
    t = tail_ms(p)
    lag_h = (NOW.timestamp() * 1000 - t) / 3600000 if t else 1e9
    flag = "" if lag_h < 3 else "  <<< 落后 %.0f 小时" % lag_h
    if lag_h >= 3:
        bad.append((k, t, lag_h))
    print("  %-14s %s -> %s%s" % (k, fmt(head_ms(p)), fmt(t), flag))
print("\n=== 进行中（有文件但 progress 无记录）===")
pending = [k for k in files if k not in prog]
print("  共 %d 个；尾部分布前 5:" % len(pending))
for k in pending[:5]:
    print("  %-14s %s -> %s" % (k, fmt(head_ms(files[k])), fmt(tail_ms(files[k]))))
print("\n结论: 已完成里尾部落后>3h 的 = %d 个 %s" % (len(bad), bad[:5]))
