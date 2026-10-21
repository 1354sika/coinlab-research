# -*- coding: utf-8 -*-
"""核验：已落盘文件是否有"中途截断"。对每个可疑币，从它文件尾部之后向 API 追问是否有数据。"""
import glob
import json
import os
import urllib.request
from datetime import datetime, timezone

D = r"E:\binance\klines_1m\1m"
P = r"E:\binance\klines_1m\progress.json"
BASE = "https://data-api.binance.vision/api/v3/klines"
NOW = datetime.now(timezone.utc)


def fmt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def tail_ms(p):
    with open(p, "rb") as f:
        f.seek(max(0, os.path.getsize(p) - 8192))
        lines = [x for x in f.read().decode("utf-8", "ignore").strip().splitlines()
                 if x and not x.startswith("symbol")]
    return int(lines[-1].split(",")[1]) if lines else None


prog = json.load(open(P, encoding="utf-8")) if os.path.exists(P) else {}
print("=== 尾部距今天 > 3 天的文件（可疑截断）===")
sus = []
for p in sorted(glob.glob(os.path.join(D, "*.csv"))):
    t = tail_ms(p)
    if t is None:
        continue
    lag_d = (NOW.timestamp() * 1000 - t) / 86400000
    if lag_d > 3:
        sus.append((os.path.basename(p)[:-4], t, lag_d, os.path.getsize(p)))
for n, t, lag, s in sorted(sus, key=lambda x: -x[2]):
    st = prog.get(n, {})
    print("  %-14s 尾部 %s (落后 %.1f 天, %.0f MB) progress=%s" % (n, fmt(t), lag, s / 1e6, st))

print("\n=== 对可疑币追问 API：断点之后是否还有数据 ===")
for n, t, lag, s in sorted(sus, key=lambda x: -x[2])[:8]:
    u = "%s?symbol=%s&interval=1m&limit=3&startTime=%d" % (BASE, n, t + 60000)
    try:
        k = json.loads(urllib.request.urlopen(u, timeout=20).read())
        if k:
            print("  %-14s 有数据 -> 下一根 %s ；最新 %s" % (n, fmt(k[0][0]), fmt(k[-1][0])))
        else:
            print("  %-14s 空（该时点后确实无 1m 数据）" % n)
    except Exception as e:
        print("  %-14s 查询失败 %r" % (n, e))
