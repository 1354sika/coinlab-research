# -*- coding: utf-8 -*-
"""分钟级下沉取数：top N 币的 1m K 线，尽量拉长（Binance vision，可达）

动机：日线尺度上实测「无滞后可套」（峰值全在 lag=0）；若传导真在秒/分钟级，
      必须用分钟数据检验，并同时把「换手成本」算进去。
做法：分页取 1m（每次 1000 根 ≈ 16.7 小时），断点续传，写入 data/klines_1m_long.csv
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, os.environ.get("MIN_OUT", "klines_1m_long.csv"))
SRC = os.path.join(DATA, "klines_1d_top502.csv")
BASE = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]

TOPN = int(os.environ.get("MIN_TOP_N", "12"))
DAYS = int(os.environ.get("MIN_DAYS", "30"))

# 按现有日线数据选成交额最大的 TOPN 个币
d1 = pd.read_csv(SRC)
syms = (d1.groupby("symbol")["quote_volume"].sum().sort_values(ascending=False)
        .head(TOPN).index.tolist())
print("target symbols (%d): %s" % (len(syms), syms), flush=True)

done = {}
if os.path.exists(OUT):
    old = pd.read_csv(OUT, usecols=["symbol", "open_time"])
    for s, g in old.groupby("symbol"):
        done[s] = int(g["open_time"].min())
print("resume: %s" % {k: v for k, v in done.items()}, flush=True)


def fetch(sym, end_ms, start_ms):
    sess = requests.Session()
    t, got = start_ms, []
    while t < end_ms:
        js = None
        for attempt in range(3):
            try:
                r = sess.get(BASE, params=dict(symbol=sym, interval="1m",
                                               startTime=t, limit=1000), timeout=25)
                if r.status_code != 200:
                    raise RuntimeError("HTTP %d" % r.status_code)
                js = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    return sym, got, str(e)
                time.sleep(1.0)
        if not js:
            break
        got += js
        t = js[-1][0] + 60000
        if len(js) < 1000:
            break
    return sym, got, None


now = int(time.time() * 1000)
span = DAYS * 86400000
first = not os.path.exists(OUT)
with open(OUT, "a", encoding="utf-8", newline="") as fh:
    if first:
        fh.write("symbol," + ",".join(COLS) + "\n")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for i, s in enumerate(syms):
            end = now   # 统一下载截止时间，保证所有币有共同时间窗
            if s in done and done[s] <= end - span + 3600000:
                print("  skip %s (have from %d)" % (s, done[s]), flush=True)
                continue
            futs[ex.submit(fetch, s, end, end - span)] = s
        n = 0
        for fu in as_completed(futs):
            s, got, err = fu.result()
            n += 1
            if err:
                print("  FAIL %s %s" % (s, err), flush=True)
                continue
            for row in got:
                fh.write("%s,%s\n" % (s, ",".join(str(x) for x in row)))
            fh.flush()
            print("  %d/%d %s bars=%d" % (n, len(futs), s, len(got)), flush=True)
print("DONE", flush=True)
