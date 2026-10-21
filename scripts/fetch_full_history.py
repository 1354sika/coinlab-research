# -*- coding: utf-8 -*-
"""补全历史：日线从 2017-01-01 起

动机：klines_1d_top502.csv 是被 limit=1000 截断的（老币只有最近 1000 天）。
      要判「簇内传导前半亏 61%/后半赚 121% 是行情还是规律」，必须加长样本。
做法：对每个 symbol 从 2017-01-01 分页取到最新，8 线程并发，断点续传（已存在 symbol 跳过）。
输出：data/klines_1d_full.csv
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "klines_1d_top502.csv")
OUT = os.path.join(DATA, "klines_1d_full.csv")
BASE = "https://data-api.binance.vision/api/v3/klines"
START = 1483228800000  # 2017-01-01 UTC
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]

syms = sorted(pd.read_csv(SRC, usecols=["symbol"])["symbol"].unique())
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, usecols=["symbol"])["symbol"].unique())
todo = [s for s in syms if s not in done]
print("symbols=%d  已完成=%d  待取=%d" % (len(syms), len(done), len(todo)), flush=True)


def fetch(s):
    sess = requests.Session()
    t, got = START, []
    while True:
        js = None
        for attempt in range(3):
            try:
                r = sess.get(BASE, params=dict(symbol=s, interval="1d",
                                               startTime=t, limit=1000), timeout=25)
                if r.status_code != 200:
                    raise RuntimeError("HTTP %d" % r.status_code)
                js = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    return s, got, str(e)
                time.sleep(1.0)
        if not js:
            break
        got += js
        t = js[-1][0] + 86400000
        if len(js) < 1000:
            break
    return s, got, None


first = not os.path.exists(OUT)
with open(OUT, "a", encoding="utf-8", newline="") as fh:
    if first:
        fh.write("symbol," + ",".join(COLS) + "\n")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, s): s for s in todo}
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
            if n % 25 == 0 or n == len(todo):
                print("  %d/%d  最近 %s bars=%d" % (n, len(todo), s, len(got)), flush=True)
print("DONE", flush=True)
