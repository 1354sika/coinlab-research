# -*- coding: utf-8 -*-
"""通用取数：任意 interval + 分页回溯 N 天，取 24h 成交额 topN 的 USDT 交易对

用法：
    python -X utf8 fetch_klines.py [interval] [days] [topN]
例：
    python -X utf8 fetch_klines.py 1h 365 20
    python -X utf8 fetch_klines.py 1m 7 10

产物：coinlab/data/klines_{interval}_top{N}_{days}d.csv
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://data-api.binance.vision"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDS", "PAX",
          "USDE", "AEUR", "EUR", "TRY", "BRL", "ARS", "JPY", "RUB", "UAH",
          "ZAR", "PLN", "RON", "CZK", "MXN", "COP", "NGN", "IDR", "THB",
          "USD1", "UST", "VAI", "BIDR", "BKRW", "BVND", "IDRT", "KZT"}

HEADER = ["symbol", "open_time", "open", "high", "low", "close",
          "volume", "quote_volume", "trades",
          "taker_buy_base", "taker_buy_quote", "close_time"]


def api(path, **params):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(1.2 * (attempt + 1))
    raise last


def row(sym, k):
    return [sym, k[0], k[1], k[2], k[3], k[4], k[5],
            k[7], k[8], k[9], k[10], k[6]]


def main():
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    t0 = time.time()

    info = api("/api/v3/exchangeInfo")
    wanted = {s["symbol"] for s in info["symbols"]
              if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
              and s["baseAsset"] not in STABLE and s["baseAsset"].isascii()}
    tk = api("/api/v3/ticker/24hr")
    ranked = sorted(((t["symbol"], float(t["quoteVolume"]))
                     for t in tk if t["symbol"] in wanted),
                    key=lambda x: -x[1])[:top_n]
    print("universe=%d  taking top %d  interval=%s  days=%d"
          % (len(wanted), len(ranked), interval, days), flush=True)

    now = int(time.time() * 1000)
    start_ms = now - days * 86400 * 1000
    out = os.path.join(DATA, "klines_%s_top%d_%dd.csv" % (interval, top_n, days))
    n_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, (sym, qv) in enumerate(ranked, 1):
            end = now
            bars = []
            while end > start_ms:
                ks = api("/api/v3/klines", symbol=sym, interval=interval,
                         endTime=end, limit=1000)
                if not ks:
                    break
                bars = ks + bars
                end = int(ks[0][0]) - 1
                if time.time() - t0 > 1500:
                    print("  [warn] time budget hit at %s" % sym, flush=True)
                    break
            seen = set()
            kept = 0
            for k in bars:
                if k[0] in seen:
                    continue
                seen.add(k[0])
                if int(k[0]) >= start_ms:
                    w.writerow(row(sym, k))
                    kept += 1
                    n_rows += 1
            print("  %d/%d %s bars=%d total=%d elapsed=%.0fs"
                  % (i, len(ranked), sym, kept, n_rows, time.time() - t0), flush=True)

    print("DONE rows=%d file=%s elapsed=%.0fs" % (n_rows, out, time.time() - t0), flush=True)


main()
