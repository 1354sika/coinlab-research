# -*- coding: utf-8 -*-
"""取数 v0：data-api.binance.vision 现货日线 -> CSV

用法：
    python -X utf8 fetch_spot.py [topN]

产物（coinlab/data/）：
    symbols_usdt.csv        TRADING + USDT 计价的全市场符号
    klines_1d_top{N}.csv    按 24h 成交额排名的前 N 个符号，日线 1000 根
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


def api(path, **params):
    q = urllib.parse.urlencode(params)
    url = BASE + path + ("?" + q if q else "")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    t0 = time.time()

    info = api("/api/v3/exchangeInfo")
    # 排除稳定币 / 法币计价对与非 ASCII 符号（它们不是「币」）
    STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDS", "PAX",
              "USDE", "USDE", "AEUR", "EUR", "TRY", "BRL", "ARS", "JPY",
              "RUB", "UAH", "ZAR", "PLN", "RON", "CZK", "MXN", "COP",
              "NGN", "IDR", "THB", "USD1", "UST", "VAI", "BIDR", "BKRW",
              "BVND", "IDRT", "KZT"}
    syms = [s for s in info["symbols"]
            if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
            and s["baseAsset"] not in STABLE
            and s["baseAsset"].isascii()]
    print("TRADING/USDT symbols = %d" % len(syms), flush=True)

    sym_path = os.path.join(DATA, "symbols_usdt.csv")
    with open(sym_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "baseAsset", "status"])
        for s in sorted(syms, key=lambda x: x["symbol"]):
            w.writerow([s["symbol"], s["baseAsset"], s["status"]])

    wanted = {s["symbol"] for s in syms}
    tk = api("/api/v3/ticker/24hr")
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in tk if t["symbol"] in wanted}
    ranked = sorted(vol.items(), key=lambda kv: -kv[1])[:top_n]
    print("ranked by 24h quoteVolume, taking top %d" % len(ranked), flush=True)

    out = os.path.join(DATA, "klines_1d_top%d.csv" % top_n)
    done = set()
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if row:
                    done.add(row[0])
        print("resume: %d symbols already in file" % len(done), flush=True)
    n_rows = 0
    per_sym = {}
    with open(out, "a" if done else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["symbol", "open_time", "open", "high", "low", "close",
                        "volume", "quote_volume", "trades",
                        "taker_buy_base", "taker_buy_quote", "close_time"])
        for i, (sym, qv) in enumerate(ranked, 1):
            if sym in done:
                continue
            ks = api("/api/v3/klines", symbol=sym, interval="1d", limit=1000)
            for k in ks:
                w.writerow([sym, k[0], k[1], k[2], k[3], k[4], k[5],
                            k[7], k[8], k[9], k[10], k[6]])
                n_rows += 1
            per_sym[sym] = len(ks)
            if i % 10 == 0:
                print("  %d/%d %s rows=%d elapsed=%.1fs"
                      % (i, len(ranked), sym, n_rows, time.time() - t0), flush=True)

    lens = sorted(per_sym.values())
    print("DONE rows=%d symbols=%d file=%s" % (n_rows, len(per_sym), out), flush=True)
    print("bars/symbol: min=%d median=%d max=%d"
          % (lens[0], lens[len(lens) // 2], lens[-1]), flush=True)
    print("elapsed=%.1fs" % (time.time() - t0), flush=True)


main()
