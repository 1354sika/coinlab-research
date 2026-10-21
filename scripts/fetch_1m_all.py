# -*- coding: utf-8 -*-
"""币安全市场 1m K 线全量落盘（落 E: 移动盘，断点续传）

用法：
  python -X utf8 fetch_1m_all.py --out E:\\binance\\klines_1m [--symbols BTCUSDT,ETHUSDT] [--workers 8]

落盘结构：
  <out>\\1m\\<SYMBOL>.csv      表头同现有 klines_*.csv（symbol,open_time,...,ignore）
  <out>\\progress.json        每币状态（done 行数 / 最后 open_time / 错误）
文件级断点：已存在则从最后一行 open_time+60s 继续，不重拉。
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
HEADER = "symbol," + ",".join(COLS) + "\n"
PAGE = 1000
PING = "https://data-api.binance.vision/api/v3/time"
lock = threading.Lock()
stat = {"req": 0, "rows": 0, "err": 0, "bytes": 0}
stop_flag = threading.Event()


def net_alive(timeout=15):
    try:
        with urllib.request.urlopen(PING, timeout=timeout) as r:
            r.read(64)
        return True
    except Exception:
        return False


def wait_for_net(max_wait=8 * 3600):
    """断网则挂起等待恢复（不把币标成失败）；超时或收到停止信号返回 False"""
    t0 = time.time()
    print("[net] 网络不可达，挂起等待恢复（最多 %.0f 分钟）..." % (max_wait / 60), flush=True)
    while time.time() - t0 < max_wait:
        if stop_flag.is_set():
            return False
        time.sleep(60)
        if net_alive():
            print("[net] 网络已恢复，继续拉取（挂起 %.0f 分钟）" % ((time.time() - t0) / 60), flush=True)
            return True
    stop_flag.set()
    return False


def api(symbol, start_ms):
    """失败重试；若判定为断网则挂起等待，恢复后继续。最终失败抛异常（由上层标 ERR）。"""
    for attempt in range(6):
        if stop_flag.is_set():
            return None
        try:
            url = "%s?symbol=%s&interval=1m&limit=%d&startTime=%d" % (BASE, symbol, PAGE, start_ms)
            with urllib.request.urlopen(url, timeout=40) as r:
                raw = r.read()
            k = json.loads(raw)
            with lock:
                stat["req"] += 1
                stat["bytes"] += len(raw)
            return k
        except Exception:
            if attempt >= 2 and not net_alive():
                if not wait_for_net():
                    return None
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("api failed after retries")


def last_open_time(path):
    """读文件最后一个完整行的 open_time；顺带把被 kill 时留下的半行截掉。空/不存在返回 None"""
    if not os.path.exists(path) or os.path.getsize(path) < 200:
        return None
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 65536))
        tail = f.read()
    if not tail.endswith(b"\n"):
        cut = tail.rfind(b"\n")
        if cut < 0:
            return None
        with open(path, "r+b") as f:
            f.truncate(size - (len(tail) - cut - 1))
        tail = tail[:cut + 1]
    for line in reversed(tail.splitlines()):
        parts = line.decode("utf-8", "ignore").split(",")
        if len(parts) < 2 or parts[0] == "symbol":
            continue
        try:
            return int(parts[1])
        except ValueError:
            continue
    return None


def run_symbol(sym, out_dir, now_ms):
    path = os.path.join(out_dir, sym + ".csv")
    got = 0
    start = last_open_time(path)
    if start is None:
        mode = "w"
        start = 0
    else:
        mode = "a"
        start = start + 60000
    if start >= now_ms - 120000:
        return sym, got, "已最新"
    buf = []
    f = open(path, mode, encoding="utf-8", newline="")
    try:
        if mode == "w":
            f.write(HEADER)
        cur = start
        while True:
            if stop_flag.is_set():
                break
            k = api(sym, cur)
            if k is None:
                return sym, got, "net-down"
            if not k:
                break
            for row in k:
                buf.append("%s,%d,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s\n" % (
                    sym, row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                    row[7], row[8], row[9], row[10], row[11]))
            got += len(k)
            cur = k[-1][0] + 60000
            if len(k) < PAGE:
                break
            if len(buf) >= 20000:
                f.write("".join(buf)); buf.clear(); f.flush()
                with lock:
                    stat["rows"] += 20000
        if buf:
            f.write("".join(buf)); f.flush()
            with lock:
                stat["rows"] += len(buf)
    finally:
        f.close()
    return sym, got, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:\binance\klines_1m")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-symbols", type=int, default=0)
    a = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if a.symbols:
        syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    else:
        p = os.path.join(root, "data", "symbols_usdt.csv")
        syms = [l.split(",")[0] for l in open(p, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    if a.limit_symbols:
        syms = syms[:a.limit_symbols]

    out_dir = os.path.join(a.out, "1m")
    os.makedirs(out_dir, exist_ok=True)
    prog_path = os.path.join(a.out, "progress.json")
    prog = {}
    if os.path.exists(prog_path):
        try:
            prog = json.load(open(prog_path, encoding="utf-8"))
        except Exception:
            prog = {}
    now_ms = int(time.time() * 1000)

    print("out=%s  symbols=%d  workers=%d  now=%s" % (
        a.out, len(syms), a.workers, time.strftime("%Y-%m-%d %H:%M", time.gmtime(now_ms / 1000))), flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_symbol, s, out_dir, now_ms): s for s in syms}
        for f in as_completed(futs):
            s = futs[f]
            done += 1
            try:
                sym, got, msg = f.result()
                prog[sym] = {"rows_added": got, "msg": msg, "ts": int(time.time())}
            except Exception as e:
                with lock:
                    stat["err"] += 1
                prog[s] = {"rows_added": 0, "msg": "ERR:%r" % (e,), "ts": int(time.time())}
            if done % 5 == 0 or done == len(syms):
                el = time.time() - t0
                with lock:
                    json.dump(prog, open(prog_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                print("[%d/%d] %.0fs  req=%d  err=%d  %.2f MB  %.1f req/s  last=%s" % (
                    done, len(syms), el, stat["req"], stat["err"], stat["bytes"] / 1e6,
                    stat["req"] / max(el, 1e-6), s), flush=True)
    with lock:
        json.dump(prog, open(prog_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("TOTAL %.0fs  req=%d  err=%d  bytes=%.1f MB" % (
        time.time() - t0, stat["req"], stat["err"], stat["bytes"] / 1e6), flush=True)


if __name__ == "__main__":
    main()
