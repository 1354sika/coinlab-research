# -*- coding: utf-8 -*-
"""1m 全市场拉取范围实测版 v2（只读，不下载 K 线本体）

v1 失败原因：data-api.binance.vision 的 exchangeInfo 不返回 onboardDate。
改法：对每个符号发一次 `klines?interval=1m&startTime=0&limit=1` 拿首根 K 线时间（=真实上市起点），
      顺带就是真实的请求速率实测。
输出：evidence/plan_1m_fetch.txt
"""
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
BASE = "https://data-api.binance.vision"
NOW = int(time.time() * 1000)
WORKERS = 8

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(s)


def get(path, timeout=30):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), time.time() - t0


syms = [l.split(",")[0] for l in open(os.path.join(DATA, "symbols_usdt.csv"), encoding="utf-8").read().splitlines()[1:] if l.strip()]
say("symbols_usdt.csv 符号数 = %d" % len(syms))

info, dt = get("/api/v3/exchangeInfo")
say("exchangeInfo: %.2fs symbols=%d" % (dt, len(info["symbols"])))
sample = info["symbols"][0]
say("字段样例 key: " + ",".join(sorted(sample.keys())))
say("onboardDate 存在? %s" % ("onboardDate" in sample))

# 各计价口径统计
cnt = {}
for s in info["symbols"]:
    if s.get("status") == "TRADING":
        cnt[s["quoteAsset"]] = cnt.get(s["quoteAsset"], 0) + 1
say("TRADING 按计价资产: " + "  ".join("%s=%d" % kv for kv in sorted(cnt.items(), key=lambda x: -x[1])[:8]))

lock = threading.Lock()
first = {}
t0 = time.time()
done = [0]


def one(sym):
    try:
        k, _ = get("/api/v3/klines?symbol=%s&interval=1m&startTime=0&limit=1" % sym, timeout=20)
        with lock:
            done[0] += 1
            if k:
                first[sym] = k[0][0]
            return True
    except Exception:
        with lock:
            done[0] += 1
        return False


say("")
say("=== 探测每个币的首根 1m K 线（%d 并发）===" % WORKERS)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(one, s) for s in syms]
    for f in as_completed(futs):
        f.result()
el = time.time() - t0
ok = len(first)
say("完成 %d 个，成功 %d，用时 %.1fs -> %.2f 请求/秒" % (len(syms), ok, el, len(syms) / el))

# 估算：分钟数 = (NOW - first)/60000，扣 5% 缺口（实测 60 币 1 年缺 4.4%）
tot = 0
detail = []
for s, ob in first.items():
    mins = max(0, (NOW - ob) // 60000)
    rows_est = int(mins * 0.95)
    tot += rows_est
    detail.append((s, ob, mins / 525600.0, rows_est))
detail.sort(key=lambda x: -x[3])

BYTES_PER_ROW = 140
REQ = tot / 1000.0
say("")
say("=== 估算 ===")
say("总行数 = {:,} 行".format(tot))
say("总字节 = {:.1f} GB".format(tot * BYTES_PER_ROW / 1e9))
say("请求数 = {:,} 次".format(int(REQ)))
say("年限分布: <=1年 %d, 1-3年 %d, 3-6年 %d, >6年 %d" % (
    sum(1 for d in detail if d[2] <= 1),
    sum(1 for d in detail if 1 < d[2] <= 3),
    sum(1 for d in detail if 3 < d[2] <= 6),
    sum(1 for d in detail if d[2] > 6)))
say("最早上市 5 个:")
for s, ob, yrs, r in detail[:5]:
    say("  %-14s %s  %.1f 年  %d 行  %.2f GB" % (
        s, time.strftime("%Y-%m-%d", time.gmtime(ob / 1000)), yrs, r, r * BYTES_PER_ROW / 1e9))
say("最晚上市 5 个:")
for s, ob, yrs, r in detail[-5:]:
    say("  %-14s %s  %.2f 年  %d 行" % (s, time.strftime("%Y-%m-%d", time.gmtime(ob / 1000)), yrs, r))

# 实测下载吞吐：连拉 20 次 1000 根，测 MB/s
say("")
say("=== 下载吞吐实测（20 次 × 1000 根）===")
sym = "BTCUSDT"
t0 = time.time()
byts = 0
for i in range(20):
    k, _ = get("/api/v3/klines?symbol=%s&interval=1m&limit=1000" % sym)
    byts += len(k)
el = time.time() - t0
say("单线程 %.2f 请求/秒  %.2f MB/s  (%.1f KB/请求)" % (20 / el, byts / el / 1e6, byts / 20 / 1e3))
say("=> 单线程预计总耗时 {:.1f} 小时；8 并发理论 {:.1f} 小时（受 6000 weight/min 限制）".format(
    REQ / (20 / el) / 3600, REQ / (20 / el) / 3600 / 8))

os.makedirs(EV, exist_ok=True)
open(os.path.join(EV, "plan_1m_fetch.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
say("\nsaved -> evidence/plan_1m_fetch.txt")
