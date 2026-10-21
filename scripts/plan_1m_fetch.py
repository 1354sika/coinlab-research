# -*- coding: utf-8 -*-
"""1m 全市场拉取范围估算（只读，不下载 K 线本体）

做法：
  1) exchangeInfo 一次拿到所有符号的 onboardDate（上市时间）
  2) 用 symbols_usdt.csv 的 502 个 USDT 现货符号交叉
  3) 按 (now - onboardDate) 估每币 1m 根数 -> 总行数 / 字节 / 请求数
  4) 实测一次 klines 往返，给出真实速率下的耗时区间
输出：evidence/plan_1m_fetch.txt
"""
import json
import os
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
BASE = "https://data-api.binance.vision"
NOW = int(time.time() * 1000)


def get(path, timeout=30):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), time.time() - t0, len(raw)


lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(s)


syms = [l.split(",")[0] for l in open(os.path.join(DATA, "symbols_usdt.csv"), encoding="utf-8").read().splitlines()[1:] if l.strip()]
say("symbols_usdt.csv 符号数 = %d" % len(syms))

info, dt, n = get("/api/v3/exchangeInfo")
say("exchangeInfo: %.2fs, %d bytes, symbols=%d" % (dt, n, len(info["symbols"])))
by = {s["symbol"]: s for s in info["symbols"]}

rows = []
for s in syms:
    d = by.get(s)
    if not d:
        rows.append((s, None, None))
        continue
    ob = d.get("onboardDate")
    rows.append((s, ob, d.get("status")))

miss = [r[0] for r in rows if r[1] is None]
have = [r for r in rows if r[1] is not None]
say("有 onboardDate = %d / 缺失 = %d" % (len(have), len(miss)))
if miss:
    say("缺失样例: " + ",".join(miss[:10]))

# 估：分钟数 = (NOW - onboardDate)/60000，扣 5% 停牌/缺口（实测 60 币 1 年缺 4.4%）
tot_rows = 0
detail = []
for s, ob, st in have:
    mins = max(0, (NOW - ob) // 60000)
    rows_est = int(mins * 0.95)
    tot_rows += rows_est
    detail.append((s, ob, mins / 525600.0, rows_est))

detail.sort(key=lambda x: -x[3])
BYTES_PER_ROW = 140   # 实测 klines_1m_1y.csv: 4,221,011,738 B / 30,150,000 行 ≈ 140
REQ = tot_rows / 1000.0
say("")
say("=== 估算（按 onboardDate 到 now，扣 5%% 缺口）===")
say("总行数      = %,.0f 行".replace(",", "").format(tot_rows) if False else "总行数      = {:,} 行".format(tot_rows))
say("总字节      = {:.1f} GB  (=行数×140B)".format(tot_rows * BYTES_PER_ROW / 1e9))
say("请求数      = {:,} 次 (每 1000 根)".format(int(REQ)))
say("按币年限分布: <=1年 %d, 1-3年 %d, 3-6年 %d, >6年 %d" % (
    sum(1 for d in detail if d[2] <= 1),
    sum(1 for d in detail if 1 < d[2] <= 3),
    sum(1 for d in detail if 3 < d[2] <= 6),
    sum(1 for d in detail if d[2] > 6)))
say("")
say("最长 5 个币:")
for s, ob, yrs, r in detail[:5]:
    say("  %-14s 上市 %s  %.1f 年  %d 行  %.2f GB" % (
        s, time.strftime("%Y-%m-%d", time.gmtime(ob / 1000)), yrs, r, r * BYTES_PER_ROW / 1e9))

# 实测单请求
say("")
say("=== 实测单请求 ===")
for sym in ("BTCUSDT", "ETHUSDT", detail[-1][0] if detail else "BTCUSDT"):
    try:
        k, dt, n = get("/api/v3/klines?symbol=%s&interval=1m&limit=1000" % sym)
        say("  %-12s %.3fs  %d bytes  %d 根  %s ~ %s" % (
            sym, dt, n, len(k),
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(k[0][0] / 1000)),
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(k[-1][0] / 1000))))
    except Exception as e:
        say("  %-12s FAIL %r" % (sym, e))

os.makedirs(EV, exist_ok=True)
open(os.path.join(EV, "plan_1m_fetch.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
say("\nsaved -> evidence/plan_1m_fetch.txt")
