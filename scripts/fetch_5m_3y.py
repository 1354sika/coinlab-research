# -*- coding: utf-8 -*-
"""跨周期验证取数：60 币 x 3 年 5m（含 2023-2026 的牛/熊/横盘）
动机：现有 1 年 1m 数据只覆盖一个市场阶段；要判断「急跌+巨量」是不是全天候机制，
      必须覆盖多个阶段（上涨期/下跌期/横盘期）。
为什么用 5m 而不是 1m：1m x 3 年 x 60 币 = 1.57 亿行（~30GB、请求量 9.5 万、约 4 小时）；
                          5m x 3 年 x 60 币 = 1890 万行（~3GB、请求量 1.9 万、约 40 分钟）。
                          本任务只判「机制是否跨阶段有效」，不需要分钟级精度。
输出：data/klines_5m_3y.csv（追加式，可续跑）
"""
import os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, os.environ.get("OUTNAME", "klines_5m_3y.csv"))
SYMS = os.path.join(DATA, "klines_1d_top502.csv")
BASE = "https://data-api.binance.vision/api/v3/klines"
DAYS = int(os.environ.get("DAYS", "1095"))
NCOIN = int(os.environ.get("NCOIN", "60"))
WORKERS = int(os.environ.get("WORKERS", "6"))
COLS = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']

lock = threading.Lock()
print('DAYS=%d NCOIN=%d interval=5m OUT=%s' % (DAYS, NCOIN, OUT), flush=True)

sdf = pd.read_csv(SYMS, usecols=['symbol', 'quote_volume'])
sdf = sdf.sort_values('quote_volume', ascending=False).drop_duplicates('symbol')
syms = [s for s in sdf.symbol.head(NCOIN).tolist() if s.endswith('USDT')]
print('symbols=%d' % len(syms), flush=True)

now = int(time.time() * 1000)
start_ms = now - DAYS * 86400 * 1000

done = set()
if os.path.exists(OUT):
    try:
        ex = pd.read_csv(OUT, usecols=['symbol'])['symbol'].unique().tolist()
        done = set(ex)
        print('resume: %d symbols present' % len(done), flush=True)
    except Exception as e:
        print('resume err', e, flush=True)


def fetch_one(sym):
    rows = []
    cur = start_ms
    sess = requests.Session()
    while cur < now:
        data = None
        for attempt in range(4):
            try:
                r = sess.get(BASE, params={'symbol': sym, 'interval': '5m',
                                           'startTime': cur, 'limit': 1000}, timeout=30)
                if r.status_code == 429:
                    time.sleep(4 + attempt * 4); continue
                r.raise_for_status()
                data = r.json(); break
            except Exception:
                if attempt == 3:
                    return sym, rows, False
                time.sleep(1 + attempt)
        if not data:
            break
        for d in data:
            rows.append([sym] + d[:12])
        last = data[-1][0]
        cur = last + 300000 if last > cur else cur + 1000 * 300000
        time.sleep(0.03)
    return sym, rows, True


t0 = time.time()
todo = [s for s in syms if s not in done]
print('todo=%d' % len(todo), flush=True)
total = 0
header = os.path.exists(OUT)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(fetch_one, s): s for s in todo}
    for i, fu in enumerate(as_completed(futs), 1):
        sym, rows, ok = fu.result()
        if not ok:
            print('  !! %s FAILED rows=%d' % (sym, len(rows)), flush=True); continue
        with lock:
            pd.DataFrame(rows, columns=COLS).to_csv(
                OUT, mode='a', header=not header, index=False, encoding='utf-8')
            header = True
            total += len(rows)
        if i % 5 == 0 or i == len(todo):
            print('  %d/%d %s rows=%d elapsed=%.0fs' % (
                i, len(todo), sym, len(rows), time.time() - t0), flush=True)

print('DONE rows=%d elapsed=%.0fs file=%s' % (total, time.time() - t0, OUT), flush=True)
print('bars/symbol target = %d' % (DAYS * 288), flush=True)
