# -*- coding: utf-8 -*-
"""1m 分钟数据「长跨」取数：N 币 x DAYS 天（币安 vision，可达）
目的（用户选 B）：
  ① 现有 502 币只有 60 天，无法判断「量比>8」门槛是长期规律还是这 60 天的巧合；
  ② 需要覆盖不同市场阶段（涨/跌/横盘）才能谈「能不能长期用」。
做法：读 klines_1d_top502.csv 的 symbol（按 24h 成交额排序取前 N），
      从 DAYS 天前分页取 1m（每次 1000 根），断点续传，多线程。
输出：data/klines_1m_1y.csv（追加式，可反复续跑）
"""
import os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "klines_1m_1y.csv")
SYMS = os.path.join(DATA, "klines_1d_top502.csv")
BASE = "https://data-api.binance.vision/api/v3/klines"
DAYS = int(os.environ.get("DAYS", "365"))
NCOIN = int(os.environ.get("NCOIN", "60"))
WORKERS = int(os.environ.get("WORKERS", "6"))
COLS = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']

lock = threading.Lock()
print('DAYS=%d NCOIN=%d OUT=%s' % (DAYS, NCOIN, OUT), flush=True)

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
        print('resume: %d symbols already present' % len(done), flush=True)
    except Exception as e:
        print('resume read err', e, flush=True)


def fetch_one(sym):
    rows = []
    cur = start_ms
    sess = requests.Session()
    while cur < now:
        for attempt in range(4):
            try:
                r = sess.get(BASE, params={'symbol': sym, 'interval': '1m',
                                           'startTime': cur, 'limit': 1000}, timeout=25)
                if r.status_code == 429:
                    time.sleep(3 + attempt * 3)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 3:
                    return sym, rows, False
                time.sleep(1 + attempt)
        else:
            return sym, rows, False
        if not data:
            break
        for d in data:
            rows.append([sym] + d[:12])
        last = data[-1][0]
        if last <= cur:
            cur += 1000 * 60000
        else:
            cur = last + 60000
        time.sleep(0.03)
    return sym, rows, True


t0 = time.time()
todo = [s for s in syms if s not in done]
print('todo=%d' % len(todo), flush=True)
total_rows = 0
wrote_header = os.path.exists(OUT)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(fetch_one, s): s for s in todo}
    for i, fu in enumerate(as_completed(futs), 1):
        sym, rows, ok = fu.result()
        if not ok:
            print('  !! %s FAILED/partial rows=%d' % (sym, len(rows)), flush=True)
            continue
        with lock:
            pd.DataFrame(rows, columns=COLS).to_csv(
                OUT, mode='a', header=not wrote_header, index=False, encoding='utf-8')
            wrote_header = True
            total_rows += len(rows)
        if i % 5 == 0 or i == len(todo):
            print('  %d/%d %s rows=%d elapsed=%.0fs' % (
                i, len(todo), sym, len(rows), time.time() - t0), flush=True)

print('DONE rows=%d elapsed=%.0fs file=%s' % (total_rows, time.time() - t0, OUT), flush=True)
print('bars/symbol (target for full range): %d' % (DAYS * 1440), flush=True)
