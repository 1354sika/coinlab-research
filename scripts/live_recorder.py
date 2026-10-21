# -*- coding: utf-8 -*-
"""实时数据录制器：把「只有现在、没有过去」的三个维度记下来
  ① 1s K 线（每 5 分钟增量拉一次，每次 1000 根 = 约 17 分钟覆盖）
  ② 订单簿 depth20 快照（每 10 秒一轮，轮流覆盖所有目标币）
  ③ aggTrades 逐笔（每 1 分钟拉最新 1000 笔）
落盘：D:\\中信证券\\coinlab\\data\\live\\<日期>\\*.csv（追加写，断点续传）
速率：控制在 Binance weight 限值内（1200/分钟）
"""
import os, time, json, csv, urllib.request, datetime, sys

DATA = r'D:\中信证券\coinlab\data'
LIVE = os.path.join(DATA, 'live')
BASE = 'https://data-api.binance.vision'

# 目标币（主流 30 个，按你已有数据里的常见币）
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT',
        'AVAXUSDT','LINKUSDT','DOTUSDT','TRXUSDT','MATICUSDT','LTCUSDT','ATOMUSDT',
        'UNIUSDT','APTUSDT','ARBUSDT','OPUSDT','SUIUSDT','PEPEUSDT','WIFUSDT',
        'NEARUSDT','INJUSDT','TIAUSDT','SEIUSDT','FILUSDT','AAVEUSDT','ETCUSDT',
        'HBARUSDT','ALGOUSDT']

def get(url, timeout=10):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def append_csv(path, header, rows):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)

def day_dir(ts=None):
    d = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    return os.path.join(LIVE, d)

def log(*a):
    print(datetime.datetime.utcnow().strftime('%H:%M:%S'), *a, flush=True)

def cycle_1s():
    """1s K 线：每个币拉最近 200 根"""
    for s in SYMS:
        try:
            d = get(BASE + '/api/v3/klines?symbol=%s&interval=1s&limit=200' % s)
            rows = [[x[0], s] + x[1:7] for x in d]     # open_time, symbol, o,h,l,c,v
            append_csv(os.path.join(day_dir(), 'klines_1s.csv'),
                       ['open_time','symbol','open','high','low','close','volume'], rows)
        except Exception as e:
            log('1s FAIL', s, str(e)[:60])

def cycle_trades():
    """aggTrades：每个币拉最新 1000 笔"""
    for s in SYMS:
        try:
            d = get(BASE + '/api/v3/aggTrades?symbol=%s&limit=1000' % s)
            rows = [[x['a'], s, x['p'], x['q'], x['T'], x['m']] for x in d]
            append_csv(os.path.join(day_dir(), 'aggTrades.csv'),
                       ['agg_id','symbol','price','qty','time','is_buyer_maker'], rows)
        except Exception as e:
            log('trades FAIL', s, str(e)[:60])

def cycle_depth():
    """订单簿 depth20：每 10 秒只在 1 个币上做（轮流），避免超限"""
    s = SYMS[int(time.time() / 10) % len(SYMS)]
    try:
        d = get(BASE + '/api/v3/depth?symbol=%s&limit=20' % s)
        ts = d.get('lastUpdateId', 0)
        now = int(time.time() * 1000)
        bids = [[now, s, ts, 'bid', i, p, q] for i, (p, q) in enumerate(d['bids'])]
        asks = [[now, s, ts, 'ask', i, p, q] for i, (p, q) in enumerate(d['asks'])]
        append_csv(os.path.join(day_dir(), 'depth20.csv'),
                   ['ts_ms','symbol','last_update_id','side','level','price','qty'], bids + asks)
    except Exception as e:
        log('depth FAIL', s, str(e)[:60])

def main():
    log('recorder start; symbols =', len(SYMS))
    last_1s = 0.0
    last_tr = 0.0
    last_dp = 0.0
    n = 0
    while True:
        now = time.time()
        try:
            if now - last_1s >= 300:       # 每 5 分钟
                cycle_1s(); last_1s = now; log('1s done')
            if now - last_tr >= 60:        # 每 1 分钟
                cycle_trades(); last_tr = now; log('trades done')
            if now - last_dp >= 10:        # 每 10 秒
                cycle_depth(); last_dp = now
        except Exception as e:
            log('cycle ERR', str(e)[:80])
        n += 1
        if n % 60 == 0:
            log('heartbeat', n, 'cycles')
        time.sleep(2)

if __name__ == '__main__':
    main()
