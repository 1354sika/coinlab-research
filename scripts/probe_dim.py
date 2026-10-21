# -*- coding: utf-8 -*-
"""探测：除 K 线外，还有哪些「新观测维度」现在就能拿到
（决定下一步是「继续在 K 线上加算子」还是「必须扩宽可观测世界」）
"""
import json, time, urllib.request

BASE = 'https://data-api.binance.vision'
FAPI = 'https://fapi.binance.com'

targets = [
    ('spot 1m klines（现用）',   BASE + '/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=2'),
    ('spot 1s klines（秒级）',   BASE + '/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=5'),
    ('order book depth20',       BASE + '/api/v3/depth?symbol=BTCUSDT&limit=20'),
    ('bookTicker 最优挂单',      BASE + '/api/v3/ticker/bookTicker?symbol=BTCUSDT'),
    ('aggTrades 逐笔(聚合)',     BASE + '/api/v3/aggTrades?symbol=BTCUSDT&limit=10'),
    ('trades 逐笔(原始)',        BASE + '/api/v3/trades?symbol=BTCUSDT&limit=10'),
    ('24hr ticker',              BASE + '/api/v3/ticker/24hr?symbol=BTCUSDT'),
    ('fapi fundingRate 资金费率', FAPI + '/fapi/v1/fundingRate?symbol=BTCUSDT&limit=5'),
    ('fapi openInterest 持仓量',  FAPI + '/fapi/v1/openInterest?symbol=BTCUSDT'),
    ('fapi premiumIndex 基差',    FAPI + '/fapi/v1/premiumIndex?symbol=BTCUSDT'),
    ('fapi depth 合约订单簿',     FAPI + '/fapi/v1/depth?symbol=BTCUSDT&limit=5'),
]

print('=' * 100)
print('探测：新观测维度可得性')
print('=' * 100)
for name, url in targets:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=12) as r:
            body = r.read(300).decode('utf-8', 'replace')
            print('[OK   %5dms] %-26s %s' % ((time.time() - t0) * 1000, name,
                                             body[:150].replace('\n', ' ')))
    except Exception as e:
        print('[FAIL %5dms] %-26s %s' % ((time.time() - t0) * 1000, name, str(e)[:110]))
print('=' * 100)
print('判读要点：')
print('  · 1s klines 若通 -> 时间分辨率可提升 60 倍（但历史仅最近 1000 根 = 约 17 分钟，必须实时录）')
print('  · depth / bookTicker 若通 -> 能看挂单（庄家痕迹最重的地方），但同样只有实时快照，历史不存')
print('  · aggTrades / trades 若通 -> 逐笔成交，可算大单方向')
print('  · fapi 系列若通 -> 资金费率 / 持仓量（杠杆情绪）')
