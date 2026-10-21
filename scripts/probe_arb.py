# -*- coding: utf-8 -*-
"""探测「结构性套利」需要的行情/费率接口可达性"""
import requests

T = [
    ("binance_fapi_time", "https://fapi.binance.com/fapi/v1/time"),
    ("binance_funding", "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=3"),
    ("binance_premium", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"),
    ("binance_vision_spot", "https://data-api.binance.vision/api/v3/time"),
    ("okx_funding", "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"),
    ("okx_basis", "https://www.okx.com/api/v5/public/mark-price?instType=SWAP&instId=BTC-USDT-SWAP"),
    ("bybit_funding", "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=3"),
    ("gate_funding", "https://api.gateio.ws/api/v4/futures/usdt/funding_rate?contract=BTC_USDT&limit=3"),
    ("bitget_funding", "https://api.bitget.com/api/v2/mix/market/current-fund-rate?symbol=BTCUSDT&productType=usdt-futures"),
    ("hyperliquid", "https://api.hyperliquid.xyz/info"),
    ("coingecko", "https://api.coingecko.com/api/v3/ping"),
]
for n, u in T:
    try:
        r = requests.get(u, timeout=12)
        print("%-22s %s %s" % (n, r.status_code, r.text[:150].replace("\n", " ")), flush=True)
    except Exception as e:
        print("%-22s FAIL %s" % (n, str(e)[:90]), flush=True)
