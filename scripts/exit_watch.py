# -*- coding: utf-8 -*-
"""出场提醒：定时扫 watchlist.json 里的持仓，P(新高) 跌破 θ 就报警。

watchlist.json 格式（改 entry 为你自己的成本价，hold 为已持有 5m 根数）：
  {"BTCUSDT": {"entry": 82200, "hold": 24}, ...}

用法：
  python -X utf8 exit_watch.py --once      # 扫一次（先看效果）
  python -X utf8 exit_watch.py             # 每 5 分钟循环
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'exit_q2.pkl')
META = os.path.join(HERE, '..', 'models', 'exit_q2.json')
WL = os.path.join(HERE, '..', 'watchlist.json')
API = 'https://data-api.binance.vision/api/v3/klines'
H, W = 96, 24


def scan(mdl, sym, entry, hold):
    b = requests.get(API, params={'symbol': sym, 'interval': '5m', 'limit': H + W + 5}, timeout=20).json()
    c = np.array([float(x[4]) for x in b])
    v = np.array([float(x[5]) for x in b])
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1
    vma = pd.Series(v).rolling(24).mean().to_numpy()
    volr = v / vma
    g = c[-1] / float(entry) - 1
    x = np.concatenate([ret[-W:], volr[-W:], [g], [hold]])
    x = np.nan_to_num(x)[None, :]
    p = float(mdl.predict_proba(x)[0, 1])
    return c[-1], g, p


def main():
    mdl = joblib.load(MODEL)
    theta = json.load(open(META, encoding='utf-8'))['theta']
    wl = json.load(open(WL, encoding='utf-8'))
    once = '--once' in sys.argv
    while True:
        print('--- %s  (theta=%.2f)' % (time.strftime('%H:%M:%S'), theta), flush=True)
        for sym, cfg in wl.items():
            try:
                px, g, p = scan(mdl, sym, cfg['entry'], int(cfg.get('hold', 24)))
                flag = '>>> 该走了 <<<' if p < theta else '持有'
                print('%-10s %12.4f  浮盈 %+7.2f%%  P=%.3f  %s' % (sym, px, g * 100, p, flag), flush=True)
            except Exception as e:
                print('%-10s ERR %s' % (sym, str(e)[:80]), flush=True)
        if once:
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
