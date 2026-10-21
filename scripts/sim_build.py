# -*- coding: utf-8 -*-
"""从 klines_5m_3y.csv 切出模拟器用的币种数据（只读源文件，写 sim/data/*.json）"""
import pandas as pd, json, os, time, sys

SRC = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/sim/data'
os.makedirs(OUT, exist_ok=True)

SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT',
        'ADAUSDT','LINKUSDT','AVAXUSDT','LTCUSDT','TRXUSDT','DOTUSDT']
N_BARS = 25920  # 90 天 × 288 根/天

t0 = time.time()
df = pd.read_csv(SRC, usecols=['symbol','open_time','open','high','low','close','volume'])
print('读入 %d 行 %.1fs' % (len(df), time.time()-t0))

meta = {}
for s in SYMS:
    d = df[df.symbol == s]
    if len(d) == 0:
        print('  缺币', s); continue
    d = d.tail(N_BARS)
    rec = {
        'symbol': s,
        'tf': '5m',
        't': d.open_time.astype('int64').tolist(),
        'o': d.open.round(8).tolist(),
        'h': d.high.round(8).tolist(),
        'l': d.low.round(8).tolist(),
        'c': d.close.round(8).tolist(),
        'v': d.volume.round(4).tolist(),
    }
    p = os.path.join(OUT, s + '.json')
    with open(p, 'w') as fh:
        json.dump(rec, fh, separators=(',',':'))
    meta[s] = {'bars': len(d), 'from': int(d.open_time.iloc[0]), 'to': int(d.open_time.iloc[-1]),
               'kb': round(os.path.getsize(p)/1024)}
    print('  %-9s %6d 根  %s ~ %s  %d KB' % (
        s, len(d),
        pd.to_datetime(d.open_time.iloc[0], unit='ms'),
        pd.to_datetime(d.open_time.iloc[-1], unit='ms'),
        meta[s]['kb']))

with open(os.path.join(OUT, 'index.json'), 'w') as fh:
    json.dump(meta, fh, indent=1)
print('TOTAL %.1fs' % (time.time()-t0))
