# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
D = r'E:\binance\klines_1m'
files = sorted(x for x in os.listdir(D) if x.endswith('.csv'))[:6]
print('files', files)
for f in files:
    try:
        df = pd.read_csv(os.path.join(D, f), usecols=['open_time', 'open', 'close'])
    except Exception as e:
        print(f, 'READ_FAIL', repr(e)[:80]); continue
    c = df['close'].to_numpy(np.float32)
    ret = c / np.roll(c, 15) - 1.0
    ret[:15] = 0.0
    print('%-14s rows=%-9d hit15m5pct=%-6d open_time=%s' %
          (f, len(df), int((ret <= -0.05).sum()), df['open_time'].dtype))
