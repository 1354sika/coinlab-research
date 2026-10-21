# -*- coding: utf-8 -*-
import os
for d in [r'E:\binance\klines_1m', r'E:\binance\klines_1m\1m']:
    print('DIR', d, 'isdir=', os.path.isdir(d))
    if os.path.isdir(d):
        try:
            items = os.listdir(d)
            print('   count=%d  first=%s' % (len(items), items[:8]))
            csvs = [x for x in items if x.lower().endswith('.csv')]
            print('   csv=%d' % len(csvs))
        except Exception as e:
            print('   ERR', e)
