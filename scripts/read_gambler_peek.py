# -*- coding: utf-8 -*-
import os, json
D = r'D:\中信证券\coinlab\evidence'
picks = ['gambler_onecard.txt', 'gambler_psych.txt', 'gambler_stophunt.csv',
         'gambler_odds_REAL.json', 'gambler_third.txt', 'gambler_memory.txt']
for f in picks:
    p = os.path.join(D, f)
    print('=' * 70)
    print('FILE', f, os.path.getsize(p) if os.path.exists(p) else 'MISSING')
    if not os.path.exists(p):
        continue
    try:
        with open(p, encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                if i >= 14:
                    break
                print(line.rstrip()[:200])
    except Exception as e:
        print('ERR', e)
