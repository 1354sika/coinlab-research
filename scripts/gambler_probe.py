# -*- coding: utf-8 -*-
"""gambler_probe.py — 现场数据边界探查（只读）
回答：5m 主数据集覆盖到哪一天、多少币、列有哪些；live 数据最新时间
"""
import os, datetime as dt

def ts(ms):
    return dt.datetime.utcfromtimestamp(int(ms) / 1000).strftime('%Y-%m-%d %H:%M')

D = r'D:\中信证券\coinlab\data'
for fn in ['klines_5m_502.csv', 'klines_5m_3y.csv', 'klines_1m_502.csv', 'klines_1m_1y.csv']:
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        print(fn, 'MISSING'); continue
    sz = os.path.getsize(p)
    f = open(p, encoding='utf-8')
    head = f.readline().strip()
    first = f.readline().strip()
    f.seek(max(0, sz - 4000))
    last = f.read().strip().split('\n')[-1]
    # 采样估算行数
    print('=== %s  %.2f GB' % (fn, sz / 1e9))
    print('   head :', head[:130])
    print('   first:', first[:110])
    print('   last :', last[:110])
    try:
        print('   span :', ts(first.split(',')[0]), '->', ts(last.split(',')[0]))
    except Exception as e:
        print('   span ERR', e)

d = os.path.join(D, 'live')
print('\n=== live 目录')
for day in sorted(os.listdir(d)):
    dd = os.path.join(d, day)
    for fn in sorted(os.listdir(dd)):
        p = os.path.join(dd, fn)
        s = os.path.getsize(p)
        g = open(p, encoding='utf-8')
        g.readline()
        g.seek(max(0, s - 600))
        last = g.read().strip().split('\n')[-1]
        print('  %s/%s  %.1f MB | last: %s' % (day, fn, s / 1e6, last[:100]))
