# -*- coding: utf-8 -*-
"""扫 E 盘 1m 全部 CSV 的末尾时间，查断档/不完整。只读。"""
import glob, os, datetime

D = r'E:\binance\klines_1m\1m'
CUT = datetime.datetime(2026, 10, 10)

rows, bad = [], []
for p in glob.glob(os.path.join(D, '*.csv')):
    sz = os.path.getsize(p)
    name = os.path.basename(p)[:-4]
    try:
        with open(p, 'rb') as f:
            f.seek(max(0, sz - 8192))
            tail = f.read().decode('utf-8', 'ignore').strip().split('\n')
        parts = tail[-1].split(',')
        t = int(parts[1])
        rows.append((name, t, sz))
    except Exception as e:
        bad.append((name, repr(e)[:60]))

cut_ms = int(CUT.timestamp() * 1000)
stale = sorted([r for r in rows if r[1] < cut_ms], key=lambda x: x[1])
rows.sort(key=lambda x: x[1])

def dt(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).date().isoformat()

print('总文件 %d  |  解析失败 %d  |  总字节 %.2f GB' % (
    len(rows), len(bad), sum(r[2] for r in rows) / 1e9))
print('末尾时间最新 5 个:', [(r[0], dt(r[1])) for r in rows[-5:]])
print('末尾时间最老 5 个:', [(r[0], dt(r[1])) for r in rows[:5]])
print('断档(<2026-10-10) 共 %d 个 / %d' % (len(stale), len(rows)))
for r in stale[:40]:
    print('   %-16s 末 %s  %6.1f MB' % (r[0], dt(r[1]), r[2] / 1e6))
if len(stale) > 40:
    print('   …另有 %d 个' % (len(stale) - 40))
if bad:
    print('解析失败明细:', bad[:10])
