# -*- coding: utf-8 -*-
"""《盘》v0.1 —— 一局完整过程（你 vs 老张）

不是设计文档，是能跑的一局。用来看"玩起来是什么样"。
"""
import random

random.seed(7)

price = 100.0
me = {'cash': 1000.0, 'stock': 0.0}
lao = {'cash': 0.0, 'stock': 100.0}
lao_sold = False
L = []
L.append('=' * 58)
L.append('《盘》v0.1   你 vs 老张（庄家）')
L.append('=' * 58)
L.append('你：钱 1000  货 0        老张：货 100  钱 0')
L.append('')

for t in range(1, 16):
    if not lao_sold:
        drift = random.uniform(0.005, 0.06)
    else:
        drift = random.uniform(-0.10, 0.01)
    price *= (1.0 + drift)
    arrow = '▲' if drift > 0 else '▼'
    L.append('第 %2d 回合   盘 %.1f %s' % (t, price, arrow))

    if not lao_sold and me['cash'] >= price:
        qty = int(40 / price) or 1
        if me['cash'] >= qty * price:
            me['cash'] -= qty * price
            me['stock'] += qty
            L.append('   你 → 接 %d 份 @ %.1f' % (qty, price))
        else:
            L.append('   你 → 等')
    elif lao_sold and me['stock'] > 0:
        qty = me['stock']
        me['stock'] = 0
        me['cash'] += qty * price
        L.append('   你 → 抛 %d 份 @ %.1f' % (qty, price))
    else:
        L.append('   你 → 等')

    if (not lao_sold) and price >= 128:
        lao_sold = True
        lao['cash'] += lao['stock'] * price
        L.append('   >>> 老张出货：抛 %d 份 @ %.1f' % (lao['stock'], price))
        lao['stock'] = 0

    L.append('   你 钱%7.0f 货%4.0f  |  老张 钱%7.0f 货%4.0f' % (
        me['cash'], me['stock'], lao['cash'], lao['stock']))
    L.append('')

me_total = me['cash'] + me['stock'] * price
lao_total = lao['cash'] + lao['stock'] * price
L.append('=' * 58)
L.append('收盘 %.1f' % price)
L.append('你 %.0f   vs   老张 %.0f   →  %s' % (
    me_total, lao_total, '你赢' if me_total > lao_total else '你输'))
L.append('=' * 58)

out = '\n'.join(L)
print(out)
with open(r'D:\中信证券\coinlab\evidence\game_proto_run1.txt', 'w', encoding='utf-8') as f:
    f.write(out)
