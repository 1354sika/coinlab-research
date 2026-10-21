# -*- coding: utf-8 -*-
"""gambler_bidir_verify · 对 bidir_trade2 的四个正期望格子做稳健性验证

bidir_trade2（job-24708-50）测试段结果：
    A 24h跌>=20%  做多  3,471 笔  +3.44%/笔  中位 -1.41%  胜率 45.6%   ← 反弹
    C 4h 涨>=20%  做空  1,587 笔  +4.49%/笔  中位 +8.87%  胜率 55.0%   ← 见顶
    D 24h跌10~20% 做多 23,954 笔  +1.05%/笔  中位 +0.58%  胜率 52.8%   ← 最健康
    B 24h涨10~20% 做多 35,294 笔  +1.03%/笔  中位 -1.53%  胜率 43.2%
    RAND          做多 15,082 笔  +0.08%/笔

本脚本查四件事（任一不过 = 不可用）：
    ① 币留出：把币按字母分两半，两边都正？
    ② 逐年：2023/2024/2025/2026 每年都正？
    ③ 去 top1%：剔除收益最高的 1% 笔，仍正？（防少数极端笔撑起来）
    ④ 耗时：平均持有多少根 5m / 多少小时
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
HOLD = 576
TP = 0.30; SL = 0.15; FEE = 0.0009

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_bidir_verify · 四个正期望格子的稳健性验证')
log('=' * 116)

RAWS = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAWS.append((sym, g['open_time'].values.astype('int64'),
                     g['close'].values.astype('float64'),
                     g['high'].values.astype('float64'),
                     g['low'].values.astype('float64')))
    del ch; gc.collect()
RAWS.sort(key=lambda x: x[0])
ncoin = len(RAWS)
half = ncoin // 2
log(f'币 {ncoin}   读盘 {time.time()-t0:.0f}s')


def scan(i, c, h, l, long_side):
    n = len(c)
    end = min(i + HOLD, n)
    if end - i < 10:
        return None
    e = c[i]
    for j in range(i + 1, end):
        if long_side:
            if l[j] <= e * (1 - SL):
                return (0, -SL - FEE, j - i)
            if h[j] >= e * (1 + TP):
                return (1, TP - FEE, j - i)
        else:
            if h[j] >= e * (1 + SL):
                return (0, -SL - FEE, j - i)
            if l[j] <= e * (1 - TP):
                return (1, TP - FEE, j - i)
    r = c[end - 1] / e - 1.0
    if not long_side:
        r = -r
    return (1 if r > 0 else 0, r - FEE, end - 1 - i)


CONDS = [
    ('A 24h跌>=20% 做多',  288, -np.inf, -0.20, True),
    ('C 4h涨>=20% 做空',    48,  0.20,   np.inf, False),
    ('D 24h跌10~20% 做多', 288, -0.20,  -0.10, True),
    ('B 24h涨10~20% 做多', 288,  0.10,   0.20, True),
]
# (组名, 币半, 年, 收益%, 胜, 根数)
ROWS = []
rng = np.random.default_rng(11)
CAP = 3000

for ci, (sym, ot, c, h, l) in enumerate(RAWS):
    n = len(c)
    if n < 288 * 4 + HOLD + 2:
        continue
    t_cut = int(n * 0.70)
    lo_i = 300; hi_i = n - HOLD - 1
    if hi_i <= lo_i:
        continue
    idx_all = np.arange(lo_i, hi_i)
    grp = 'H1' if ci < half else 'H2'
    for (name, lag, lo, hi, side) in CONDS:
        with np.errstate(invalid='ignore', divide='ignore'):
            r = c / np.roll(c, lag) - 1.0
        m = (r >= lo) & (r < hi)
        hit = idx_all[m[idx_all]]
        if len(hit) == 0:
            continue
        hit = hit[hit >= t_cut]          # 只看测试段
        if len(hit) == 0:
            continue
        if len(hit) > CAP:
            hit = rng.choice(hit, CAP, replace=False)
        for i in hit:
            a = scan(i, c, h, l, side)
            if a is None:
                continue
            yr = int(pd.Timestamp(ot[i], unit='ms').year)
            ROWS.append((name, grp, yr, a[1] * 100, a[0], a[2]))
    del c, h, l
    gc.collect()

log(f'扫描完成 {time.time()-t0:.0f}s   样本 {len(ROWS):,}')

df = pd.DataFrame(ROWS, columns=['cond', 'half', 'year', 'ret', 'win', 'bars'])

log('')
log('=' * 116)
log('【一】总览（测试段）')
log('=' * 116)
log(f'{"条件":<22} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"平均持有h":>10}')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50:
        log(f'{name:<22} {len(g):>8,}  (样本不足)')
        continue
    log(f'{name:<22} {len(g):>8,} {g.ret.mean():>8.2f}% {g.ret.median():>7.2f}% {100*g.win.mean():>6.1f}% {g.bars.mean()*5/60:>9.1f}')

log('')
log('=' * 116)
log('【二】币留出（两半都要正）')
log('=' * 116)
log(f'{"条件":<22} {"H1笔":>8} {"H1期望%":>9} {"H2笔":>8} {"H2期望%":>9}  判定')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50:
        continue
    a = g[g.half == 'H1']; b = g[g.half == 'H2']
    if len(a) < 20 or len(b) < 20:
        log(f'{name:<22} (单侧样本不足)')
        continue
    ok = '★通过' if (a.ret.mean() > 0 and b.ret.mean() > 0) else '✗不稳'
    log(f'{name:<22} {len(a):>8,} {a.ret.mean():>8.2f}% {len(b):>8,} {b.ret.mean():>8.2f}%  {ok}')

log('')
log('=' * 116)
log('【三】逐年（每年都要正）')
log('=' * 116)
years = sorted(df.year.unique())
hdr = f'{"条件":<22} ' + ' '.join(f'{y:>10}' for y in years) + '  判定'
log(hdr)
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50:
        continue
    cells = []; allpos = True; enough = True
    for y in years:
        gy = g[g.year == y]
        if len(gy) < 20:
            cells.append(f'{"n/a":>10}'); continue
        v = gy.ret.mean()
        if v <= 0: allpos = False
        cells.append(f'{v:>9.2f}%')
    ok = '★通过' if allpos else '✗有负年'
    log(f'{name:<22} ' + ' '.join(cells) + f'  {ok}')

log('')
log('=' * 116)
log('【四】去 top1% 极端笔（剔除收益最高的 1%）')
log('=' * 116)
log(f'{"条件":<22} {"原期望%":>9} {"去top1%期望%":>13}  判定')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 100:
        continue
    q = g.ret.quantile(0.99)
    g2 = g[g.ret < q]
    ok = '★通过' if g2.ret.mean() > 0 else '✗靠极端笔'
    log(f'{name:<22} {g.ret.mean():>8.2f}% {g2.ret.mean():>12.2f}%  {ok}')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bidir_verify_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
df.to_csv(os.path.join(OUT, 'bidir_verify_rows.csv'), index=False, encoding='utf-8-sig')
log('')
log(f'报告已写 {os.path.join(OUT, "bidir_verify_summary.txt")}')
log(f'TOTAL {time.time()-t0:.0f}s')
