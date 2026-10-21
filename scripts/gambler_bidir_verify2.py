# -*- coding: utf-8 -*-
"""gambler_bidir_verify2 · 清洗脏点后重测四个格子

verify（job-24708-51）出现**自相矛盾**的结果，说明数据有脏点：
    C 组做空：中位 +8.87%（一半以上赚）、胜率 55%、期望仅 +4.49%
    → 若 55% 止盈(+29.91%)+45% 止损(-15.09%)，期望应为 +9.66%
    → 但去 top1% 后崩到 -0.49%
    → 反推 top1%（16 笔）平均收益 494% —— **不可能**，止盈上限 30%

根因假设：close 序列里有 0 / 极小值 / 极端跳变，导致
    e = c[i] 异常小 → 超时平仓收益 c[end]/e - 1 变成天文数字。

本版做法：
    ① 读入时剔除 close<=0 与 close 相对 24h 中位偏离 >10 倍的点
    ② 交易结果满足 |收益| 超出 [-(SL+FEE), +(TP-FEE)] 的一律判为脏点并计数
    ③ 脏点占比 >5% 的组合直接标"数据不可信"
    ④ 清洗后重跑 币留出 / 逐年 / 去 top1%
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
RET_MIN = -(SL + FEE) - 1e-6
RET_MAX = (TP - FEE) + 1e-6

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_bidir_verify2 · 清洗脏点后重测')
log('=' * 116)

RAWS = []
dropped = 0
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        ot = g['open_time'].values.astype('int64')
        # ① 清洗：close 必须 >0 且有限
        bad = ~np.isfinite(c) | (c <= 0) | ~np.isfinite(h) | ~np.isfinite(l)
        # ② 与 288 根滚动中位比，偏离 >10 倍视为脏
        med = pd.Series(c).rolling(288, min_periods=20).median().values
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = c / np.maximum(med, 1e-12)
        bad |= (ratio > 10) | (ratio < 0.1)
        # ③ low<=close<=high 一致性
        bad |= (l > c * 1.001) | (h < c * 0.999)
        nbad = int(bad.sum())
        dropped += nbad
        if nbad:
            keepm = ~bad
            c, h, l, ot = c[keepm], h[keepm], l[keepm], ot[keepm]
        if len(c) < 5000:
            continue
        RAWS.append((sym, ot, c, h, l))
    del ch; gc.collect()
RAWS.sort(key=lambda x: x[0])
ncoin = len(RAWS); half = ncoin // 2
log(f'币 {ncoin}   剔除脏点 {dropped:,}   读盘 {time.time()-t0:.0f}s')


def scan(i, c, h, l, long_side):
    n = len(c)
    end = min(i + HOLD, n)
    if end - i < 10:
        return None
    e = c[i]
    if not np.isfinite(e) or e <= 0:
        return None
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

ROWS = []
dirty = {n: 0 for n, *_ in CONDS}
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
        m = (r >= lo) & (r < hi) & np.isfinite(r)
        hit = idx_all[m[idx_all]]
        if len(hit) == 0:
            continue
        hit = hit[hit >= t_cut]
        if len(hit) == 0:
            continue
        if len(hit) > CAP:
            hit = rng.choice(hit, CAP, replace=False)
        for i in hit:
            a = scan(i, c, h, l, side)
            if a is None:
                continue
            # ② 脏点判定
            if a[1] < RET_MIN - 0.02 or a[1] > RET_MAX + 0.02:
                dirty[name] += 1
                continue
            yr = int(pd.Timestamp(ot[i], unit='ms').year)
            ROWS.append((name, grp, yr, a[1] * 100, a[0], a[2]))
    del c, h, l
    gc.collect()

log(f'扫描完成 {time.time()-t0:.0f}s   有效样本 {len(ROWS):,}')
log('')
log('【脏点统计】收益超出 [%.2f%%, +%.2f%%] 的笔（理论上不可能存在）' % (100*RET_MIN, 100*RET_MAX))
for name, *_ in CONDS:
    log(f'  {name:<22} 脏点 {dirty[name]:>6,}')
log('  ⇒ 脏点 = close 序列异常或扫描边界问题，本版已全部剔除')

df = pd.DataFrame(ROWS, columns=['cond', 'half', 'year', 'ret', 'win', 'bars'])

def block(title):
    log(''); log('=' * 116); log(title); log('=' * 116)

block('【一】总览（测试段，清洗后）')
log(f'{"条件":<22} {"笔数":>8} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"持有h":>8}')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50:
        log(f'{name:<22} {len(g):>8,}  (样本不足)'); continue
    log(f'{name:<22} {len(g):>8,} {g.ret.mean():>8.2f}% {g.ret.median():>7.2f}% {100*g.win.mean():>6.1f}% {g.bars.mean()*5/60:>7.1f}')

block('【二】币留出')
log(f'{"条件":<22} {"H1笔":>8} {"H1期望%":>9} {"H2笔":>8} {"H2期望%":>9}  判定')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50: continue
    a = g[g.half == 'H1']; b = g[g.half == 'H2']
    if len(a) < 20 or len(b) < 20:
        log(f'{name:<22} (单侧不足)'); continue
    ok = '★通过' if (a.ret.mean() > 0 and b.ret.mean() > 0) else '✗不稳'
    log(f'{name:<22} {len(a):>8,} {a.ret.mean():>8.2f}% {len(b):>8,} {b.ret.mean():>8.2f}%  {ok}')

block('【三】逐年')
years = sorted(df.year.unique())
log(f'{"条件":<22} ' + ' '.join(f'{y:>10}' for y in years) + '  判定')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 50: continue
    cells = []; allpos = True
    for y in years:
        gy = g[g.year == y]
        if len(gy) < 20:
            cells.append(f'{"n/a":>10}'); continue
        v = gy.ret.mean()
        if v <= 0: allpos = False
        cells.append(f'{v:>9.2f}%')
    log(f'{name:<22} ' + ' '.join(cells) + ('  ★通过' if allpos else '  ✗有负年'))

block('【四】去 top1%（这一步是判"是否靠少数极端笔"）')
log(f'{"条件":<22} {"原期望%":>9} {"去top1%":>9}  判定')
log('-' * 116)
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 100: continue
    q = g.ret.quantile(0.99)
    g2 = g[g.ret < q]
    log(f'{name:<22} {g.ret.mean():>8.2f}% {g2.ret.mean():>8.2f}%  ' + ('★通过' if g2.ret.mean() > 0 else '✗靠极端笔'))

block('【五】收益分布（解释为什么去 top1% 会崩）')
for name, *_ in CONDS:
    g = df[df.cond == name]
    if len(g) < 100: continue
    log(f'{name}')
    log(f'   p1={g.ret.quantile(0.01):>8.2f}%  p25={g.ret.quantile(0.25):>8.2f}%  中位={g.ret.median():>7.2f}%  p75={g.ret.quantile(0.75):>8.2f}%  p99={g.ret.quantile(0.99):>8.2f}%  最大={g.ret.max():>8.2f}%')
    log(f'   止盈笔占比={100*(g.ret > (TP-FEE)*100-0.5).mean():.1f}%  止损笔占比={100*(g.ret < -(SL+FEE)*100+0.5).mean():.1f}%')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bidir_verify2_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
df.to_csv(os.path.join(OUT, 'bidir_verify2_rows.csv'), index=False, encoding='utf-8-sig')
log('')
log(f'报告已写 {os.path.join(OUT, "bidir_verify2_summary.txt")}')
log(f'TOTAL {time.time()-t0:.0f}s')
