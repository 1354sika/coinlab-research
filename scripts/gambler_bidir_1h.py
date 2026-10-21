# -*- coding: utf-8 -*-
"""gambler_bidir_1h · 不搞 5 分钟：聚合成 1h 重跑，并量化 5m 样本虚高倍数

回答两件事：
  1) 5m 粒度上，一个「事件」被重复计了多少次（命中点数 ÷ 连续段数）
  2) 换 1h 后，A/B/C/D 四条件的期望净收益是否还成立
     粗粒度副作用：单根内可能同时触及 TP 与 SL → 两种口径各报一遍
     （保守=算止损 / 中性=算 0），避免把「粒度代价」误读成「信号消失」
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
TP, SL, FEE = 0.30, 0.15, 0.0009
HOLD = 48

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_bidir_1h · 5m→1h 聚合重跑 + 5m 样本虚高量化')
log('=' * 112)

CONDS = [
    ('A 24h跌>=20%',  86400, -np.inf, -0.20),
    ('B 24h涨10~20%', 86400,  0.10,   0.20),
    ('C 4h涨>=20%',   14400,  0.20,   np.inf),
    ('D 24h跌10~20%', 86400, -0.20,  -0.10),
]

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64',
                             'close': 'float32', 'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 20000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   5m根 {sum(len(x[1]) for x in RAW):,}   读盘 {time.time()-t0:.0f}s')

# ---------- 一、5m 样本虚高 ----------
log('\n' + '=' * 112)
log('【一】5m 上：命中点数 vs 独立事件数（连续命中算一个事件）')
log('=' * 112)
for (nm, sec, lo, hi) in CONDS:
    lag5 = sec // 300
    pts = seg = 0
    for (sym, ot, c, h, l) in RAW:
        n = len(c)
        if n <= lag5 + 10:
            continue
        r = np.full(n, np.nan)
        r[lag5:] = c[lag5:] / c[:-lag5] - 1.0
        m = (r >= lo) & (r < hi)
        m[:lag5] = False
        pts += int(m.sum())
        seg += int((np.diff(m.astype('int8'), prepend=0) == 1).sum())
    log(f'{nm:16s} 点数 {pts:>9,}   独立事件 {seg:>7,}   虚高 {pts/max(seg,1):>5.1f}x')

# ---------- 二、聚合 1h ----------
log('\n' + '=' * 112)
log('【二】5m → 1h 聚合')
log('=' * 112)
H1 = []
for (sym, ot, c, h, l) in RAW:
    key = ot // 3600000
    uniq, st = np.unique(key, return_index=True)
    en = np.append(st[1:], len(key)) - 1
    H1.append((sym, uniq * 3600000, c[en],
               np.maximum.reduceat(h, st), np.minimum.reduceat(l, st)))
log(f'1h根 {sum(len(x[1]) for x in H1):,}   {time.time()-t0:.0f}s')
del RAW; gc.collect()

# ---------- 三、1h 交易扫描 ----------
def scan(i, c, h, l, long_side):
    n = len(c); end = min(i + HOLD, n)
    if end - i < 5:
        return None
    e = c[i]
    for j in range(i + 1, end):
        if long_side:
            hs = l[j] <= e * (1 - SL); ht = h[j] >= e * (1 + TP)
        else:
            hs = h[j] >= e * (1 + SL); ht = l[j] <= e * (1 - TP)
        if hs and ht:
            return (2, -SL - FEE)
        if hs:
            return (0, -SL - FEE)
        if ht:
            return (1, TP - FEE)
    r = c[end - 1] / e - 1.0
    if not long_side:
        r = -r
    return (3, r - FEE)

def stat(codes, rets):
    codes = np.asarray(codes); rets = np.asarray(rets)
    if len(rets) == 0:
        return (0, np.nan, np.nan, np.nan, np.nan, np.nan)
    neu = np.where(codes == 2, 0.0, rets)
    return (len(rets), rets.mean() * 100, neu.mean() * 100,
            np.median(neu) * 100, (neu > 0).mean() * 100, (codes == 2).mean() * 100)

allot = np.concatenate([x[1] for x in H1])
TUNIT = 's' if allot[0] < 1e11 else 'ms'
cut = int(np.quantile(allot, 0.70))
log(f'全样本 {len(allot):,}   open_time 单位={TUNIT}   测试段起点 {pd.to_datetime(cut, unit=TUNIT)}')

log('\n' + '=' * 112)
log('【三】1h 重跑（TP+30% / SL-15% / 48h；保守=双触算止损，中性=双触算0）')
log('=' * 112)
hdr = f'{"条件":18s}{"方向":6s}{"口径":8s}{"笔":>8s}{"保守%":>9s}{"中性%":>9s}{"中位%":>9s}{"胜率%":>8s}{"双触%":>7s}'
for (nm, sec, lo, hi) in CONDS:
    lag = sec // 3600
    rec = []      # (t, k, long, code, ret)
    for k, (sym, ot, c, h, l) in enumerate(H1):
        n = len(c)
        if n <= lag + 5:
            continue
        r = np.full(n, np.nan)
        r[lag:] = c[lag:] / c[:-lag] - 1.0
        m = (r >= lo) & (r < hi)
        m[:lag] = False
        idxs = np.flatnonzero(m)
        if len(idxs) == 0:
            continue
        for i in idxs:
            for long_side in (True, False):
                res = scan(i, c, h, l, long_side)
                if res:
                    rec.append((int(ot[i]), k, long_side, res[0], res[1]))
    log('\n' + '-' * 112)
    log(hdr); log('-' * 112)
    for long_side in (True, False):
        side = '做多' if long_side else '做空'
        sub = [x for x in rec if x[2] == long_side]
        for tag, name, filt in (
            ('T', '测试段', lambda x: x[0] >= cut),
            ('A', '币组A', lambda x: x[0] >= cut and x[1] % 2 == 0),
            ('B', '币组B', lambda x: x[0] >= cut and x[1] % 2 == 1),
        ):
            rows = [x for x in sub if filt(x)]
            n, cons, neu, med, win, dbl = stat([x[3] for x in rows], [x[4] for x in rows])
            log(f'{nm:18s}{side:6s}{name:8s}{n:>8,}{cons:>9.2f}{neu:>9.2f}{med:>9.2f}{win:>8.1f}{dbl:>7.1f}')
    # 逐年（测试段）
    log('  逐年（测试段，中性口径）：')
    for long_side in (True, False):
        side = '做多' if long_side else '做空'
        rows = [x for x in rec if x[2] == long_side and x[0] >= cut]
        if not rows:
            continue
        yrs = pd.to_datetime([x[0] for x in rows], unit=TUNIT).year
        s = []
        for y in sorted(set(yrs)):
            m = yrs == y
            r = np.array([x[4] for x in rows])[m]
            cd = np.array([x[3] for x in rows])[m]
            neu = np.where(cd == 2, 0.0, r)
            s.append(f'{y}: n={m.sum():>5,} {neu.mean()*100:+6.2f}%')
        log(f'    {side}  ' + '   '.join(s))

log('\n' + '=' * 112)
log('【判读】')
log('=' * 112)
log('  · 测试段/币组A/B 三列同号为正 ⇒ 信号不是单币或单段贡献')
log('  · 双触% 高 ⇒ 该粒度下 OHLC 无法决定先触谁，收益区间只能给上下界')
log('  · 与 5m 版对照：笔数应大幅下降（5m 的虚高倍数见【一】）')

out = os.path.join(OUT, 'bidir_1h_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
