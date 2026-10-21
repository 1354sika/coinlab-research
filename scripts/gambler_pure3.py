# -*- coding: utf-8 -*-
"""gambler_pure3.py — 止损档位敏感性：真局率高却赚不到，是不是止损太紧把赢家甩下车？
承 pure2（job-47368-36）：
    R3(volr24>=3 且 r12>=5%) +50%TP/-15%SL → 期望净 +0.578%，但**中位 -2.08%**、止损率 12.9%
    R5(volr24>=5 且 r12 5~15%)              → 期望净 +0.870%，但**中位 -2.38%**、止损率 15.3%
    对照 R0 候选池随机 +0.097%
⇒ 提纯把期望净拉了 9 倍，但绝对水平仍薄、中位为负。

本脚本固定止盈，扫**止损档位**（-15% / -25% / -35% / -50% / 不止损），看"真局达成前的回撤"
  是否把赢家甩下车。全程无前视（入场条件只用 t 时刻可见量）。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_pure3.txt')
WDAY = 288
FEE = 0.0004 + 0.0005

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_pure3 · 止损档位敏感性（R3 / R5 入场，无前视）')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','quote_volume':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

def simulate(lo, c, h, l, tp, sl, to=WDAY):
    """sl=None 表示不止损"""
    n = len(c)
    end = min(n - 1, lo + to)
    if end <= lo + 1: return None
    tp_p = c[lo] * (1 + tp)
    ht = np.flatnonzero(h[lo+1:end+1] >= tp_p)
    it = ht[0] if len(ht) else 10**9
    if sl is None:
        isl = 10**9
    else:
        sl_p = c[lo] * (1 + sl)
        hs = np.flatnonzero(l[lo+1:end+1] <= sl_p)
        isl = hs[0] if len(hs) else 10**9
    if it == 10**9 and isl == 10**9: return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if it <= isl: return tp - FEE, 'tp'
    return sl - FEE, 'sl'

B = {}
def add(k, r, kind):
    B.setdefault(k, []).append((r, kind))

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    ok = np.isfinite(r12) & np.isfinite(volr24)
    base = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    R3 = base & (volr24 >= 3) & (r12 >= 0.05)
    R5 = base & (volr24 >= 5) & (r12 >= 0.05) & (r12 < 0.15)
    for tag, m in (('R3', R3), ('R5', R5)):
        cand = np.flatnonzero(ok & m)
        for tp in (0.50, 1.00):
            for sl in (-0.15, -0.25, -0.35, -0.50, None):
                last = -10 ** 9
                for t in cand:
                    if t - last < 12: continue
                    r = simulate(t, c, h, l, tp, sl)
                    if r is None: continue
                    add(f'{tag} TP{int(100*tp)}% SL{"无" if sl is None else str(abs(int(100*sl)))+"%"}',
                        r[0], r[1]); last = t

log('')
log('=' * 100)
log('【止损敏感性】+50%/+100% 止盈 × 各止损档；扣 0.09%')
log('=' * 100)
log(f'  {"组合":<28} {"笔数":>7} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
keys = []
for tag in ('R3', 'R5'):
    for tp in (50, 100):
        for sl in ('15', '25', '35', '50', '无'):
            keys.append(f'{tag} TP{tp}% SL{sl}%' if sl != '无' else f'{tag} TP{tp}% SL无')
for k in keys:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<28} {0:>7}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<28} {len(rs):>7,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若"不止损"期望净 显著高于各止损档，且中位转正 ⇒ 真局达成前确实被止损甩下车')
log('  · 若期望净随止损放宽单调上升但风险剧增 ⇒ 只能靠极小仓位+不止损博尾部')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
