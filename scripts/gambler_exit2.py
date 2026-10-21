# -*- coding: utf-8 -*-
"""退出策略第二轮：截断亏损、让利润奔跑（严格无前视）

入场：zigzag 确认底那一刻（= 追涨）
本轮对比（全部从 e+1 逐根检查，不用任何未来信息）：
  G 止损2% 持576（无固定止盈）
  H 止损2% + 移动止损（赚3%回1%）
  I 止损1.5% + 移动止损（赚2%回0.8%）
  J 止损3% 固定持176
  K 无止损 固定持176（基线）
  L 止损2% + 移动止损（赚5%回2%）

输出：evidence/exit_learn2.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('EL2_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\exit_learn2.txt'
Z = float(os.environ.get('EL2_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('EL2_SLIP', '0.0005'))
NSYM = int(os.environ.get('EL2_NSYM', '60'))
COST = FEE + SLIP
MAX_H = int(os.environ.get('EL2_MAXH', '576'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('退出策略第二轮 · 截断亏损、让利润奔跑（严格无前视）')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}  MAX_H={MAX_H}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def zigzag_entry(C, z):
    T = len(C)
    segs = []
    trend = 0
    hi_i = lo_i = 0
    hi_v = lo_v = C[0]
    entry_i = None
    for i in range(1, T):
        c = C[i]
        if trend >= 0:
            if c > hi_v:
                hi_v, hi_i = c, i
            if c <= hi_v * (1 - z):
                if trend == 1 and entry_i is not None:
                    segs.append((lo_i, entry_i, hi_i))
                trend = -1
                lo_v, lo_i = c, i
                entry_i = None
        if trend <= 0:
            if c < lo_v:
                lo_v, lo_i = c, i
            if c >= lo_v * (1 + z):
                trend = 1
                hi_v, hi_i = c, i
                entry_i = i
    return segs

def sim(C, Hh, Ll, e, entry, style, p):
    """从 e+1 逐根检查，返回净收益（已扣成本）。只用 j 及之前的价格。"""
    T = len(C)
    end = min(T, e + MAX_H + 1)
    if end <= e + 1:
        return None
    hi = entry
    if style == 'SL':
        sl = p
        for j in range(e + 1, end):
            if Ll[j] <= entry * (1 - sl):
                return -sl - COST
        return C[end - 1] / entry - 1.0 - COST
    if style == 'SL_H':
        sl, h = p
        jend = min(end, e + int(h) + 1)
        for j in range(e + 1, jend):
            if Ll[j] <= entry * (1 - sl):
                return -sl - COST
        return C[jend - 1] / entry - 1.0 - COST
    if style == 'SL_TRAIL':
        sl, arm, back = p
        armed = False
        stop = 0.0
        for j in range(e + 1, end):
            if Ll[j] <= entry * (1 - sl):
                return -sl - COST
            if Hh[j] > hi:
                hi = Hh[j]
            if not armed and hi >= entry * (1 + arm):
                armed = True
                stop = entry * (1 + arm - back)
            if armed:
                if Ll[j] <= stop:
                    return stop / entry - 1.0 - COST
                ns = hi * (1 - back)
                if ns > stop:
                    stop = ns
        return C[end - 1] / entry - 1.0 - COST
    if style == 'H':
        j = min(T - 1, e + int(p))
        return C[j] / entry - 1.0 - COST
    return None

STYLES = [
    ('G 止损2% 持576', 'SL', 0.02),
    ('H 止损2%+移动3%→1%', 'SL_TRAIL', (0.02, 0.03, 0.01)),
    ('I 止损1.5%+移动2%→0.8%', 'SL_TRAIL', (0.015, 0.02, 0.008)),
    ('J 止损3% 持176', 'SL_H', (0.03, 176)),
    ('K 无止损 持176(基线)', 'H', 176),
    ('L 止损2%+移动5%→2%', 'SL_TRAIL', (0.02, 0.05, 0.02)),
]

res = {k: [] for k, _, _ in STYLES}
res['t'] = []
say('  模拟退出 ...')

for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    Hh = g['high'].values.astype(float)
    Ll = g['low'].values.astype(float)
    ts = g['open_time'].values
    if len(C) < 3000:
        continue
    for (a, e, b) in zigzag_entry(C, Z):
        entry = C[e]
        if entry <= 0:
            continue
        r = {}
        ok = True
        for k, st, p in STYLES:
            v = sim(C, Hh, Ll, e, entry, st, p)
            if v is None:
                ok = False
                break
            r[k] = v
        if not ok:
            continue
        for k in res:
            if k != 't':
                res[k].append(r[k])
        res['t'].append(ts[e])

n = len(res['t'])
say(f'  可用样本 {n:,}  {time.time()-t0:.0f}s')
if n < 5000:
    say('  ❌ 样本不足')
    sys.exit(1)

tt = np.array(res['t'])
order = np.argsort(tt)

say('')
say('=' * 100)
say('【一】各退出策略（全部样本）')
say('=' * 100)
say(f'  {"策略":<26s} {"笔数":>8s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"t值":>7s}')
for k, _, _ in STYLES:
    v = np.array(res[k])[order]
    mu = v.mean()
    sd = v.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(v))) if sd > 0 else 0
    wr = (v > 0).mean()
    say(f'  {k:<26s} {len(v):>8,} {mu*100:>+8.3f}% {np.median(v)*100:>+8.3f}% {wr:>6.1%} {tv:>+7.2f}')

say('')
say('=' * 100)
say('【二】四段稳定性（每段单独看单笔均值）')
say('=' * 100)
seg_arr = pd.qcut(np.arange(n), 4, labels=False)
for k, _, _ in STYLES:
    v = np.array(res[k])[order]
    row = []
    for sg in range(4):
        m = (seg_arr == sg)
        row.append(f'{v[m].mean()*100:>+7.3f}%')
    same = all((v[seg_arr == sg].mean() > 0) for sg in range(4))
    say(f'  {k:<26s} 段1 {row[0]}  段2 {row[1]}  段3 {row[2]}  段4 {row[3]}   {"✅四段同正" if same else "❌有段非正"}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 与上一轮对比：上一轮最好的是 B 固定H=100 +0.144%（四段不同正）')
say('  · 若本轮任一"止损+移动止损"同时做到：均值 > B 且四段同正 ⇒ 这个退出结构可用')
say('  · 若全都劣于 B ⇒ 说明"让利润奔跑"在这个市场不成立（回撤太快，趋势走不远）')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
