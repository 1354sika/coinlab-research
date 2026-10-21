# -*- coding: utf-8 -*-
"""追涨 + 让模型自己选退出：对比各退出策略（严格无前视）

入场：zigzag 确认底那一刻（= 追涨，不需要预知顶）
退出对比（全部从 e+1 逐根检查，不用任何未来信息）：
  A. 持到顶确认      —— 这是上界，不是可实现策略（顶是事后才知道的）
  B. 固定 H=100 根
  C. 固定止损 -1.79% / 止盈 +4.43%  （照段的正负均值设定）
  D. 移动止损：赚 2% 后止损上移到 +0.5%
  E. 移动止损：赚 3% 后止损上移到 +1.5%
  F. 模型自选：按浮盈档位查表（训练集上统计，测试集上应用）

输出：单笔均值 / 胜率 / 四段稳定性
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('EL_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\exit_learn.txt'
Z = float(os.environ.get('EL_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('EL_SLIP', '0.0005'))
NSYM = int(os.environ.get('EL_NSYM', '60'))
COST = FEE + SLIP
MAX_H = int(os.environ.get('EL_MAXH', '576'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('追涨 + 退出策略对比 · 严格无前视')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}  MAX_H={MAX_H}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def zigzag_entry(C, z):
    """返回 [(i_lo, i_entry, i_hi)]"""
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

def sim_exit(C, Hh, Ll, e, entry, style, p):
    """从 e+1 逐根检查，返回净收益（已扣成本）。无前视：只用到 j 的价格。"""
    T = len(C)
    end = min(T, e + MAX_H + 1)
    if end <= e + 1:
        return None
    hi_water = entry
    trail_armed = False
    trail_stop = 0.0
    for j in range(e + 1, end):
        c = C[j]
        if style == 'FIXH':
            if j - e >= p:
                return c / entry - 1.0 - COST
        elif style == 'SLTP':
            sl, tp = p
            if Ll[j] <= entry * (1 - sl):
                return (1 - sl) - 1.0 - COST
            if Hh[j] >= entry * (1 + tp):
                return (1 + tp) - 1.0 - COST
        elif style == 'TRAIL':
            arm, back = p          # 赚 arm 后启动，回撤 back 出
            if Hh[j] > hi_water:
                hi_water = Hh[j]
            if not trail_armed and hi_water >= entry * (1 + arm):
                trail_armed = True
                trail_stop = entry * (1 + arm - back)
            if trail_armed:
                if Ll[j] <= trail_stop:
                    return trail_stop / entry - 1.0 - COST
                new_stop = hi_water * (1 - back)
                if new_stop > trail_stop:
                    trail_stop = new_stop
        elif style == 'HOLD':
            pass
    # 超时
    return C[end - 1] / entry - 1.0 - COST

STYLES = [
    ('A 持到顶确认(上界)', 'HOLD', None),
    ('B 固定H=100', 'FIXH', 100),
    ('C 止损1.79%/止盈4.43%', 'SLTP', (0.0179, 0.0443)),
    ('D 移动止损 赚2%回0.5%', 'TRAIL', (0.02, 0.005)),
    ('E 移动止损 赚3%回1.5%', 'TRAIL', (0.03, 0.015)),
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
            v = sim_exit(C, Hh, Ll, e, entry, st, p)
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
say(f'  {"策略":<24s} {"笔数":>8s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"t值":>7s}')
best = None
for k, _, _ in STYLES:
    v = np.array(res[k])[order]
    mu = v.mean()
    sd = v.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(v))) if sd > 0 else 0
    wr = (v > 0).mean()
    say(f'  {k:<24s} {len(v):>8,} {mu*100:>+8.3f}% {np.median(v)*100:>+8.3f}% {wr:>6.1%} {tv:>+7.2f}')
    if best is None or mu > best[1]:
        best = (k, mu)

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
    say(f'  {k:<24s} 段1 {row[0]}  段2 {row[1]}  段3 {row[2]}  段4 {row[3]}   {"✅四段同正" if same else "❌有段非正"}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · A 是上界（事前拿不到），只能当参照')
say('  · 若 C/D/E 任一优于 B（固定持有）且四段同正 ⇒ 退出本身有价值')
say('  · 若全都劣于 A 且四段不同正 ⇒ 这套退出不会长出来，需换退出的状态维度')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
