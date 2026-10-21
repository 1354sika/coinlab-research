# -*- coding: utf-8 -*-
"""交叉验证：volr 择时到底是"状态信号"还是"时期标签"

疑问：volr 高档单笔 +0.706%（t=+6.41）很亮，但四段不同正。
      是不是它只是在识别"2023-2024 年"这段活跃期？

做法：volr 分 5 档 × 时间分 8 段，看每个格子的单笔均值。
      如果高 volr 档只在前两段为正、后六段为负 ⇒ 时期标签，不是状态信号。

输出：evidence/regime_xperiod.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('RX_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\regime_xperiod.txt'
Z = float(os.environ.get('RX_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('RX_SLIP', '0.0005'))
NSYM = int(os.environ.get('RX_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('RX_HOLD', '176'))
NP = int(os.environ.get('RX_NP', '8'))
NQ = int(os.environ.get('RX_NQ', '5'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('交叉验证 · volr 择时是状态信号还是时期标签')
say(f'  {os.path.basename(DATA)}  Z={Z:.1%}  持 {HOLD} 根  成本 {COST:.3%}  {NQ}档 × {NP}段')
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

rows = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    sv = pd.Series(V)
    volr = (sv.rolling(60).mean() / np.maximum(sv.rolling(1440).mean(), 1e-12)).values
    for (a, e, b) in zigzag_entry(C, Z):
        j = min(e + HOLD, T - 1)
        if j <= e or not np.isfinite(volr[e]):
            continue
        rows.append((ts[e], C[j] / C[e] - 1.0 - COST, volr[e]))

R = pd.DataFrame(rows, columns=['t', 'ret', 'volr'])
say(f'  入场 {len(R):,} 笔  {time.time()-t0:.0f}s')

R['p'] = pd.qcut(R.t, NP, labels=False)
R['vq'] = pd.qcut(R.volr, NQ, labels=False)

pv = R.pivot_table(index='vq', columns='p', values='ret', aggfunc='mean')
pn = R.pivot_table(index='vq', columns='p', values='ret', aggfunc='size')

say('')
say('=' * 100)
say('【一】单笔均值矩阵（行=volr档，列=时期）')
say('=' * 100)
head = '  volr档  ' + ''.join(f'   期{int(c)+1:>1d}    ' for c in pv.columns)
say(head)
for r in pv.index:
    line = f'   {int(r)+1:>2d}  '
    for c in pv.columns:
        v = pv.loc[r, c]
        line += f' {v*100:>+7.2f}% ' if np.isfinite(v) else '    n/a  '
    say(line)

say('')
say('=' * 100)
say('【二】样本数矩阵')
say('=' * 100)
say(head)
for r in pn.index:
    line = f'   {int(r)+1:>2d}  '
    for c in pn.columns:
        v = pn.loc[r, c]
        line += f' {int(v) if np.isfinite(v) else 0:>7d}  '
    say(line)

say('')
say('=' * 100)
say('【三】最高 volr 档在各时期的均值（关键行）')
say('=' * 100)
top = pv.index.max()
for c in pv.columns:
    v = pv.loc[top, c]
    n = pn.loc[top, c]
    if np.isfinite(v):
        say(f'  期{int(c)+1}: n={int(n):>6,}  单笔 {v*100:>+7.3f}%   {"✅" if v > 0 else "❌"}')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 若最高 volr 档在多数时期为正 ⇒ 它是状态信号，模型可以学')
say('  · 若只在最早 1~2 期为正、之后全负 ⇒ 它其实是"时期标签"（早期活跃期），不是可迁移的状态信号')
say('  · 后者意味着：模型在这份数据上找不到"能反复用"的下注条件')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
