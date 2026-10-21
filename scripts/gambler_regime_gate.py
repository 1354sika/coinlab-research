# -*- coding: utf-8 -*-
"""市场活跃度择时：volr 高的时候下注、低的时候不注

诊断发现：成交量从 4.59 缩到 0.11，早期有 alpha、后期没有。
本脚本验证：入场时刻的 volr（近 60 根均量 / 近 1440 根均量）
能不能把"该下注"的时刻与"不该下注"的时刻分开。

做法：
  对每个入场，算入场时刻的 volr，按 volr 分 5 档
  每档统计：单笔均值（固定持 176）、t 值、四段稳定性

严格无前视：volr 只用 e 及之前的成交量。

输出：evidence/regime_gate.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('RG_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\regime_gate.txt'
Z = float(os.environ.get('RG_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('RG_SLIP', '0.0005'))
NSYM = int(os.environ.get('RG_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('RG_HOLD', '176'))
NQ = int(os.environ.get('RG_NQ', '5'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('市场活跃度择时 · volr 能否分开"该下注"与"不该下注"')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  持有 {HOLD} 根  成本 {COST:.3%}  币 {NSYM}')
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
say('  切段 + 算 volr ...')
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    sv = pd.Series(V)
    v60 = sv.rolling(60).mean().values
    v1440 = sv.rolling(1440).mean().values
    volr = v60 / np.maximum(v1440, 1e-12)
    # 绝对成交量水平：近 1440 根均量 / 全序列均量（判断市场整体冷热）
    vabs = v1440 / np.maximum(np.nanmean(V), 1e-12)
    for (a, e, b) in zigzag_entry(C, Z):
        j = min(e + HOLD, T - 1)
        if j <= e or not np.isfinite(volr[e]) or not np.isfinite(vabs[e]):
            continue
        r = C[j] / C[e] - 1.0 - COST
        rows.append((ts[e], r, volr[e], vabs[e]))

R = pd.DataFrame(rows, columns=['t', 'ret', 'volr', 'vabs'])
say(f'  入场 {len(R):,} 笔  {time.time()-t0:.0f}s')

def bucket_report(col, name, nb):
    say('')
    say('=' * 100)
    say(f'【按 {name} 分 {nb} 档】')
    say('=' * 100)
    q = pd.qcut(R[col], nb, labels=False, duplicates='drop')
    say(f'  {"档":>3s} {"区间":>22s} {"笔数":>8s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"t值":>7s} {"四段":>14s}')
    for b in sorted(q.dropna().unique()):
        s = R[q == b]
        if len(s) < 100:
            continue
        mu = s.ret.mean()
        sd = s.ret.std(ddof=1)
        tv = mu / (sd / np.sqrt(len(s))) if sd > 0 else 0
        lo_, hi_ = s[col].min(), s[col].max()
        ss = s.sort_values('t')
        seg = pd.qcut(np.arange(len(ss)), 4, labels=False)
        segs = [ss.ret[seg == k].mean() for k in range(4)]
        same = all(x > 0 for x in segs)
        tag = '✅四段同正' if same else '❌'
        say(f'  {int(b)+1:>3d} [{lo_:>9.3f},{hi_:>9.3f}] {len(s):>8,} {mu*100:>+8.3f}% {s.ret.median()*100:>+8.3f}% {(s.ret>0).mean():>6.1%} {tv:>+7.2f} {tag:>14s}')

bucket_report('volr', 'volr（近60/近1440）', NQ)
bucket_report('vabs', 'vabs（近1440/全序列均量）', NQ)

say('')
say('=' * 100)
say('【组合：vabs 最活跃档 × volr 各档】')
say('=' * 100)
qv = pd.qcut(R.vabs, NQ, labels=False, duplicates='drop')
qr = pd.qcut(R.volr, NQ, labels=False, duplicates='drop')
top_v = qv.max()
sub = R[(qv == top_v) & qr.notna()]
say(f'  vabs 最高档（市场最活跃）共 {len(sub):,} 笔')
for b in sorted(qr[qv == top_v].dropna().unique()):
    s = R[(qv == top_v) & (qr == b)]
    if len(s) < 100:
        continue
    mu = s.ret.mean()
    sd = s.ret.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(s))) if sd > 0 else 0
    say(f'    volr 档{int(b)+1}: n={len(s):>7,}  单笔 {mu*100:>+7.3f}%  t={tv:>+6.2f}')

say('')
say('=' * 100)
say('【判据】')
say('=' * 100)
say('  · 若 vabs 高档单笔均值为正、低档为负 ⇒ 市场活跃度就是"该不该下注"的实时信号')
say('  · 若 volr 高档显著高于低档 ⇒ 短期放量也是信号（可叠加）')
say('  · 若全档都接近 0 ⇒ volr 不是有效择时，模型要从别处学"何时不下注"')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
