# -*- coding: utf-8 -*-
"""分年诊断：哪段时间追涨有机会、哪段时间没有

回答一个问题：如果模型只学两件事（下注/不下注、走/继续持），
那它该在什么时期"不下注"？

做法：按时间切 8 段，每段统计
  ① 入场次数（追涨确认底）
  ② 固定持 176 根的单笔均值、正段占比
  ③ 该时期的市场特征（总涨幅、波动、成交量）

输出：evidence/diag_period.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('DP_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\diag_period.txt'
Z = float(os.environ.get('DP_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('DP_SLIP', '0.0005'))
NSYM = int(os.environ.get('DP_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('DP_HOLD', '176'))
NP = int(os.environ.get('DP_NP', '8'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('分年诊断 · 哪段时间追涨有机会')
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
say('  切段 ...')
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    for (a, e, b) in zigzag_entry(C, Z):
        j = min(e + HOLD, T - 1)
        if j <= e:
            continue
        r = C[j] / C[e] - 1.0 - COST
        rows.append((ts[e], r, C[e], C[j], np.mean(V[e:j+1])))

R = pd.DataFrame(rows, columns=['t', 'ret', 'p_in', 'p_out', 'vol'])
say(f'  入场 {len(R):,} 笔  {time.time()-t0:.0f}s')

R['period'] = pd.qcut(R.t, NP, labels=False)
tmin, tmax = R.t.min(), R.t.max()
say(f'  时间范围 {pd.to_datetime(tmin, unit="ms")} ~ {pd.to_datetime(tmax, unit="ms")}')

say('')
say('=' * 100)
say('【一】按时间等分 8 段：追涨 + 固定持有的表现')
say('=' * 100)
say(f'  {"段":>3s} {"起":>12s} {"止":>12s} {"入场":>8s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"正段占比":>9s} {"t值":>7s}')
for p in range(NP):
    s = R[R.period == p]
    if len(s) < 50:
        continue
    mu = s.ret.mean()
    sd = s.ret.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(s))) if sd > 0 else 0
    d1 = pd.to_datetime(s.t.min(), unit='ms').strftime('%Y-%m-%d')
    d2 = pd.to_datetime(s.t.max(), unit='ms').strftime('%Y-%m-%d')
    say(f'  {p+1:>3d} {d1:>12s} {d2:>12s} {len(s):>8,} {mu*100:>+8.3f}% {s.ret.median()*100:>+8.3f}% {(s.ret>0).mean():>6.1%} {(s.ret>0).mean():>8.1%} {tv:>+7.2f}')

say('')
say('=' * 100)
say('【二】每段的市场特征（判断当时是什么行情）')
say('=' * 100)
say(f'  {"段":>3s} {"样本均价":>10s} {"平均涨幅":>10s} {"涨的币占比":>10s} {"成交量比":>9s}')
base_vol = R.groupby('period').vol.mean()
for p in range(NP):
    s = R[R.period == p]
    if len(s) < 50:
        continue
    up = (s.p_out > s.p_in).mean()
    chg = (s.p_out / s.p_in - 1).mean()
    vb = s.vol.mean() / base_vol.mean() if base_vol.mean() > 0 else 1
    say(f'  {p+1:>3d} {s.p_in.mean():>10.4g} {chg*100:>+9.2f}% {up:>9.1%} {vb:>9.2f}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 若某几段单笔均值为正且 t>2 ⇒ 那些时期有 alpha，模型该在那些时期下注')
say('  · 若全为负 ⇒ 追涨在这份数据上从没有过 alpha，问题在入场判据本身')
say('  · 若只有最早几段为正 ⇒ 市场结构变了（早期有效，后来被套利平掉）')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
