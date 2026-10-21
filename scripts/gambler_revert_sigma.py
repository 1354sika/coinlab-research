# -*- coding: utf-8 -*-
"""v4-1 修复版：把「跌多少」交给模型（波动标准化）+ 四段验证

上一版 bug：循环里 break 导致每个时刻只记进第一个命中的档位，1.5σ 之后全 0 笔。
本版：每个口径独立扫描，互不干扰。

口径：
  A 固定跌 5%（基线，和 revert_deep 对齐）
  B 跌 kσ，k ∈ {1, 1.5, 2, 2.5, 3, 4}，σ = 过去 240 根的 r5 标准差

严格无前视：σ 与 r5 只用 i 及之前的数据。

输出：evidence/revert_sigma.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('RS_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\revert_sigma.txt'
FEE = 0.0004
SLIP = float(os.environ.get('RS_SLIP', '0.0005'))
NSYM = int(os.environ.get('RS_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('RS_HOLD', '176'))
W = 240

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('把「跌多少」交给模型 · 波动标准化（跌几个 σ）[修复版]')
say(f'  {os.path.basename(DATA)}  持 {HOLD} 根  成本 {COST:.3%}  σ窗 {W}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

data = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    if len(C) < 3000:
        continue
    ts = g['open_time'].values
    T = len(C)
    r5 = np.full(T, np.nan)
    r5[5:] = C[5:] / C[:-5] - 1.0
    sig = pd.Series(r5).rolling(W).std().values
    data.append((sym, C, ts, r5, sig))
say(f'  可用币 {len(data)}  {time.time()-t0:.0f}s')

def scan(thr, use_sigma):
    """独立扫描：命中就买，持 HOLD 根后结算，然后跳过 HOLD 根"""
    out = []
    for sym, C, ts, r5, sig in data:
        T = len(C)
        i = W + 5
        while i < T - HOLD:
            if use_sigma:
                if not np.isfinite(sig[i]) or sig[i] <= 0:
                    i += 1
                    continue
                cond = r5[i] <= -thr * sig[i]
            else:
                cond = r5[i] <= -thr
            if cond:
                out.append((ts[i], C[i + HOLD] / C[i] - 1.0 - COST))
                i += HOLD
            else:
                i += 1
    return out

CASES = [('固定跌5%', 0.05, False),
         ('跌1σ', 1.0, True), ('跌1.5σ', 1.5, True), ('跌2σ', 2.0, True),
         ('跌2.5σ', 2.5, True), ('跌3σ', 3.0, True), ('跌4σ', 4.0, True)]

say('')
say('=' * 100)
say('【一】各口径：笔数 / 单笔均值 / t / 四段')
say('=' * 100)
say(f'  {"口径":<10s} {"笔数":>9s} {"单笔均值":>9s} {"胜率":>7s} {"t值":>7s}   四段')
best = None
for name, thr, use_sig in CASES:
    rows = scan(thr, use_sig)
    if len(rows) < 300:
        say(f'  {name:<10s} {len(rows):>9,}  样本太少')
        continue
    rows.sort(key=lambda x: x[0])
    r = np.array([x[1] for x in rows])
    mu = r.mean(); sd = r.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(r))) if sd > 0 else 0
    seg = pd.qcut(np.arange(len(r)), 4, labels=False)
    segs = [r[seg == s].mean() for s in range(4)]
    same = all(x > 0 for x in segs)
    segstr = '  '.join(f'{x*100:>+6.2f}%' for x in segs)
    say(f'  {name:<10s} {len(r):>9,} {mu*100:>+8.3f}% {(r>0).mean():>6.1%} {tv:>+7.2f}   [{segstr}] {"✅四段同正" if same else "❌"}')
    if same and (best is None or mu > best[1]):
        best = (name, mu, len(r), tv)

say('')
say('=' * 100)
say('【二】判据')
say('=' * 100)
if best:
    say(f'  ✅ 四段同正里最好的是 {best[0]}：单笔 {best[1]*100:+.3f}%、{best[2]:,} 笔、t={best[3]:+.2f}')
else:
    say('  ❌ 没有一个口径四段同正 ⇒ 抄底只在特定时期有效，还需要"何时不打"的第二层')
say('  · 若 kσ 口径笔数明显多于固定 5% 且单笔不降 ⇒ 波动标准化有效（模型自适应生效）')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
