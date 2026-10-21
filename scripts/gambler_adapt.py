# -*- coding: utf-8 -*-
"""适应市场：牌池 + 自动停牌

用户定义：模型得适应市场 —— 市场变了，旧牌要停，不能一直打。

机制：
  牌池 4 个入场规则（追涨/反弹/突破/区间低位），全部只做多、固定持 176 根
  每笔交易到来时，只看**这个牌最近 500 笔**的均值
    · 均值 > 0 ⇒ 这个牌还在赚钱，下注（1/4 凯利，单笔上限 2%）
    · 均值 <= 0 ⇒ 停牌（不下注）
  严格无前视：只用已结算的历史

对比三条线：
  A 全牌全下（不老练的赌徒）
  B 只下注"近期为正"的牌（适应的赌徒）
  C 永远空仓（基线）

输出：evidence/adapt.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('AD_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\adapt.txt'
FEE = 0.0004
SLIP = float(os.environ.get('AD_SLIP', '0.0005'))
NSYM = int(os.environ.get('AD_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('AD_HOLD', '176'))
LOOK = int(os.environ.get('AD_LOOK', '500'))
KELLY = float(os.environ.get('AD_KELLY', '0.25'))
MAXLOSS = float(os.environ.get('AD_MAXLOSS', '0.02'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('适应市场 · 牌池 + 自动停牌')
say(f'  {os.path.basename(DATA)}  持 {HOLD} 根  窗 {LOOK} 笔  成本 {COST:.3%}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def sig_chase(C, i):
    if i < 20: return False
    return C[i] > C[i-5] and C[i] > C[i-20]

def sig_revert(C, i):
    if i < 20: return False
    return C[i] / C[i-5] - 1.0 < -0.03

def sig_break(C, i):
    if i < 240: return False
    return C[i] > C[i-240:i].max()

def sig_range(C, i):
    if i < 240: return False
    lo = C[i-240:i].min(); hi = C[i-240:i].max()
    if hi <= lo: return False
    return (C[i] - lo) / (hi - lo) < 0.2

POOL = {'chase': sig_chase, 'revert': sig_revert, 'break': sig_break, 'range': sig_range}

say('  跑各牌的交易流 ...')
allrows = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    for name, fn in POOL.items():
        i = 0
        while i < T - HOLD:
            if fn(C, i):
                r = C[i + HOLD] / C[i] - 1.0 - COST
                allrows.append((ts[i], name, r))
                i += HOLD
            else:
                i += 1

R = pd.DataFrame(allrows, columns=['t', 'card', 'ret']).sort_values('t').reset_index(drop=True)
say(f'  交易 {len(R):,} 笔  牌 {R.card.nunique()} 个  {time.time()-t0:.0f}s')

# ---- 各牌整体表现 ----
say('')
say('=' * 100)
say('【一】各牌整体（不分时期）')
say('=' * 100)
say(f'  {"牌":<10s} {"笔数":>8s} {"单笔均值":>9s} {"胜率":>7s} {"t值":>7s}')
for name, s in R.groupby('card'):
    mu = s.ret.mean(); sd = s.ret.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(s))) if sd > 0 else 0
    say(f'  {name:<10s} {len(s):>8,} {mu*100:>+8.3f}% {(s.ret>0).mean():>6.1%} {tv:>+7.2f}')

# ---- 逐笔自适应 ----
say('')
say('  逐笔自适应回放 ...')
hist = {k: [] for k in POOL}
capA = 1.0; capB = 1.0
peakA = peakB = 1.0
ddA = ddB = 0.0
on_count = 0
rowsB = []

for _, row in R.iterrows():
    card = row.card
    r = row.ret
    h = hist[card]
    # A：全下（固定 2% 仓）
    fA = 0.02
    capA *= (1.0 + fA * r)
    peakA = max(peakA, capA)
    ddA = min(ddA, capA / peakA - 1.0)
    # B：只有近期为正的牌才下
    if len(h) >= LOOK:
        arr = np.array(h[-LOOK:])
        mu = arr.mean()
        los = -arr[arr <= 0].mean() if (arr <= 0).any() else 1e-9
        win = arr[arr > 0].mean() if (arr > 0).any() else 0.0
        b = win / max(los, 1e-9)
        p = (arr > 0).mean()
        edge = p * b - (1 - p)
        if mu > 0 and edge > 0:
            f = min(KELLY * edge / max(b, 1e-9), MAXLOSS / max(los, 1e-9), 0.10)
            f = max(0.0, f)
        else:
            f = 0.0
    else:
        f = 0.0
    capB *= (1.0 + f * r)
    peakB = max(peakB, capB)
    ddB = min(ddB, capB / peakB - 1.0)
    if f > 0:
        on_count += 1
    rowsB.append((row.t, f, r))
    h.append(r)

say('')
say('=' * 100)
say('【二】三条件对比')
say('=' * 100)
yrs = (R.t.max() - R.t.min()) / (365.25 * 24 * 3600 * 1000)
say(f'  样本期 {yrs:.2f} 年')
say(f'  {"策略":<22s} {"期末资金":>14s} {"最大回撤":>10s} {"下注率":>8s}')
say(f'  {"A 全牌全下(2%仓)":<22s} {capA:>14.2f} {ddA*100:>9.2f}% {100.0:>7.1f}%')
say(f'  {"B 只下注近期为正":<22s} {capB:>14.4f} {ddB*100:>9.2f}% {on_count/len(R)*100:>7.1f}%')
say(f'  {"C 永远空仓":<22s} {1.0:>14.4f} {0.0:>9.2f}% {0.0:>7.1f}%')

# ---- 按季度看 B ----
B = pd.DataFrame(rowsB, columns=['t', 'f', 'ret'])
B['dt'] = pd.to_datetime(B.t, unit='ms')
B['yq'] = B.dt.dt.year.astype(str) + 'Q' + B.dt.dt.quarter.astype(str)
B['pnl'] = B.f * B.ret
gq = B.groupby('yq').agg(n=('ret','size'), on=('f', lambda x: (x>0).mean()),
                         mean_ret=('ret','mean'), pnl=('pnl','sum'))
say('')
say('=' * 100)
say('【三】B 的按季表现（是否在萎缩期自动停牌）')
say('=' * 100)
say(f'  {"季度":<8s} {"笔数":>8s} {"该季下注率":>10s} {"该季单笔均值":>11s} {"该季累计盈亏":>12s}')
for k, v in gq.iterrows():
    say(f'  {k:<8s} {int(v.n):>8,} {v.on*100:>9.1f}% {v.mean_ret*100:>+10.3f}% {v.pnl*100:>+11.2f}%')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 若 B 的期末资金 > 1 且最大回撤 < A 的回撤 ⇒ 适应机制有效（少亏=适应）')
say('  · 若 2025 之后下注率明显下降 ⇒ 模型确实在"萎缩期自动收手"')
say('  · 若 B 仍亏损 ⇒ 牌池里没有能用的牌，机制对但原料错')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
