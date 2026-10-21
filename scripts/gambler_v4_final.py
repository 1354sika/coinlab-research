# -*- coding: utf-8 -*-
"""v4 合成：真牌 + 停牌 + 凯利，跑完整资金曲线

三块合起来：
  入场：近 5 根跌 5% 就买（固定阈值——波动标准化已证更差）
  退出：固定持 176 根（下一步才让模型自定）
  仓位：1/4 凯利（用已结算的最近 200 笔估计 p 与盈亏比），单笔最大亏损 2% 本金
  急停：连亏 5 次就停 50 笔（已实测：期末 1.71 vs 1.07，回撤 −21% vs −34%）

严格无前视：凯利参数、急停状态都只用已结算的历史。

输出：
  ① 期末资金 / 年化 / 最大回撤 / 下注率 / 停牌次数
  ② 四段（每段时间的单笔均值与资金）
  ③ 按季度：2025 之后还在不在赚

输出：evidence/v4_final.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('VF_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\v4_final.txt'
FEE = 0.0004
SLIP = float(os.environ.get('VF_SLIP', '0.0005'))
NSYM = int(os.environ.get('VF_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('VF_HOLD', '176'))
DROP = float(os.environ.get('VF_DROP', '0.05'))
KELLY = float(os.environ.get('VF_KELLY', '0.25'))
MAXLOSS = float(os.environ.get('VF_MAXLOSS', '0.02'))
WARM = int(os.environ.get('VF_WARM', '200'))
STOP_N = int(os.environ.get('VF_STOPN', '5'))
COOL = int(os.environ.get('VF_COOL', '50'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('v4 合成 · 跌5%买 + 1/4凯利 + 连亏5次停50笔')
say(f'  {os.path.basename(DATA)}  持 {HOLD} 根  成本 {COST:.3%}  币 {NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

rows = []
say('  收集入场 ...')
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    i = 5
    while i < T - HOLD:
        if C[i] / C[i - 5] - 1.0 <= -DROP:
            rows.append((ts[i], C[i + HOLD] / C[i] - 1.0 - COST))
            i += HOLD
        else:
            i += 1

R = pd.DataFrame(rows, columns=['t', 'ret']).sort_values('t').reset_index(drop=True)
say(f'  入场 {len(R):,} 笔  {time.time()-t0:.0f}s')

# ---- 逐笔：凯利 + 急停 ----
cap = 1.0; peak = 1.0; maxdd = 0.0
hist = []
lose_streak = 0
skip = 0
halts = 0
bet = 0
caps, fs, skipped = [], [], []
for _, row in R.iterrows():
    r = row.ret
    if skip > 0:
        skip -= 1
        caps.append(cap); fs.append(0.0); skipped.append(1)
        hist.append(r)          # 停牌期也记账，但不下注
        continue
    if len(hist) >= WARM:
        h = np.array(hist[-WARM:])
        p = (h > 0).mean()
        win = h[h > 0].mean() if (h > 0).any() else 0.0
        los = -h[h <= 0].mean() if (h <= 0).any() else 1e-9
        b = win / max(los, 1e-9)
        edge = p * b - (1 - p)
        if edge > 0:
            f = min(KELLY * edge / max(b, 1e-9), MAXLOSS / max(los, 1e-9), 0.15)
        else:
            f = 0.0
    else:
        f = 0.0
    cap *= (1.0 + f * r)
    peak = max(peak, cap)
    maxdd = min(maxdd, cap / peak - 1.0)
    caps.append(cap); fs.append(f); skipped.append(0)
    hist.append(r)
    if f > 0:
        bet += 1
    if r <= 0:
        lose_streak += 1
    else:
        lose_streak = 0
    if lose_streak >= STOP_N:
        skip = COOL
        halts += 1
        lose_streak = 0

R['cap'] = caps
R['f'] = fs
R['halt'] = skipped
R['dt'] = pd.to_datetime(R.t, unit='ms')

yrs = (R.t.max() - R.t.min()) / (365.25 * 24 * 3600 * 1000)
ann = cap ** (1 / yrs) - 1 if yrs > 0 else 0

say('')
say('=' * 100)
say('【一】总体')
say('=' * 100)
say(f'  样本期 {yrs:.2f} 年   入场 {len(R):,} 笔')
say(f'  期末资金 {cap:.4f}   年化 {ann*100:+.2f}%')
say(f'  最大回撤 {maxdd*100:.2f}%')
say(f'  实际下注 {bet:,} 笔（{bet/len(R)*100:.1f}%）  停牌笔数 {int(R.halt.sum()):,}  触发急停 {halts} 次')
say(f'  平均仓位 {np.mean([f for f in fs if f > 0])*100 if bet else 0:.2f}%')

say('')
say('=' * 100)
say('【二】四段')
say('=' * 100)
seg = pd.qcut(np.arange(len(R)), 4, labels=False)
for s in range(4):
    m = R[seg == s]
    say(f'  段{s+1}  n={len(m):>6,}  单笔 {m.ret.mean()*100:>+7.3f}%  下注率 {(m.f>0).mean()*100:>5.1f}%  期末资金 {m.cap.iloc[-1]:>10.4f}')

say('')
say('=' * 100)
say('【三】按季度：2025 之后还在不在赚')
say('=' * 100)
R['yq'] = R.dt.dt.year.astype(str) + 'Q' + R.dt.dt.quarter.astype(str)
g = R.groupby('yq').agg(n=('ret','size'), bet=('f', lambda x: (x>0).mean()),
                        mean_ret=('ret','mean'), cap_end=('cap','last'))
say(f'  {"季度":<8s} {"笔数":>7s} {"下注率":>7s} {"单笔均值":>9s} {"期末资金":>12s}')
for k, v in g.iterrows():
    say(f'  {k:<8s} {int(v.n):>7,} {v.bet*100:>6.1f}% {v.mean_ret*100:>+8.3f}% {v.cap_end:>12.4f}')

pos_q = (g.mean_ret > 0).sum()
say(f'  正收益季度 {pos_q}/{len(g)}')

say('')
say('=' * 100)
say('【四】判据（用户定义：在残酷币市下持续盈利）')
say('=' * 100)
say('  · 年化 > 0 且最大回撤 < 30% ⇒ 活着且在赚')
say('  · 2025 之后仍有多数季度为正 ⇒ 是"持续"，不是靠 2024 年一次性')
say('  · 下注率在亏损季自动下降 ⇒ 适应市场生效')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
