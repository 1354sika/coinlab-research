# -*- coding: utf-8 -*-
"""资金曲线检验：在真实币市涨跌概率下，模型能不能"持续盈利"

用户定义：赌徒 = 在残酷的币市涨跌概率下不断盈利。
所以判据不是"单笔均值"，是"资金曲线是不是一路向上"。

本脚本：
  1. 入场 = zigzag 确认底（追涨）
  2. 仓位 = 1/4 凯利（用过去 200 笔滚动估计 p 与盈亏比 b），外加单笔最大亏损 2% 本金
  3. 结算 = 固定持 176 根
  4. 严格无前视：仓位只用已结算的交易估计

输出四个存活指标：
  ① 期末资金 / 年化
  ② 最大回撤
  ③ 破产概率（对历史逐笔收益做 10000 次 bootstrap 重排）
  ④ 按月盈利：多少个月为正 / 为负

输出：evidence/capital.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CP_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\capital.txt'
Z = float(os.environ.get('CP_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('CP_SLIP', '0.0005'))
NSYM = int(os.environ.get('CP_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('CP_HOLD', '176'))
KELLY = float(os.environ.get('CP_KELLY', '0.25'))
MAXLOSS = float(os.environ.get('CP_MAXLOSS', '0.02'))
WARM = int(os.environ.get('CP_WARM', '200'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('资金曲线检验 · 在真实涨跌概率下能不能持续盈利')
say(f'  {os.path.basename(DATA)}  持有 {HOLD} 根  成本 {COST:.3%}  {KELLY}凯利  单笔上限 {MAXLOSS:.1%}')
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
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    for (a, e, b) in zigzag_entry(C, Z):
        j = min(e + HOLD, T - 1)
        if j <= e:
            continue
        rows.append((ts[e], C[j] / C[e] - 1.0 - COST))

R = pd.DataFrame(rows, columns=['t', 'ret']).sort_values('t').reset_index(drop=True)
say(f'  入场 {len(R):,} 笔  {time.time()-t0:.0f}s')

# ---- 1/4 凯利 + 单笔亏损硬顶，严格用历史估计 ----
cap = 1.0
peak = 1.0
maxdd = 0.0
hist = []
f_list = []
cap_series = []
for i, row in R.iterrows():
    r = row.ret
    if len(hist) >= WARM:
        h = np.array(hist[-WARM:])
        p = (h > 0).mean()
        win = h[h > 0].mean() if (h > 0).any() else 0.0
        los = -h[h <= 0].mean() if (h <= 0).any() else 1e-9
        b = win / max(los, 1e-9)
        edge = p * b - (1 - p)
        f = KELLY * edge / max(b, 1e-9) if b > 0 else 0.0
        f = max(0.0, min(f, MAXLOSS / max(los, 1e-9), 0.25))
    else:
        f = 0.0
    pnl = f * r
    cap *= (1.0 + pnl)
    cap_series.append(cap)
    if cap > peak:
        peak = cap
    dd = cap / peak - 1.0
    if dd < maxdd:
        maxdd = dd
    hist.append(r)
    f_list.append(f)

R['cap'] = cap_series
R['f'] = f_list
R['dt'] = pd.to_datetime(R.t, unit='ms')

say('')
say('=' * 100)
say('【一】资金曲线')
say('=' * 100)
yrs = (R.t.max() - R.t.min()) / (365.25 * 24 * 3600 * 1000)
ann = cap ** (1 / yrs) - 1 if yrs > 0 else 0
say(f'  样本期 {yrs:.2f} 年')
say(f'  期末资金 {cap:.4f}   年化 {ann*100:+.2f}%')
say(f'  最大回撤 {maxdd*100:.2f}%')
say(f'  平均仓位 {np.mean([f for f in f_list if f > 0])*100 if any(f>0 for f in f_list) else 0:.2f}%  '
    f'弃牌率 {(np.array(f_list)==0).mean()*100:.1f}%')

say('')
say('=' * 100)
say('【二】按年 / 按季度：是不是"持续"盈利')
say('=' * 100)
R['yq'] = R.dt.dt.year.astype(str) + 'Q' + R.dt.dt.quarter.astype(str)
g = R.groupby('yq').agg(n=('ret', 'size'), mean_ret=('ret', 'mean'),
                        f_mean=('f', 'mean'), cap_end=('cap', 'last'))
say(f'  {"季度":<8s} {"笔数":>8s} {"单笔均值":>9s} {"平均仓位":>9s} {"期末资金":>10s}')
for k, v in g.iterrows():
    say(f'  {k:<8s} {int(v.n):>8,} {v.mean_ret*100:>+8.3f}% {v.f_mean*100:>8.2f}% {v.cap_end:>10.4f}')

pos_q = (g.mean_ret > 0).sum()
say(f'  正收益季度 {pos_q}/{len(g)}')

say('')
say('=' * 100)
say('【三】破产概率（1 万次 bootstrap 重排）')
say('=' * 100)
rng = np.random.default_rng(20261018)
rets = R.ret.values
fs = R.f.values
N = 10000
broke = 0
final = []
for _ in range(N):
    idx = rng.integers(0, len(rets), len(rets))
    c = 1.0
    for k in idx:
        c *= (1.0 + fs[k] * rets[k])
        if c < 0.01:
            broke += 1
            break
    final.append(c)
final = np.array(final)
say(f'  破产（<1% 本金）概率 {broke/N*100:.2f}%')
say(f'  期末资金分位: p5 {np.percentile(final,5):.4f}  p50 {np.percentile(final,50):.4f}  p95 {np.percentile(final,95):.4f}')

say('')
say('=' * 100)
say('【四】判据（按用户定义：在残酷币市下持续盈利）')
say('=' * 100)
say('  · 年化 > 0 且最大回撤 < 30% 且破产概率 < 1% ⇒ 活下来了')
say('  · 正收益季度 >= 3/4 ⇒ 是"持续"，不是靠某一个季度撑起来的')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
