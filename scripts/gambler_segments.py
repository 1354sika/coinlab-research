# -*- coding: utf-8 -*-
"""筛出"扣掉滑点和手续费还稳赚"的低买高卖段。

核心难点：不能用峰谷价记账（那是事后诸葛亮）。
必须用**确认价**：
  底：从最低点反弹 Z% 才确认是底 → 在 low*(1+Z) 买入
  顶：从最高点回落 Z% 才确认是顶 → 在 high*(1-Z) 卖出

一段真实可拿收益 = high*(1-Z) / (low*(1+Z)) - 1 - 成本
成本 = 手续费 0.04% + 滑点（可调，默认 0.05%）

输出：
  ① 全部段 / 扣成本后为正的段 / 净赚 >0.5% 的段  各多少
  ② 这些段平均赚多少、持续多少根
  ③ 四段稳定性（正收益段的比例是否随时间变化）
  ④ 按币分布
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('SG_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\segments.txt'
Z = float(os.environ.get('SG_Z', '0.03'))        # 反弹/回落多少确认转折
FEE = 0.0004
SLIP = float(os.environ.get('SG_SLIP', '0.0005'))
NSYM = int(os.environ.get('SG_NSYM', '60'))
COST = FEE + SLIP
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('低买高卖段筛选 · 用确认价记账，扣滑点与手续费')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}(费{FEE:.2%}+滑{SLIP:.2%})  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def zigzag(C, z):
    """返回 [(i_lo, i_hi)]：每个上升段。用确认价是调用方的事。"""
    T = len(C)
    segs = []
    trend = 0            # 0 未知, 1 上升, -1 下降
    hi_i = lo_i = 0
    hi_v = lo_v = C[0]
    for i in range(1, T):
        c = C[i]
        if trend >= 0:
            if c > hi_v:
                hi_v, hi_i = c, i
            if c <= hi_v * (1 - z):
                # 顶确认
                if trend == 1:
                    segs.append((lo_i, hi_i))
                trend = -1
                lo_v, lo_i = c, i
        if trend <= 0:
            if c < lo_v:
                lo_v, lo_i = c, i
            if c >= lo_v * (1 + z):
                trend = 1
                hi_v, hi_i = c, i
    return segs

rows = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    T = len(C)
    if T < 2000:
        continue
    segs = zigzag(C, Z)
    for (a, b) in segs:
        lo = C[a]; hi = C[b]
        entry = lo * (1 + Z)      # 确认底部后买入
        exit_ = hi * (1 - Z)      # 确认顶部后卖出
        gross = exit_ / entry - 1.0
        net = gross - COST
        rows.append((sym, a, b, b - a, gross, net))

D = pd.DataFrame(rows, columns=['symbol', 'i_lo', 'i_hi', 'bars', 'gross', 'net'])
say(f'  找到 {len(D):,} 个上涨段 / {D.symbol.nunique()} 币  {time.time()-t0:.0f}s')

say('')
say('=' * 100)
say('【一】筛选结果')
say('=' * 100)
n = len(D)
pos = (D.net > 0).sum()
say(f'  全部段            {n:>8,}')
say(f'  扣成本后为正       {pos:>8,}  ({pos/n*100:5.1f}%)')
for thr in (0.005, 0.01, 0.02, 0.05):
    c = (D.net > thr).sum()
    say(f'  净赚 > {thr:>5.1%}        {c:>8,}  ({c/n*100:5.1f}%)')

say('')
say('=' * 100)
say('【二】净赚段长什么样')
say('=' * 100)
P = D[D.net > 0]
say(f'  数量 {len(P):,}   平均净赚 {P.net.mean()*100:+.3f}%   中位 {P.net.median()*100:+.3f}%')
say(f'  平均持有 {P.bars.mean():.0f} 根（5m 粒度 ≈ {P.bars.mean()/288:.1f} 天）  中位 {P.bars.median():.0f} 根')
say(f'  净赚分位: p10 {P.net.quantile(.1)*100:+.2f}%  p50 {P.net.quantile(.5)*100:+.2f}%  p90 {P.net.quantile(.9)*100:+.2f}%')
say(f'  每币中位数 {P.groupby("symbol").size().median():.0f} 段')

say('')
say('=' * 100)
say('【三】四段稳定性：正收益段占比是否随时间变化')
say('=' * 100)
D['seg'] = pd.qcut(D.groupby('symbol').cumcount(), 4, labels=False)
say(f'  {"段":>3s} {"总段数":>10s} {"净赚段":>10s} {"占比":>8s} {"净赚段均值":>12s}')
for sg in range(4):
    s = D[D.seg == sg]
    p = s[s.net > 0]
    say(f'  {sg+1:>3d} {len(s):>10,} {len(p):>10,} {len(p)/max(len(s),1)*100:>7.1f}% {p.net.mean()*100 if len(p) else 0:>11.3f}%')

say('')
say('=' * 100)
say('【四】按币分布（净赚段数最多的 10 个 / 最少的 5 个）')
say('=' * 100)
cnt = D[D.net > 0].groupby('symbol').agg(n=('net', 'size'), m=('net', 'mean')).sort_values('n', ascending=False)
say('  最多:')
for k, v in cnt.head(10).iterrows():
    say(f'    {k:14s} {int(v.n):>5} 段  均值 {v.m*100:+.3f}%')
say('  最少:')
for k, v in cnt.tail(5).iterrows():
    say(f'    {k:14s} {int(v.n):>5} 段  均值 {v.m*100:+.3f}%')

say('')
say('=' * 100)
say('【五】判据与诚实边界')
say('=' * 100)
say('  · 这些段是"事后确认出来的"：底要在涨了 Z% 之后才确认，顶要在跌了 Z% 之后才确认')
say('  · 所以入场价 = 底*(1+Z)、出场价 = 顶*(1-Z)，已经扣掉了确认延迟的代价')
say('  · 下一步要问的是：这些段的**入场那一刻**，状态能不能认出来（AUC）')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
