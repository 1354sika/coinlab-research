# -*- coding: utf-8 -*-
"""补上 segments 缺的那一半：亏损段有多亏。

上一版只报了"净赚段 22,012 个、平均 +4.43%"，没报亏的 30,303 段平均亏多少。
没有这个数，期望就是未知的——42% 赚钱不等于这件事赚钱。

本脚本算：
  ① 正段/负段各多少、均值多少
  ② 单笔期望 = p*平均盈利 + (1-p)*平均亏损
  ③ 按时间顺序把每段都做一遍，资金曲线是什么形状
  ④ 四段稳定性
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('SG_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\segments_exp.txt'
Z = float(os.environ.get('SG_Z', '0.03'))
FEE, SLIP = 0.0004, 0.0005
COST = FEE + SLIP
NSYM = int(os.environ.get('SG_NSYM', '60'))
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('段的期望值 · 正段赚多少、负段亏多少')
say(f'  Z={Z:.1%}  成本={COST:.3%}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]

def zigzag(C, z):
    T = len(C); segs = []
    trend = 0; hi_i = lo_i = 0; hi_v = lo_v = C[0]
    for i in range(1, T):
        c = C[i]
        if trend >= 0:
            if c > hi_v:
                hi_v, hi_i = c, i
            if c <= hi_v * (1 - z):
                if trend == 1:
                    segs.append((lo_i, hi_i))
                trend = -1; lo_v, lo_i = c, i
        if trend <= 0:
            if c < lo_v:
                lo_v, lo_i = c, i
            if c >= lo_v * (1 + z):
                trend = 1; hi_v, hi_i = c, i
    return segs

rows = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    if len(C) < 2000:
        continue
    for (a, b) in zigzag(C, Z):
        entry = C[a] * (1 + Z)
        exit_ = C[b] * (1 - Z)
        rows.append((sym, a, b - a, exit_ / entry - 1.0 - COST))

D = pd.DataFrame(rows, columns=['symbol', 'i', 'bars', 'net'])
say(f'  段数 {len(D):,}  {time.time()-t0:.0f}s')

say('')
say('=' * 100)
say('【一】正段 vs 负段')
say('=' * 100)
P = D[D.net > 0]; L = D[D.net <= 0]
say(f'  正段 {len(P):>7,} ({len(P)/len(D)*100:5.1f}%)  平均 {P.net.mean()*100:+.3f}%  中位 {P.net.median()*100:+.3f}%')
say(f'  负段 {len(L):>7,} ({len(L)/len(D)*100:5.1f}%)  平均 {L.net.mean()*100:+.3f}%  中位 {L.net.median()*100:+.3f}%')
say(f'  全体平均 {D.net.mean()*100:+.3f}%')
say(f'  正段平均持有 {P.bars.mean():.0f} 根   负段平均持有 {L.bars.mean():.0f} 根')
say('')
say('  负段分位: p10 {:.2f}%  p50 {:.2f}%  p90 {:.2f}%'.format(
    L.net.quantile(.1)*100, L.net.quantile(.5)*100, L.net.quantile(.9)*100))
say(f'  负段最惨 {L.net.min()*100:.2f}%   正段最好 {P.net.max()*100:.2f}%')

say('')
say('=' * 100
     if False else '=' * 100)
say('【二】单笔期望与资金曲线')
say('=' * 100)
E = D.net.mean()
say(f'  单笔期望 {E*100:+.4f}%   （正段贡献 {len(P)/len(D)*P.net.mean()*100:+.4f}%，负段贡献 {len(L)/len(D)*L.net.mean()*100:+.4f}%）')
say(f'  {"✅ 正期望" if E > 0 else "❌ 负期望"}')
# 按时间顺序（全市场合并）复利
D2 = D.sort_values('i')
eq = np.cumprod(1 + D2.net.values)
say(f'  全做（每段都下 100% 仓）资金: {eq[-1]:.4f}')
say(f'  最好 {eq.max():.3f}   最差 {eq.min():.4f}')

say('')
say('=' * 100)
say('【三】四段稳定性')
say('=' * 100)
D['seg'] = pd.qcut(D.groupby('symbol').cumcount(), 4, labels=False)
for sg in range(4):
    s = D[D.seg == sg]
    say(f'  段{sg+1}  n={len(s):>7,}  均值 {s.net.mean()*100:>+8.4f}%  正段占比 {(s.net>0).mean()*100:>5.1f}%')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 若全体平均 > 0 ⇒ 这个 zigzag 段本身就是正期望策略（含确认延迟与成本）')
say('  · 若全体平均 <= 0 ⇒ 42% 的正段是被 58% 的负段吃掉的，不能只用正段训练')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
