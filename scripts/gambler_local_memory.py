# -*- coding: utf-8 -*-
r"""
gambler_local_memory.py — 局部记忆（K=5），不是求平均（K=50）

用户点破：**每一个买低卖高所需要的入场都不一样**
  → K=50 求平均，等于把"这个具体局面"和"别的局面"混在一起，抹平了差异
  → 赌徒记住的只是「这张牌上次出现，后面怎么了」

设计：
  · 状态 5 维：r5, r15, r60, sig, volr（连续，不分桶）
  · 记忆 = 全部历史（含已走过的测试段）——边测边学
  · 查询 K=5 最近邻（局部，不平均）
  · 仓位 = 局部期望/局部方差 × 0.25（凯利）；K=5 方差大 ⇒ 自动小仓
  · 停牌：最近 8 笔实际盈亏累计 < -0.4% ⇒ 停 50 笔
  · 判据：资金曲线（存活 + 持续盈利），不是单笔均值
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\local_memory.txt'
NSYM = int(os.environ.get('LM_NSYM', '60'))

COST      = 0.0009
H         = 176
STEP_TR   = 100
STEP_TE   = 50
OBSERVE   = 0.50
NBLOCK    = 30
K         = 5
KELLY     = 0.25
POS_MAX   = 0.02
STOP_N    = 8
STOP_LOSS = 0.004
STOP_BARS = 50

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def build(g):
    c = g['close'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    F = []
    for k in (5, 15, 60):
        F.append(np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1.0]))
    vma = rmean(v, 60)
    F.append(v/np.where(vma > 0, vma, np.nan))
    F.append(pd.Series(c).pct_change().rolling(240).std().values)
    # 顺序: r5, r15, r60, volr, sig
    return np.column_stack(F)

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def main():
    say('='*100)
    say('gambler_local_memory · 局部记忆 K=5（每个局面有它自己的答案）')
    say('='*100)
    say(f'K={K} | 块 {NBLOCK} | 观察 {OBSERVE:.0%} | 凯利×{KELLY} 上限{POS_MAX*100:.0f}% | 停牌 {STOP_N}笔/{STOP_LOSS*100:.1f}%')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    Xs, ys, ts = [], [], []
    for s, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        F = build(g)
        c = g['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
        Xs.append(F[ok]); ys.append(fwd[ok].astype(np.float64)); ts.append(g['open_time'].values[ok])
    X = np.concatenate(Xs); y = np.concatenate(ys); T = np.concatenate(ts)
    o = np.argsort(T, kind='stable'); X = X[o]; y = y[o]; T = T[o]
    say(f'  样本 {len(X):,}  {time.time()-t0:.0f}s')

    n = len(X)
    cut0 = int(n*OBSERVE)
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0)+1e-9
    Xn = (X-mu)/sd
    say(f'  观察段结束 {pd.to_datetime(T[cut0], unit="ms")}')

    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    pos_l, ret_l = [], []

    for i in range(NBLOCK):
        tr_end = edges[i]
        te_s, te_e = edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)      # 记忆：全部过去（含已走过的测试段）
        if len(sub) < 200:
            say(f'    块 {i+1}/{NBLOCK} 记忆太少'); continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp = r.mean(1)
        var = r.var(1) + 1e-10
        pos = np.where(exp > 0, np.clip(exp/var*KELLY, 0, POS_MAX), 0.0)
        pos_l.append(pos); ret_l.append(y[te_idx])
        say(f'    块 {i+1:>2}/{NBLOCK} 记忆{len(sub):>6} 下注率{(pos>0).mean()*100:5.1f}% 均仓{pos.mean()*100:5.3f}% {time.time()-t0:.0f}s')

    P0 = np.concatenate(pos_l); R = np.concatenate(ret_l)

    # 停牌（逐点）
    pause = 0
    g_hist = deque(maxlen=40)
    P = np.zeros_like(P0)
    n_pause = 0
    for k in range(len(P0)):
        if pause > 0:
            pause -= 1; n_pause += 1; continue
        P[k] = P0[k]
        if P0[k] > 0:
            g_hist.append(P0[k]*R[k])
            if len(g_hist) >= STOP_N and sum(list(g_hist)[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS

    pnl = P*R
    cap = np.cumprod(1.0+pnl)
    dd = (cap/np.maximum.accumulate(cap)-1.0).min()

    say()
    say('='*100)
    say('【一】资金曲线')
    say('='*100)
    say(f'  期末 {cap[-1]:.4f}   最大回撤 {dd*100:.2f}%')
    say(f'  下注 {(P>0).sum():,} 次  停牌 {n_pause:,} 次  下注率 {(P>0).mean()*100:.2f}%')
    say(f'  仓位均值 {P.mean()*100:.3f}%   仓位加权单笔 {(P*R).sum()/max(P.sum(),1e-9)*100:+.3f}%')
    cap1 = np.cumprod(1.0+R)
    say(f'  A 满仓对照 期末 {cap1[-1]:.8f}')

    say()
    say('='*100)
    say('【二】四段')
    say('='*100)
    n4 = len(P); ok4 = True
    for i in range(4):
        aa = i*n4//4; bb = (i+1)*n4//4
        seg = pnl[aa:bb]
        c_seg = np.cumprod(1.0+seg)[-1] if len(seg) else 1.0
        say(f'  段{i+1}  下注率 {(P[aa:bb]>0).mean()*100:5.2f}%  段内资金 {c_seg:.4f}')
        if c_seg <= 1.0: ok4 = False
    say(f'  {"✅ 四段都赚" if ok4 else "❌ 有段亏损"}')

    say()
    say('='*100)
    say('【三】判据（存活 + 持续盈利）')
    say('='*100)
    ok = cap[-1] > 1.0 and dd > -0.30
    say(f'  期末 {cap[-1]:.4f} {"✅" if cap[-1]>1 else "❌"}   回撤 {dd*100:.2f}% {"✅" if dd>-0.30 else "❌"}')
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}；四段 {"✅" if ok4 else "❌"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'期末 {cap[-1]:.4f} 回撤 {dd:.4f}\n')

if __name__ == '__main__':
    main()
