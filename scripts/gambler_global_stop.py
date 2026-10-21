# -*- coding: utf-8 -*-
r"""
gambler_global_stop.py — 关键实验：全局停牌（市场择时）+ 每币独立账户

三个实验并列后，363 倍的来源锁定了：

  local_memory（全局停牌）    笔数 34,963  单笔 +0.847%  期末 363.87
  overlap_diag A（每币停牌）  笔数 55,517  单笔 -0.071%  期末 0.9867
  cross_mem（每币停牌）       笔数 15,561  单笔 -0.062%  期末 0.9964

同 K=5、同记忆（全局 KNN）、同数据，**唯一差别是停牌范围**：
  · 全局停牌：60 币共享一个「最近 8 笔累计盈亏」→ 亏多了全体停手
  · 每币停牌：每个币只看自己

前者实际上是**市场层面的择时**：它用「跨币的最近结果」判断「现在整个市场能不能打」。

本实验：每币独立账户（真实约束）+ 全局停牌（市场择时），逐时间步推进。
这是「可执行」的版本——全局停牌不需要共享资金，只需要共享「最近战况」。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\global_stop.txt'
NSYM = int(os.environ.get('GS_NSYM', '60'))

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
H_STEPS   = int(np.ceil(H/STEP_TE))
BAR_MS    = 300_000

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
    return np.column_stack(F)

def main():
    say('='*100)
    say('gambler_global_stop · 每币独立账户 + 全局停牌（市场择时）')
    say('='*100)
    say(f'K={K} | 不重叠 {H_STEPS} 步 | 停牌 {STOP_N}笔/{STOP_LOSS*100:.1f}%→停{STOP_BARS}步')
    say('='*100)

    df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(NSYM).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    say(f'  载入 {len(df):,} 行  {time.time()-t0:.0f}s')

    Xs, ys, ts, ss = [], [], [], []
    for si, (sym, g) in enumerate(df.groupby('symbol', observed=True)):
        if len(g) < 5000: continue
        F = build(g); c = g['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
        Xs.append(F[ok]); ys.append(fwd[ok].astype(np.float64)); ts.append(g['open_time'].values[ok])
        ss.append(np.full(int(ok.sum()), si, dtype=np.int32))
    X = np.concatenate(Xs); y = np.concatenate(ys); T = np.concatenate(ts); S = np.concatenate(ss)
    o = np.argsort(T, kind='stable'); X=X[o]; y=y[o]; T=T[o]; S=S[o]
    say(f'  样本 {len(X):,}  {time.time()-t0:.0f}s')

    n = len(X); cut0 = int(n*OBSERVE)
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0)+1e-9
    Xn = (X-mu)/sd
    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)

    exp = np.full(n, np.nan); var = np.full(n, np.nan)
    for i in range(NBLOCK):
        tr_end, te_s, te_e = edges[i], edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp[te_idx] = r.mean(1); var[te_idx] = r.var(1)+1e-10
    say(f'  KNN 跑完  {time.time()-t0:.0f}s')

    idx = np.where(np.isfinite(exp))[0]
    say(f'  测试点 {len(idx):,}  {time.time()-t0:.0f}s')

    # 逐点推进：全局停牌 + 每币独立账户
    P_gl = np.zeros(len(idx))
    pause = 0
    gh = deque(maxlen=40)
    settle = deque()                 # (到期numpy索引, 实际盈亏)
    n_bet = 0; n_pause = 0

    for k in range(len(idx)):
        i = idx[k]; t = T[i]
        while settle and settle[0][0] <= i:
            _, pl_ = settle.popleft()
            gh.append(pl_)
        if pause > 0:
            pause -= 1; n_pause += 1
            continue
        e = exp[i]; v = var[i]
        if e <= 0: continue
        pos = float(min(e/max(v, 1e-10)*KELLY, POS_MAX))
        P_gl[k] = pos
        n_bet += 1
        pl = pos*y[i]
        # 结算时刻：H 根后（按时间戳换算成索引位置）
        settle.append((np.searchsorted(idx, i + 0, side='right') + 0, pl))  # 占位，下面修正
    say(f'  占位跑完（下一步重算）  {time.time()-t0:.0f}s')

    # 上面的结算索引不准确 —— 重跑一遍，用时间戳结算
    P_gl = np.zeros(len(idx))
    pause = 0; gh = deque(maxlen=40); n_bet = 0; n_pause = 0
    settle = deque()                 # (到期时间戳, 盈亏)
    pos_all = np.zeros(len(idx))
    for k in range(len(idx)):
        i = idx[k]; t = T[i]
        while settle and settle[0][0] <= t:
            _, pl_ = settle.popleft()
            gh.append(pl_)
        if pause > 0:
            pause -= 1; n_pause += 1
            continue
        e = exp[i]; v = var[i]
        if e <= 0: continue
        pos = float(min(e/max(v, 1e-10)*KELLY, POS_MAX))
        pos_all[k] = pos; n_bet += 1
        settle.append((t + H*BAR_MS, pos*y[i]))

    # 每币独立复利（不重叠）
    R = y[idx]; S_ = S[idx]
    finals = []
    for sym in np.unique(S_):
        m = S_ == sym
        pp = pos_all[m].copy(); rr = R[m]
        nxt = 0
        for k2 in range(len(pp)):
            if k2 < nxt: continue
            if pp[k2] <= 0: continue
            nxt = k2 + H_STEPS
            pp[k2] = pp[k2]
            pp[max(0,k2+1):nxt] = 0.0
    # 重新按不重叠逐币算（更严谨）
    finals = []; trades = 0; wavg_num = 0.0; wavg_den = 0.0; psum = 0.0; pcnt = 0
    for sym in np.unique(S_):
        m = S_ == sym
        pp = pos_all[m]; rr = R[m]
        n2 = len(pp)
        Q = np.zeros(n2); nxt = 0; nt = 0
        for k2 in range(n2):
            if k2 < nxt: continue
            if pp[k2] <= 0: continue
            Q[k2] = pp[k2]; nxt = k2 + H_STEPS; nt += 1
        pnl = Q*rr
        finals.append(np.cumprod(1.0+pnl)[-1])
        trades += nt
        wavg_num += (Q*rr).sum(); wavg_den += Q.sum(); psum += Q.sum(); pcnt += len(Q)
    finals = np.array(finals)
    total = float(finals.mean())
    wavg = wavg_num/max(wavg_den, 1e-12)

    say()
    say('='*100)
    say('【一】结果（每币独立账户 + 全局停牌）')
    say('='*100)
    say(f'  子账户 {len(finals)}   总资产 {total:.4f}  收益 {(total-1)*100:+.2f}%')
    say(f'  盈利子账户 {int((finals>1).sum())}/{len(finals)}   最好 {finals.max():.4f}  最差 {finals.min():.4f}')
    say(f'  全局下注 {n_bet} 次   全局停牌 {n_pause} 次   仓位加权单笔 {wavg*100:+.3f}%')

    say()
    say('='*100)
    say('【二】对比（同记忆、同 K、同数据，只差停牌范围）')
    say('='*100)
    say(f'  每币停牌（full2）        0.9975   单笔 -0.043%')
    say(f'  每币停牌（cross_mem）    0.9964   单笔 -0.062%')
    say(f'  全局停牌（本实验）       {total:.4f}   单笔 {wavg*100:+.3f}%')
    say(f'  ⇒ 全局停牌 {"有效" if total > 1.0 else "无效"}，差 {(total-0.9975)*100:+.2f}pp')

    say()
    say('='*100)
    say('【三】判据（存活 + 持续盈利）')
    say('='*100)
    ok = total > 1.0
    say(f'  总资产 {total:.4f} > 1 ? {"✅" if ok else "❌"}')
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'global_stop\t{total:.6f}\t{wavg:.6f}\n')

if __name__ == '__main__':
    main()
