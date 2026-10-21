# -*- coding: utf-8 -*-
r"""
gambler_final_diag.py — 363 倍的最后一击

已排除的原因（都做过实验）：
  ✗ 重叠（overlap_diag：A 0.9867 < B 0.9968）
  ✗ 跨币记忆（cross_mem：0.9964）
  ✗ 全局停牌（global_stop：0.9822）
  ✗ 仓位太小（sizing：放大 5 倍只到 1.0080）

剩下的唯一嫌疑：**全局一条复利曲线 + 肥尾**。

本诊断用同一份 exp/var/R，跑两种口径，把差别摊开：
  A 全局口径（local_memory 的做法）：所有币的点混在一条 cumprod 上
  B 每币独立口径（真实账户）

输出：两条 cap、每笔贡献 P*R 的分位数、最好 10 笔占全部盈亏的比例。
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\final_diag.txt'
NSYM = int(os.environ.get('FD_NSYM', '60'))

COST    = 0.0009
H       = 176
STEP_TR = 100
STEP_TE = 50
OBSERVE = 0.50
NBLOCK  = 30
K       = 5
KELLY   = 0.25
POS_MAX = 0.02
STOP_N  = 8
STOP_LOSS = 0.004
STOP_BARS = 50
H_STEPS = int(np.ceil(H/STEP_TE))

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
    say('gambler_final_diag · 全局复利曲线 vs 每币独立（同一份 exp/var/R）')
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
    e = exp[idx]; v = var[idx]; R = y[idx]; S_ = S[idx]
    raw = np.where(e > 0, np.clip(e/np.maximum(v, 1e-10)*KELLY, 0, POS_MAX), 0.0)
    contrib = raw*R                     # 每笔对"总资金"的贡献

    say()
    say('='*100)
    say('【一】每笔贡献 P*R 的分布（这是关键）')
    say('='*100)
    nz = contrib[raw > 0]
    say(f'  有仓位的点 {len(nz):,} / {len(contrib):,}')
    for q in (1, 5, 25, 50, 75, 95, 99, 99.9):
        say(f'  P{q:<5} {np.percentile(nz, q)*100:>+8.4f}%')
    say(f'  均值 {nz.mean()*100:+.4f}%   中位 {np.median(nz)*100:+.4f}%')
    srt = np.sort(nz)[::-1]
    tot = nz.sum()
    say(f'  最好 10 笔合计 {srt[:10].sum()*100:+.2f}%  占全部 {srt[:10].sum()/max(tot,1e-12)*100:+.1f}%')
    say(f'  最好 100 笔合计 {srt[:100].sum()*100:+.2f}%  占全部 {srt[:100].sum()/max(tot,1e-12)*100:+.1f}%')
    say(f'  （若"占全部"远大于 100%，说明总盈亏被少数笔主导）')

    say()
    say('='*100)
    say('【二】两种口径的期末')
    say('='*100)
    # A 全局口径（local_memory 的做法）
    capA = np.cumprod(1.0+contrib)
    # B 每币独立（不重叠 + 每币停牌）
    finals = []
    for sym in np.unique(S_):
        m = S_ == sym
        pp = raw[m]; rr = R[m]
        n2 = len(pp); Q = np.zeros(n2); nxt = 0
        pause = 0; gh = []
        for k2 in range(n2):
            if k2 < nxt: continue
            if pause > 0: pause -= 1; nxt = k2+1; continue
            if pp[k2] <= 0: continue
            Q[k2] = pp[k2]; nxt = k2+H_STEPS
            gh.append(pp[k2]*rr[k2])
            if len(gh) >= STOP_N and sum(gh[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS
        finals.append(np.cumprod(1.0+Q*rr)[-1])
    finals = np.array(finals)
    say(f'  A 全局一条曲线（local_memory 口径）  期末 {capA[-1]:>12.4f}')
    say(f'  B 每币独立（真实账户）              期末 {finals.mean():>12.4f}')
    say(f'  ⇒ 差 {capA[-1]/max(finals.mean(),1e-9):.1f} 倍')

    say()
    say('='*100)
    say('【三】结论')
    say('='*100)
    say('  · 全局口径把 60 个币的「同一时刻同时开仓」串成一条复利曲线')
    say('  · 真实账户是 60 个独立子账户，各自复利，最后等权平均')
    say('  · 两者差多少倍，就是「虚假并发」贡献了多少')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'global {capA[-1]:.4f}\npercoin {finals.mean():.4f}\n')

if __name__ == '__main__':
    main()
