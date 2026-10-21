# -*- coding: utf-8 -*-
r"""
gambler_cross_mem.py — 关键实验：记忆是「跨币」还是「同币」

overlap_diag 的诊断推翻了"重叠造成虚增"的假设：
  A 全部触发（重叠）0.9867  vs  B 不重叠 0.9968   → 重叠反而更差，虚增 1.0 倍
  事后最优 C 也不比 B 好                              → 不是"选错笔"

那 local_memory 的 363.87 / +0.847% 是从哪来的？
  local_memory 的 KNN 建立在「全局拼接、按时间排序」的样本上
  ⇒ 一个币的邻居，很可能来自**另一个币**

full2 / cross / overlap_diag 全是「每币独立建 KNN」（同币记忆）

本实验唯一改动：**记忆跨币，账户仍按币独立**
  · 每块（30 天）用 tr_end 之前的**全部币**的点建 KNN
  · 每个测试点查 KNN 得 exp/var，照旧定仓
  · **账户：每币独立复利，各 1/N 资金**（真实约束）
  · 不重叠 + 连亏停牌

对比 A（同币记忆）vs B（跨币记忆），同一套切分与约束。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\cross_mem.txt'
NSYM = int(os.environ.get('CM_NSYM', '60'))

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

def per_symbol_account(S, R, exp, var, mask):
    """按币独立复利：只处理 mask 内为 True 的预测点"""
    out = {}
    idx = np.where(mask)[0]
    for sym in np.unique(S[idx]):
        ii = idx[S[idx] == sym]
        e = exp[ii]; v = var[ii]; r = R[ii]
        raw = np.where(e > 0, np.clip(e/np.maximum(v, 1e-10)*KELLY, 0, POS_MAX), 0.0)
        pause = 0; gh = deque(maxlen=40)
        P = np.zeros_like(raw); nxt = 0; nt = 0
        for k in range(len(raw)):
            if k < nxt: continue
            if pause > 0:
                pause -= 1; nxt = k+1; continue
            if raw[k] <= 0: continue
            P[k] = raw[k]; nxt = k+H_STEPS; nt += 1
            gh.append(raw[k]*r[k])
            if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS
        pnl = P*r
        cap = np.cumprod(1.0+pnl)
        dd = (cap/np.maximum.accumulate(cap)-1.0).min()
        n4 = len(pnl); segs = []
        for i in range(4):
            aa = i*n4//4; bb = (i+1)*n4//4
            segs.append(np.cumprod(1.0+pnl[aa:bb])[-1] if bb > aa else 1.0)
        out[sym] = dict(cap=cap[-1], dd=dd, segs=segs, trades=nt,
                        wavg=(P*r).sum()/max(P.sum(), 1e-9), pmean=P.mean())
    return out

def main():
    say('='*100)
    say('gambler_cross_mem · 记忆跨币 vs 同币（账户都按币独立）')
    say('='*100)
    say(f'K={K} 不重叠{H_STEPS}步 | 凯利×{KELLY} 上限{POS_MAX*100:.0f}%')
    say('='*100)

    df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(NSYM).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    say(f'  载入 {len(df):,} 行  {time.time()-t0:.0f}s')

    Xs, ys, ts, ss = [], [], [], []
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        F = build(g); c = g['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
        Xs.append(F[ok]); ys.append(fwd[ok].astype(np.float64)); ts.append(g['open_time'].values[ok])
        ss.append(np.full(int(ok.sum()), int(list(map(str, keep)).index(str(sym))), dtype=np.int32))
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
        sub = np.arange(0, tr_end, STEP_TR)          # ← 跨币：全部币的点
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp[te_idx] = r.mean(1); var[te_idx] = r.var(1)+1e-10
        if i % 5 == 0: say(f'    块 {i+1}/{NBLOCK}  {time.time()-t0:.0f}s')

    mask = np.isfinite(exp)
    say(f'  跨币记忆跑完  {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【一】跨币记忆 + 每币独立账户')
    say('='*100)
    acc = per_symbol_account(S, y, exp, var, mask)
    if not acc:
        say('  无子账户'); return
    finals = np.array([v['cap'] for v in acc.values()])
    dds = np.array([v['dd'] for v in acc.values()])
    segs = np.array([v['segs'] for v in acc.values()])
    wavg = np.array([v['wavg'] for v in acc.values()])
    pm = np.array([v['pmean'] for v in acc.values()])
    trades = sum(v['trades'] for v in acc.values())
    total = float(finals.mean()); dd = float(dds.mean()); seg_avg = segs.mean(0)
    ok4 = bool(all(s > 1.0 for s in seg_avg))
    say(f'  子账户 {len(acc)}   总资产 {total:.4f}  收益 {(total-1)*100:+.2f}%  回撤 {dd*100:.2f}%')
    say(f'  单笔 {np.mean(wavg)*100:+.3f}%  均仓 {np.mean(pm)*100:.2f}%  笔数 {trades}')
    say(f'  盈利子账户 {int((finals>1).sum())}/{len(finals)}')
    say(f'  四段 {"✅" if ok4 else "❌"} [{seg_avg[0]:.3f} {seg_avg[1]:.3f} {seg_avg[2]:.3f} {seg_avg[3]:.3f}]')

    say()
    say('='*100)
    say('【二】对比（同口径，账户都按币独立）')
    say('='*100)
    say(f'  同币记忆（full2）   总资产 0.9975  单笔 -0.043%')
    say(f'  跨币记忆（本实验）  总资产 {total:.4f}  单笔 {np.mean(wavg)*100:+.3f}%')
    say(f'  ⇒ 跨币 {"有效" if total > 1.0 else "无效"}，差 {(total-0.9975)*100:+.2f}pp')

    say()
    say('='*100)
    say('【三】判据（存活 + 持续盈利）')
    say('='*100)
    ok = total > 1.0 and dd > -0.30 and ok4
    say(f'  总资产 {total:.4f} {"✅" if total>1 else "❌"}   回撤 {dd*100:.2f}% {"✅" if dd>-0.30 else "❌"}   四段 {"✅" if ok4 else "❌"}')
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'跨币\t{total:.6f}\t{dd:.6f}\n')

if __name__ == '__main__':
    main()
