# -*- coding: utf-8 -*-
r"""
gambler_robust_check.py — 验伪：K=5 的 363 倍是不是过拟合/肥尾

刚跑出 K=5 局部记忆：期末 363.87、回撤 -11.70%、四段全赚
但 363 倍太夸张，必须先验：

  1. K 敏感性：K=3/5/10/20/50 —— 若只有 K=5 有效，就是过拟合
  2. 去肥尾：删掉最好的 1% / 5% 笔，看还剩多少
  3. 分币：每币单独看，是否集中在少数币

无前视口径与 local_memory 完全一致（每块记忆 = 块之前的全部数据）。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\robust_check.txt'
NSYM = int(os.environ.get('RC_NSYM', '60'))

COST      = 0.0009
H         = 176
STEP_TR   = 100
STEP_TE   = 50
OBSERVE   = 0.50
NBLOCK    = 30
K_LIST    = [3, 5, 10, 20, 50]
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
    return np.column_stack(F)

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def run_k(K, X, y, T, Xn, cut0, n, syms):
    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    pos_l, ret_l, sym_l = [], [], []
    for i in range(NBLOCK):
        tr_end = edges[i]; te_s, te_e = edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 200: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp = r.mean(1); var = r.var(1)+1e-10
        pos = np.where(exp > 0, np.clip(exp/var*KELLY, 0, POS_MAX), 0.0)
        pos_l.append(pos); ret_l.append(y[te_idx]); sym_l.append(syms[te_idx])
    P0 = np.concatenate(pos_l); R = np.concatenate(ret_l); S = np.concatenate(sym_l)
    # 停牌
    pause = 0; g_hist = deque(maxlen=40); P = np.zeros_like(P0)
    for k in range(len(P0)):
        if pause > 0:
            pause -= 1; continue
        P[k] = P0[k]
        if P0[k] > 0:
            g_hist.append(P0[k]*R[k])
            if len(g_hist) >= STOP_N and sum(list(g_hist)[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS
    pnl = P*R
    cap = np.cumprod(1.0+pnl)
    dd = (cap/np.maximum.accumulate(cap)-1.0).min()
    return P, R, S, pnl, cap, dd

def main():
    say('='*100)
    say('gambler_robust_check · 验伪 K=5 的 363 倍')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    Xs, ys, ts, ss = [], [], [], []
    for s, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        F = build(g)
        c = g['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
        Xs.append(F[ok]); ys.append(fwd[ok].astype(np.float64)); ts.append(g['open_time'].values[ok])
        ss.append(np.full(int(ok.sum()), str(s), dtype=object))
    X = np.concatenate(Xs); y = np.concatenate(ys); T = np.concatenate(ts); S = np.concatenate(ss)
    o = np.argsort(T, kind='stable'); X=X[o]; y=y[o]; T=T[o]; S=S[o]
    say(f'  样本 {len(X):,}  {time.time()-t0:.0f}s')

    n = len(X); cut0 = int(n*OBSERVE)
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0)+1e-9
    Xn = (X-mu)/sd

    say()
    say('='*100)
    say('【一】K 敏感性（若只有 K=5 有效 ⇒ 过拟合信号）')
    say('='*100)
    results = {}
    for K in K_LIST:
        P, R, S_, pnl, cap, dd = run_k(K, X, y, T, Xn, cut0, n, S)
        w = P.sum()
        wavg = (P*R).sum()/w if w > 0 else 0.0
        # 四段
        n4 = len(P); segs = []
        for i in range(4):
            aa = i*n4//4; bb = (i+1)*n4//4
            segs.append(np.cumprod(1.0+pnl[aa:bb])[-1] if bb>aa else 1.0)
        ok4 = all(v > 1.0 for v in segs)
        results[K] = dict(P=P, R=R, S=S_, pnl=pnl, cap=cap, dd=dd, wavg=wavg, segs=segs, ok4=ok4)
        say(f'  K={K:<3} 期末 {cap[-1]:>12.4f}  回撤 {dd*100:7.2f}%  加权单笔 {wavg*100:+7.3f}%  下注率 {(P>0).mean()*100:5.2f}%  四段 {"✅" if ok4 else "❌"} [{segs[0]:.2f} {segs[1]:.2f} {segs[2]:.2f} {segs[3]:.2f}]')
        say(f'        {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【二】去肥尾（K=5）：删掉最好的 1% / 5% 笔后还剩多少')
    say('='*100)
    r5 = results[K_LIST[1]] if 5 in results else None
    if r5 is not None:
        P, R = r5['P'], r5['R']
        m = P > 0
        contrib = (P*R)[m]      # 每笔下注的实际盈亏
        idx = np.argsort(contrib)[::-1]
        for cut in (0.01, 0.05, 0.10):
            k_ = int(len(contrib)*cut)
            drop = idx[:k_]
            pr = contrib.copy(); pr[drop] = 0.0
            cap_x = np.cumprod(1.0 + pr)
            say(f'  删最好 {cut*100:.0f}%（{k_} 笔） 期末 {cap_x[-1]:>12.4f}')
        say(f'  不删            期末 {np.cumprod(1.0+contrib)[-1]:>12.4f}')
        say(f'  最好 1% 笔占全部盈亏的 {contrib[idx[:max(1,int(len(contrib)*0.01))]].sum()/max(contrib.sum(),1e-9)*100:.1f}%')

    say()
    say('='*100)
    say('【三】分币（K=5）：收益是否集中在少数币')
    say('='*100)
    if r5 is not None:
        P, R, S_ = r5['P'], r5['R'], r5['S']
        pnl = P*R
        uniq = np.unique(S_)
        per = []
        for u in uniq:
            mm = S_ == u
            per.append((u, pnl[mm].sum(), int((P[mm]>0).sum())))
        per.sort(key=lambda x: -x[1])
        tot = sum(p[1] for p in per)
        say(f'  币数 {len(per)}   总盈亏 {tot*100:+.1f}%')
        for u, v, c_ in per[:5]:
            say(f'    {u:<14} {v*100:+8.1f}%  下注 {c_:>6}')
        say('    …')
        for u, v, c_ in per[-3:]:
            say(f'    {u:<14} {v*100:+8.1f}%  下注 {c_:>6}')
        pos_cnt = sum(1 for _, v, _ in per if v > 0)
        say(f'  盈利的币 {pos_cnt}/{len(per)}')
        top5 = sum(p[1] for p in per[:5])
        say(f'  前 5 币占总盈亏 {top5/max(tot,1e-9)*100:.1f}%')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    say('  · K 从 3 到 50 都为正 ⇒ 不是"只有 K=5 才行"，是真信号')
    say('  · 删最好 1% 后仍 > 1 ⇒ 不是靠少数肥尾')
    say('  · 盈利币 > 50% ⇒ 不是靠少数币')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for K, r in results.items():
            f.write(f'K={K} 期末 {r["cap"][-1]:.4f} 回撤 {r["dd"]:.4f}\n')

if __name__ == '__main__':
    main()
