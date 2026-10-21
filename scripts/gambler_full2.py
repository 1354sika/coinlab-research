# -*- coding: utf-8 -*-
r"""
gambler_full2.py — 真实资金约束：同币不重叠 + 资金共享

full 的 1.0056 仍有虚高：STEP_TE=50、H=176 ⇒ 同币同时背 3.5 层仓。

本版两处硬约束：
  · **同币不重叠**：持仓 176 根内不再开新仓（next_free 指针）
  · **资金共享**：每币子账户 1/60 资金，仓位 = 该子账户当前资金 × P%
    （cumprod 天然按"平仓后资金"滚动，不重叠后这个口径才对）
  · 停牌：连亏 8 笔停 50 步（停牌期间同样不开新仓）

输出：等权组合总资产、盈利子账户数、四段。
这是能拿去实盘形态检验的数字。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\full2.txt'
NSYM = int(os.environ.get('F2_NSYM', '60'))

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

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def run_one(g):
    F = build(g); c = g['close'].values
    fwd = np.full(len(c), np.nan)
    fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
    ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
    X, y = F[ok], fwd[ok]
    n = len(X)
    if n < 2000: return None
    cut0 = int(n*OBSERVE)
    if cut0 < 300: return None
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0)+1e-9
    Xn = (X-mu)/sd
    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    P_l, R_l = [], []
    for i in range(NBLOCK):
        tr_end, te_s, te_e = edges[i], edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp = r.mean(1); var = r.var(1)+1e-10
        pos = np.where(exp > 0, np.clip(exp/var*KELLY, 0, POS_MAX), 0.0)
        P_l.append(pos); R_l.append(y[te_idx])
    if not P_l: return None
    P0 = np.concatenate(P_l); R = np.concatenate(R_l)

    # 同币不重叠 + 停牌
    pause = 0; gh = deque(maxlen=40)
    P = np.zeros_like(P0)
    nxt = 0; n_trades = 0; n_pause = 0
    for k in range(len(P0)):
        if k < nxt: continue
        if pause > 0:
            pause -= 1; nxt = k+1; n_pause += 1; continue
        if P0[k] <= 0:
            continue
        P[k] = P0[k]
        nxt = k + H_STEPS
        n_trades += 1
        gh.append(P0[k]*R[k])
        if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
            pause = STOP_BARS
    pnl = P*R
    return dict(cap=np.cumprod(1.0+pnl), pnl=pnl, P=P, R=R,
                n_obs=len(P), n_trades=n_trades, n_pause=n_pause)

def main():
    say('='*100)
    say('gambler_full2 · 真实资金约束（同币不重叠 + 资金共享）')
    say('='*100)
    say(f'不重叠步数 {H_STEPS}（H={H} / 步长 {STEP_TE}）| K={K} | 凯利×{KELLY} | 停牌 {STOP_N}笔/{STOP_LOSS*100:.1f}%')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    subs = []
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        r = run_one(g)
        if r is None: continue
        subs.append((str(sym), r))
    say(f'  子账户 {len(subs)} 个  {time.time()-t0:.0f}s')

    finals = np.array([r['cap'][-1] for _, r in subs])
    trades = np.array([r['n_trades'] for _, r in subs])

    say()
    say('='*100)
    say('【一】每个子账户（等权，各 1/N）')
    say('='*100)
    say(f'  期末：P10 {np.percentile(finals,10):.4f}  P25 {np.percentile(finals,25):.4f}  P50 {np.percentile(finals,50):.4f}  P75 {np.percentile(finals,75):.4f}  P90 {np.percentile(finals,90):.4f}')
    say(f'  盈利 {int((finals>1).sum())}/{len(finals)}   最好 {finals.max():.4f}  最差 {finals.min():.4f}')
    say(f'  每币成交笔数：P50 {np.percentile(trades,50):.0f}  最多 {trades.max()}  最少 {trades.min()}  合计 {int(trades.sum())}')

    total = finals.mean()
    say()
    say('='*100)
    say('【二】总资产（等权组合）')
    say('='*100)
    say(f'  初始 1.0 → 期末 {total:.4f}   （收益 {(total-1)*100:+.2f}%）')
    say(f'  对照：满仓买入持有 BTC = 见下方时间跨度内价格变化')

    say()
    say('='*100)
    say('【三】四段（等权组合）')
    say('='*100)
    seg_sum = np.zeros(4); cnt = 0
    for _, r in subs:
        pnl = r['pnl']; n4 = len(pnl)
        if n4 < 4: continue
        for i in range(4):
            aa = i*n4//4; bb = (i+1)*n4//4
            seg_sum[i] += np.cumprod(1.0+pnl[aa:bb])[-1]
        cnt += 1
    ok4 = False
    if cnt:
        seg_avg = seg_sum/cnt
        for i in range(4):
            say(f'  段{i+1}  平均段内资金 {seg_avg[i]:.4f}')
        ok4 = all(v > 1.0 for v in seg_avg)
        say(f'  {"✅ 四段都赚" if ok4 else "❌ 有段亏损"}')

    say()
    say('='*100)
    say('【四】判据（存活 + 持续盈利）')
    say('='*100)
    say(f'  总资产 {total:.4f} > 1 ? {"✅" if total>1 else "❌"}')
    say(f'  盈利子账户 {int((finals>1).sum())}/{len(finals)}')
    say(f'  四段 {"✅" if ok4 else "❌"}')
    ok = total > 1.0 and ok4
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for s, r in subs:
            f.write(f'{s}\t{r["cap"][-1]:.6f}\t{r["n_trades"]}\n')
        f.write(f'TOTAL\t{total:.6f}\n')

if __name__ == '__main__':
    main()
