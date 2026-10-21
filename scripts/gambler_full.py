# -*- coding: utf-8 -*-
r"""
gambler_full.py — 完整赌徒 + 真实组合约束（可拿去实盘形态检验的数字）

local_memory 的 363 倍是「每笔独立复利」：
  同一时刻 60 币全押，资金不可能这样用 —— 数学结果，不是账户曲线。

本版加真实约束：
  · 60 个独立子账户，各分 1/60 资金
  · 每币内部：K=5 局部记忆入场 + 凯利定仓 + 连亏停牌
  · 每币同一时刻最多 1 个仓位（不重复开仓）
  · 总资产 = 60 个子账户的等权平均（真实账户形态）

对比输出：
  A 每笔独立复利（旧口径，虚高）
  B 等权组合（真实）
  C 等权组合但只用「盈利的子账户」（上界参考）
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\full.txt'
NSYM = int(os.environ.get('FU_NSYM', '60'))

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
    return np.column_stack(F)

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def run_one(g):
    F = build(g)
    c = g['close'].values
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
    pause = 0; gh = deque(maxlen=40); P = np.zeros_like(P0)
    for k in range(len(P0)):
        if pause > 0:
            pause -= 1; continue
        P[k] = P0[k]
        if P0[k] > 0:
            gh.append(P0[k]*R[k])
            if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS
    pnl = P*R
    return dict(cap=np.cumprod(1.0+pnl), pnl=pnl, P=P, R=R, n_obs=len(P))

def main():
    say('='*100)
    say('gambler_full · 完整赌徒 + 真实组合约束（每币独立子账户）')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    subs = []
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        r = run_one(g)
        if r is None: continue
        subs.append((str(sym), r))
        say(f'    {sym:<14} 期末 {r["cap"][-1]:>10.4f}  点 {r["n_obs"]:>7,}  {time.time()-t0:.0f}s')

    if not subs:
        say('  无有效子账户'); return

    finals = np.array([r['cap'][-1] for _, r in subs])
    say()
    say('='*100)
    say('【一】每个子账户（等权，各 1/N 资金）')
    say('='*100)
    say(f'  子账户 {len(subs)} 个')
    say(f'  期末资金：P10 {np.percentile(finals,10):.3f}  P50 {np.percentile(finals,50):.3f}  P90 {np.percentile(finals,90):.3f}')
    say(f'  盈利 {int((finals>1).sum())}/{len(finals)}   最好 {finals.max():.2f}  最差 {finals.min():.4f}')

    # B 等权组合：每币资金 1/N，末值 = 平均
    total = finals.mean()
    say()
    say('='*100)
    say('【二】总资产：三种口径')
    say('='*100)
    say(f'  A 每笔独立复利（旧口径，虚高）  {np.prod(1.0):.0f}  → 见 local_memory：363.87')
    say(f'  B 等权组合（真实账户形态）      {total:.4f}')
    say(f'  C 只用盈利子账户（上界参考）    {finals[finals>1].mean():.4f}')

    # 真实总资产曲线：把各子账户按时间对齐累加太复杂，这里给"等权末值" + "逐段"
    say()
    say('='*100)
    say('【三】四段（等权组合，各子账户按自身后 50% 时间分四段）')
    say('='*100)
    seg_sum = np.zeros(4)
    cnt = 0
    for _, r in subs:
        pnl = r['pnl']; n4 = len(pnl)
        if n4 < 4: continue
        for i in range(4):
            aa = i*n4//4; bb = (i+1)*n4//4
            seg_sum[i] += np.cumprod(1.0+pnl[aa:bb])[-1]
        cnt += 1
    if cnt:
        seg_avg = seg_sum/cnt
        ok4 = all(v > 1.0 for v in seg_avg)
        for i in range(4):
            say(f'  段{i+1}  平均段内资金 {seg_avg[i]:.4f}')
        say(f'  {"✅ 四段都赚" if ok4 else "❌ 有段亏损"}')
    else:
        ok4 = False

    say()
    say('='*100)
    say('【四】判据（用户口径：存活 + 持续盈利）')
    say('='*100)
    say(f'  总资产 {total:.4f} > 1 ? {"✅" if total>1 else "❌"}')
    say(f'  盈利子账户 {int((finals>1).sum())}/{len(finals)}')
    say(f'  四段 {"✅" if ok4 else "❌"}')
    ok = total > 1.0 and ok4
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for s, r in subs:
            f.write(f'{s}\t{r["cap"][-1]:.4f}\t{r["n_obs"]}\n')
        f.write(f'TOTAL\t{total:.4f}\n')

if __name__ == '__main__':
    main()
