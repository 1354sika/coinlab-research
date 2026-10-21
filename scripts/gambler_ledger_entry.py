# -*- coding: utf-8 -*-
r"""
gambler_ledger_entry.py — 入场由「经验账本」决定（不是预测）

前六种入场全败，共同点：都在**预测**（算期望 → 下注）
  KNN / PCA / 分桶 / 分位 / 跌5% / 自适应窗口，全是"我算准了才下"

赌徒不预测，他**记账**：
  · 状态分 125 格（跌幅分位 × 波动分位 × 量比分位，各 5 档）
  · 每格记「最近 50 笔在该格下注的真实结果」（滑动窗口，新压旧）
  · 仓位 = 该格最近记录的 期望/方差 × 凯利系数，上限 2%
  · 该格经验不足 10 笔 ⇒ 仓位 0（不猜，等经验）
  · 结算延迟 H 根，绝不前视

这就是「失败就吸取教训」：教训存在账本里，不靠预测能力。
边测边学：账本随交易滚动，永远只看最近 50 笔。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\ledger_entry.txt'
NSYM = int(os.environ.get('LE_NSYM', '60'))

COST     = 0.0009
H        = 176
STEP_TE  = 50
DAY_MS   = 86_400_000
OBSERVE  = 0.50
NB       = 5                 # 每维 5 档
NCELL    = NB**3             # 125 格
K_HIST   = 50                # 每格记最近 50 笔
MIN_HIST = 10                # 经验不足 10 笔不下注
POS_MAX  = 0.02
KELLY    = 0.25
BAR_MS   = 300_000           # 5m

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
    r60 = np.concatenate([np.full(60, np.nan), c[60:]/c[:-60]-1.0])
    vma = rmean(v, 60)
    volr = v/np.where(vma > 0, vma, np.nan)
    sig = pd.Series(c).pct_change().rolling(240).std().values
    return np.column_stack([r60, volr, sig])

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def main():
    say('='*100)
    say('gambler_ledger_entry · 入场由经验账本决定（记账，不预测）')
    say('='*100)
    say(f'状态 {NB}³={NCELL}格 | 每格记最近 {K_HIST} 笔 | 经验<{MIN_HIST}笔不押 | 凯利×{KELLY} 上限{POS_MAX*100:.0f}%')
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
    say(f'  观察段结束 {pd.to_datetime(T[cut0], unit="ms")}')

    # 分位点（只用观察段，之后固定 —— 账本才是会变的部分）
    qs = []
    for j in range(3):
        q = np.percentile(X[:cut0, j], np.linspace(0, 100, NB+1))
        q[0], q[-1] = -np.inf, np.inf
        qs.append(q)

    te_all = np.arange(cut0, n, STEP_TE)
    cell_all = np.zeros(len(te_all), dtype=np.int32)
    for j in range(3):
        b = np.clip(np.searchsorted(qs[j], X[te_all, j], side='right')-1, 0, NB-1)
        cell_all = cell_all*NB + b

    ledger = [deque(maxlen=K_HIST) for _ in range(NCELL)]
    settle = deque()          # (到期时间戳, 格, 收益)
    pos_out = np.zeros(len(te_all))
    cnt_used = np.zeros(len(te_all), dtype=np.int32)
    first_bet = -1

    for k in range(len(te_all)):
        i = te_all[k]
        t = T[i]
        # 1) 结算到期的（绝无前视：只有 T[i] >= 结算时间才入账）
        while settle and settle[0][0] <= t:
            _, c_, r_ = settle.popleft()
            ledger[c_].append(r_)
        # 2) 决策
        c = cell_all[k]
        hist = ledger[c]
        pos_out[k] = 0.0
        cnt_used[k] = len(hist)
        if len(hist) >= MIN_HIST:
            a = np.fromiter(hist, dtype=np.float64, count=len(hist))
            e = a.mean()
            v = a.var() + 1e-10
            if e > 0:
                pos_out[k] = min(e/v*KELLY, POS_MAX)
                if first_bet < 0: first_bet = k
        # 3) 挂账（H 根后结算）
        settle.append((t + H*BAR_MS, c, y[i]))
        if k % 200000 == 0 and k > 0:
            say(f'    … {k}/{len(te_all)}  {time.time()-t0:.0f}s')

    R = y[te_all]
    P = pos_out
    say(f'  跑完，首次下注在第 {first_bet} 个点  {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【一】结果')
    say('='*100)
    m1 = R.mean(); t1 = m1/(R.std()/np.sqrt(len(R))+1e-12)
    say(f'  A 每根满仓（基准）      {len(R):>7,} 笔  单笔 {m1*100:+7.3f}%  t={t1:+6.2f}')
    w = P.sum()
    if w > 0:
        wavg = (P*R).sum()/w
        say(f'  B 账本定仓位            下注率 {(P>0).mean()*100:5.1f}%  均仓 {P.mean()*100:.2f}%')
        say(f'     仓位加权单笔 {wavg*100:+7.3f}%   对比满仓 {(wavg-m1)*100:+.3f}%')
    else:
        say('  B 全程仓位 0')

    say()
    say('='*100)
    say('【二】账本学到了什么')
    say('='*100)
    nz = [(c_, np.array(ledger[c_])) for c_ in range(NCELL) if len(ledger[c_]) >= MIN_HIST]
    say(f'  有经验的格子 {len(nz)}/{NCELL}   经验量中位 {np.median([len(v) for _,v in nz]) if nz else 0:.0f} 笔')
    if nz:
        exps = np.array([v.mean() for _,v in nz])
        say(f'  各格期望：P10 {np.percentile(exps,10)*100:+.3f}%  P50 {np.percentile(exps,50)*100:+.3f}%  P90 {np.percentile(exps,90)*100:+.3f}%')
        say(f'  期望>0 的格子 {int((exps>0).sum())}/{len(exps)}  期望>0.09%(成本线) 的 {int((exps>COST).sum())}')

    say()
    say('='*100)
    say('【三】四段（按仓位加权）')
    say('='*100)
    n4 = len(P); ok4 = True
    for k in range(4):
        aa = k*n4//4; bb = (k+1)*n4//4
        pp = P[aa:bb]; rr = R[aa:bb]
        ww = pp.sum()
        if ww <= 0: say(f'  段{k+1} 无仓位'); ok4 = False; continue
        av = (pp*rr).sum()/ww
        say(f'  段{k+1}  n={len(pp):>6,}  下注率 {(pp>0).mean()*100:5.1f}%  加权单笔 {av*100:+7.3f}%')
        if av <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    say('  · 若账本定仓位 > 满仓 ⇒ 记账本身有价值（经验能学出来）')
    say('  · 若 > 0 且四段同正 ⇒ 入场由账本决定这条路成立')
    if w > 0 and wavg > 0 and ok4:
        say('  ✅ 成立')
    elif w > 0 and wavg > 0:
        say('  ⚠ 正期望但四段不稳')
    else:
        say('  ❌ 仍未成立')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_ledger_entry 完成\n')

if __name__ == '__main__':
    main()
