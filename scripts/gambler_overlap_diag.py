# -*- coding: utf-8 -*-
r"""
gambler_overlap_diag.py — 查清「363 倍 → 1.0」的机制

现象：
  local_memory（重叠口径）  仓位加权单笔 +0.847%，期末 363.87
  full2 / cross（不重叠）   仓位加权单笔 -0.043%，期末 0.9975

同一个 K=5、同一份数据、同一个公式，差 0.89pp。
唯一区别：**重叠 vs 不重叠**。

本诊断要回答：是「重叠放大了收益」，还是「保留策略选错了那一笔」？

做法（每币）：
  1. 算出所有点的 exp / var / R（K=5 局部记忆）
  2. 三种口径：
     A 全部触发（不限重叠）—— 就是 local_memory 的口径
     B 不重叠，保留第一个触发（next_free）
     C 不重叠，但保留「持仓期内 exp 最高」的那一个（事后最优，仅作上界）
  3. 比较 R 的分布、仓位加权收益、复利末值

无前视：C 是事后上界，只用来说明「有没有可改进空间」，不作结论。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\overlap_diag.txt'
NSYM = int(os.environ.get('OD_NSYM', '60'))

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

def collect(g):
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
    e_l, v_l, r_l = [], [], []
    for i in range(NBLOCK):
        tr_end, te_s, te_e = edges[i], edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        e_l.append(r.mean(1)); v_l.append(r.var(1)+1e-10); r_l.append(y[te_idx])
    if not e_l: return None
    return np.concatenate(e_l), np.concatenate(v_l), np.concatenate(r_l)

def sim(exp, var, R, mode):
    raw = np.where(exp > 0, np.clip(exp/np.maximum(var, 1e-10)*KELLY, 0, POS_MAX), 0.0)
    P = np.zeros_like(raw)
    pause = 0; gh = deque(maxlen=40)
    nxt = 0; n = len(raw)
    if mode == 'A':                       # 全部触发（重叠，local_memory 口径）
        for k in range(n):
            P[k] = raw[k]
    else:
        k = 0
        while k < n:
            if k < nxt or pause > 0:
                if pause > 0: pause -= 1
                k += 1; continue
            if raw[k] <= 0:
                k += 1; continue
            if mode == 'C':               # 事后最优：持仓期内挑 exp 最大的
                end = min(k+H_STEPS, n)
                seg = raw[k:end]
                if len(seg) and seg.max() > 0:
                    k = k + int(np.argmax(seg))
            P[k] = raw[k]
            nxt = k + H_STEPS
            gh.append(raw[k]*R[k])
            if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
                pause = STOP_BARS
            k += 1
    pnl = P*R
    cap = np.cumprod(1.0+pnl)
    w = P.sum()
    return dict(cap=cap[-1], trades=int((P>0).sum()), wavg=(P*R).sum()/max(w,1e-9),
                pmean=P.mean(), rmean=float(R[P>0].mean()) if (P>0).any() else 0.0)

def main():
    say('='*100)
    say('gambler_overlap_diag · 363 倍 → 1.0 的机制')
    say('='*100)
    say('  A 全部触发（重叠）｜B 不重叠保留第一个｜C 不重叠但事后挑 exp 最高的（上界）')
    say('='*100)

    df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(NSYM).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    say(f'  载入 {len(df):,} 行  {time.time()-t0:.0f}s')

    agg = {m: dict(caps=[], trades=0, wavg=[], pmean=[], rmean=[]) for m in 'ABC'}
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        r = collect(g)
        if r is None: continue
        exp, var, R = r
        for m in 'ABC':
            o = sim(exp, var, R, m)
            agg[m]['caps'].append(o['cap']); agg[m]['trades'] += o['trades']
            agg[m]['wavg'].append(o['wavg']); agg[m]['pmean'].append(o['pmean'])
            agg[m]['rmean'].append(o['rmean'])
    say(f'  子账户 {len(agg["A"]["caps"])} 个  {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【一】三种口径（60 币等权）')
    say('='*100)
    names = {'A':'A 全部触发（重叠）', 'B':'B 不重叠·保留第一个', 'C':'C 不重叠·事后最优（上界）'}
    for m in 'ABC':
        a = agg[m]
        total = float(np.mean(a['caps']))
        say(f'  {names[m]:<28} 期末 {total:>10.4f}  笔数 {a["trades"]:>7}  均仓 {np.mean(a["pmean"])*100:>5.2f}%')
        say(f'  {"":<28} 仓位加权单笔 {np.mean(a["wavg"])*100:>+7.3f}%   被选中笔的裸收益均值 {np.mean(a["rmean"])*100:>+7.3f}%')

    say()
    say('='*100)
    say('【二】判据')
    say('='*100)
    aTot = float(np.mean(agg['A']['caps'])); bTot = float(np.mean(agg['B']['caps'])); cTot = float(np.mean(agg['C']['caps']))
    say(f'  A��重叠）{aTot:.4f}  →  B（不重叠）{bTot:.4f}   ⇒ 重叠带来的虚增 {aTot/max(bTot,1e-9):.1f} 倍')
    say(f'  B（保留第一个）{bTot:.4f}  →  C（事后挑最好，上界）{cTot:.4f}')
    if cTot > bTot * 1.2:
        say('  ✅ 说明「保留哪一笔」影响巨大 —— 有可改进空间（但 C 用了未来信息，不可直接实现）')
    else:
        say('  ❌ 即使事后挑最好也不明显 —— 重叠只是因为「多开了几笔」，不是「选错了笔」')
    say(f'  被选中笔的裸收益：A {np.mean(agg["A"]["rmean"])*100:+.3f}%  B {np.mean(agg["B"]["rmean"])*100:+.3f}%  C {np.mean(agg["C"]["rmean"])*100:+.3f}%')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for m in 'ABC':
            f.write(f'{m}\t{np.mean(agg[m]["caps"]):.4f}\t{np.mean(agg[m]["wavg"]):.6f}\n')

if __name__ == '__main__':
    main()
