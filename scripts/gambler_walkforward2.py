# -*- coding: utf-8 -*-
r"""
gambler_walkforward2.py — 变量由模型构造 + 边测边学

前几版的根子错误：用我拍的 9 个特征（r5/r15/r60/r240/r1440/volr/vabs/pos/sig）
  → 这组变量已证明无预测力（AUC 0.501，各种口径都试过）

本版把「变量长什么样」也交给模型：
  · 原始输入：15 个尺度的收益（1,2,3,5,8,13,21,34,55,89,144,233,377,610,987 根）
    + 量比 + 波动 + 区间位置（共 18 维原始量）
  · 模型自己学的投影：每块用「这块的记忆」拟合 PCA(10 维)
    —— 10 个维度不是人挑的，是模型在历史里学出来的「哪几种形态最值得区分」
  · KNN 在这 10 维空间里找相似（相似 = 模型学的相似）
  · 边测边学：30 块滚动前进，记忆窗口由波动决定

下注判据不变（期望 > 1.5×成本），**不靠"选少"来抬均值**。
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\walkforward2.txt'
NSYM = int(os.environ.get('W2_NSYM', '60'))

COST    = 0.0009
H       = 176
K       = 50
SAFETY  = 1.5
NCOMP   = 10
STEP_TR = 20
STEP_TE = 50
DAY_MS  = 86_400_000
W_BASE  = 30.0
W_MIN, W_MAX = 15.0, 365.0
NBLOCK  = 30
OBSERVE = 0.50

SCALES = [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987]

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def build(g):
    """原始量：15 尺度收益 + 量比 + 波动 + 区间位置 = 18 维"""
    c = g['close'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    F = []
    for k in SCALES:
        F.append(np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1.0]))
    vma = rmean(v, 60)
    F.append(v/np.where(vma > 0, vma, np.nan))
    hi = pd.Series(g['high'].values).rolling(240).max().values
    lo = pd.Series(g['low'].values).rolling(240).min().values
    rng = np.where(hi >= lo, hi-lo, np.nan)
    F.append(np.where(rng > 0, (c-lo)/rng, np.nan))
    F.append(pd.Series(c).pct_change().rolling(240).std().values)
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
    say('gambler_walkforward2 · 变量由模型构造（PCA 学投影）+ 边测边学')
    say('='*100)
    say(f'原始 18 维 → PCA {NCOMP} 维 | BLOCK {NBLOCK} | 观察 {OBSERVE:.0%} | K={K} | 成本={COST*100:.3f}%')
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
    say(f'  样本 {len(X):,}  原始 {X.shape[1]} 维  {time.time()-t0:.0f}s')

    n = len(X)
    cut0 = int(n*OBSERVE)
    sig_idx = X.shape[1] - 1
    sig_med = float(np.median(X[:cut0][:, sig_idx]))
    thr = COST * SAFETY
    say(f'  观察段结束于 {pd.to_datetime(T[cut0], unit="ms")}  波动中位 {sig_med*100:.3f}%')

    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    bet_r, bet_w, bet_i, bet_nc = [], [], [], []
    comp_used = []

    for i in range(NBLOCK):
        tr_end = edges[i]
        te_s, te_e = edges[i], edges[i+1]
        if te_e <= te_s: continue
        t_now = T[te_s]
        back = X[max(0,tr_end-200):tr_end, sig_idx]
        sig_now = float(np.median(back)) if len(back) else sig_med
        w_days = float(np.clip(W_BASE * sig_med / max(sig_now, 1e-10), W_MIN, W_MAX))

        age = t_now - T[:tr_end]
        m = (age <= w_days*DAY_MS) & (np.arange(tr_end) % STEP_TR == 0)
        sub = np.where(m)[0]
        if len(sub) < 200:
            say(f'    块 {i+1}/{NBLOCK}  记忆太少 ({len(sub)}) 跳过'); continue

        # —— 模型自己学的投影（只用这块的记忆）——
        Xs_sub = X[sub]
        mu = Xs_sub.mean(0); sd = Xs_sub.std(0) + 1e-9
        Zs_sub = (Xs_sub - mu) / sd
        pca = PCA(n_components=min(NCOMP, Zs_sub.shape[1]), svd_solver='randomized', random_state=0).fit(Zs_sub)
        Z_sub = pca.transform(Zs_sub)

        te_idx = np.arange(te_s, te_e, STEP_TE)
        Z_te = pca.transform((X[te_idx] - mu) / sd)

        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Z_sub)
        _, nb = nn.kneighbors(Z_te)
        r = y[sub][nb]
        exp = r.mean(1)
        b = exp > thr
        if b.sum() > 0:
            bet_r.append(y[te_idx][b])
            bet_w.append(np.full(int(b.sum()), w_days))
            bet_i.append(te_idx[b])
            bet_nc.append(np.full(int(b.sum()), len(sub)))
        comp_used.append(pca.explained_variance_ratio_[:NCOMP].sum())
        say(f'    块 {i+1:>2}/{NBLOCK}  窗 {w_days:5.1f}天  记忆 {len(sub):>5}  下注 {int(b.sum()):>4}/{len(te_idx):>4}  PCA解释 {comp_used[-1]*100:4.1f}%  {time.time()-t0:.0f}s')

    if not bet_r:
        say('\n  ❌ 全程无下注'); return

    rA = np.concatenate(bet_r); wA = np.concatenate(bet_w); iA = np.concatenate(bet_i)
    r_all = y[np.arange(cut0, n, STEP_TE)]

    say()
    say('='*100)
    say('【一】总体（后 50% 时间，边走边学，变量由模型学）')
    say('='*100)
    def summ(r, name):
        if len(r) < 10: say(f'  {name:<32} 样本太少'); return None
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  {name:<32} {len(r):>6,} 笔  单笔 {r.mean()*100:+7.3f}%  中位 {np.median(r)*100:+7.3f}%  胜率 {(r>0).mean()*100:4.1f}%  t={t:+6.2f}')
        return r.mean()

    a0 = summ(r_all, 'A 全样（随机下注）')
    a1 = summ(rA,    'C 变量由模型学 + 边测边学')
    say(f'  下注率 {len(rA)/len(r_all)*100:.1f}%   净优势 {(a1-a0)*100:+.3f}%/笔' if a1 is not None and a0 is not None else '')
    say(f'  PCA 平均解释方差 {np.mean(comp_used)*100:.1f}%')

    say()
    say('='*100)
    say('【二】模型自己看多久（窗口天数）')
    say('='*100)
    say('  ' + '   '.join(f'P{q}:{np.percentile(wA,q):.1f}天' for q in (10,25,50,75,90)))

    say()
    say('='*100)
    say('【三】四段')
    say('='*100)
    oq = np.argsort(iA); rS = rA[oq]
    n4 = len(rS); ok4 = True
    for k in range(4):
        aa = k*n4//4; bb = (k+1)*n4//4
        if bb - aa < 10: say(f'  段{k+1} 样本太少'); ok4 = False; continue
        r = rS[aa:bb]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  段{k+1}  n={len(r):>5}  单笔 {r.mean()*100:+7.3f}%  t={t:+6.2f}')
        if r.mean() <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    if a1 is not None and a0 is not None:
        say(f'  变量由模型学 {a1*100:+.3f}%  vs  全样 {a0*100:+.3f}%   差 {(a1-a0)*100:+.3f}%')
    say(f'  四段 {"同正 ✅" if ok4 else "非全正 ❌"}')
    if a1 is not None and a0 is not None and a1 > a0 and ok4:
        say('  ✅ 变量由模型构造这条路成立')
    elif a1 is not None and a1 > 0:
        say('  ⚠ 正期望但四段不稳')
    else:
        say('  ❌ 仍未成立')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_walkforward2 完成\n')

if __name__ == '__main__':
    main()
