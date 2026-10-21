# -*- coding: utf-8 -*-
r"""
gambler_entry_odds.py — 入场由模型决定（赌徒方式：算赔率，不预测）

之前错误的做法：把入场做成"二分类预测（未来 176 根涨不涨）"→ AUC 0.501
赌徒的做法：
  贴近当下 → 找历史上状态最像的 K 个时刻
  → 看它们后来赚多少（mean_up）、亏多少（mean_dn）、赚的概率（p_up）
  → 算净期望 = 邻居真实收益均值
  → 期望 > 成本 × 安全边际 ⇒ 下注（否则弃牌）
  → 仓位 = 优势 / 波动² （凯利简化），上限 2%

无前视：
  - 特征全 rolling（只用过去）
  - 标准化统计量只用训练段
  - KNN 邻居只从训练段取
  - 测试段回放时才用决策

对比：
  A 全样（随机下注）—— 基线
  B 硬编码"近 5 根跌 5%" —— 之前的简单规则
  C 赌徒（期望 > 1.5×成本）—— 本次要证明的东西
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\entry_odds.txt'
NSYM = int(os.environ.get('EO_NSYM', '60'))

COST     = 0.0009        # 往返成本 0.09%
H        = 176           # 持有根数（与 v4 对齐）
TRAIN    = 0.70          # 训练段比例（按时间切）
K        = 100           # 近邻数
SAFETY   = 1.5           # 安全边际：期望要 > 1.5 × 成本才下注
STEP_TR  = 100           # 训练段采样步长（降 KNN 规模）
STEP_TE  = 50            # 测试段采样步长

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
    for k in (5, 15, 60, 240, 1440):
        F.append(np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1.0]))
    vma = rmean(v, 60)
    F.append(v/np.where(vma > 0, vma, np.nan))
    F.append(np.log1p(v))
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
    say('gambler_entry_odds · 入场由模型决定（赌徒方式：算赔率，不预测）')
    say('='*100)
    say(f'  K={K}  安全边际={SAFETY}×  成本={COST*100:.3f}%  持有={H}根  训练段={TRAIN:.0%}')
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
    o = np.argsort(T, kind='stable')
    X = X[o]; y = y[o]; T = T[o]
    say(f'  样本 {len(X):,}  正 {y.mean()*100:.1f}%  {time.time()-t0:.0f}s')

    cut = int(len(X)*TRAIN)
    mu = X[:cut].mean(0); sd = X[:cut].std(0) + 1e-9
    Xn = (X - mu) / sd

    tr_idx = np.arange(0, cut, STEP_TR)
    te_idx = np.arange(cut, len(X), STEP_TE)
    say(f'  训练 {len(tr_idx):,}  测试 {len(te_idx):,}')

    say('  建 KNN（只用训练段）...')
    nn = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(Xn[tr_idx])
    say(f'  建完 {time.time()-t0:.0f}s')

    say('  测试段逐点找近邻、算赔率 ...')
    dist, nbr = nn.kneighbors(Xn[te_idx])
    nb_ret = y[tr_idx][nbr]                     # (n_te, K)

    # 赔率三件：p_up / 赚时均值 / 亏时均值（向量化）
    up = nb_ret > 0
    p_up  = up.mean(1)
    c_up  = up.sum(1)
    m_up  = np.where(c_up > 0, (nb_ret*up).sum(1)/np.maximum(c_up,1), 0.0)
    dn    = nb_ret < 0
    c_dn  = dn.sum(1)
    m_dn  = np.where(c_dn > 0, (-nb_ret*dn).sum(1)/np.maximum(c_dn,1), 0.0)

    exp_ret = nb_ret.mean(1)                    # 净期望（已扣成本）
    var_ret = nb_ret.var(1) + 1e-10

    thr = COST * SAFETY
    bet = exp_ret > thr
    # 凯利简化：仓位 = 优势 / 方差 × 0.25，上限 2%
    pos = np.where(bet, np.clip(exp_ret/np.maximum(var_ret,1e-8)*0.25, 0, 0.02), 0.0)

    r5 = X[te_idx][:, 0]
    hard = r5 < -0.05
    say(f'  下注 {bet.sum():,}/{len(bet):,} ({bet.mean()*100:.1f}%)   硬编码跌5% {hard.sum():,} 笔  {time.time()-t0:.0f}s')

    r_te = y[te_idx]
    def summ(mask, name):
        if mask.sum() < 10:
            say(f'  {name:<28} 样本太少（{mask.sum()}）'); return None
        r = r_te[mask]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  {name:<28} {mask.sum():>6,} 笔  单笔 {r.mean()*100:+7.3f}%  中位 {np.median(r)*100:+7.3f}%  胜率 {(r>0).mean()*100:4.1f}%  t={t:+6.2f}')
        return r.mean()

    say()
    say('='*100)
    say('【一】测试段对比')
    say('='*100)
    a0 = summ(np.ones(len(r_te), bool), 'A 全样（随机下注）')
    a1 = summ(hard, 'B 硬编码 近5根跌5%')
    a2 = summ(bet, 'C 赌徒（期望>1.5×成本）')

    say()
    say('='*100)
    say('【二】赌徒自己决定压多重（仓位分布）')
    say('='*100)
    if bet.sum() > 0:
        pz = pos[bet]
        say(f'  下注笔数 {bet.sum():,}   平均仓位 {pz.mean()*100:.2f}%   中位 {np.median(pz)*100:.2f}%   最大 {pz.max()*100:.2f}%')
        say(f'  仓位 0~0.5%: {(pz<0.005).sum():,}   0.5~1%: {((pz>=0.005)&(pz<0.01)).sum():,}   1~2%: {(pz>=0.01).sum():,}')

    say()
    say('='*100)
    say('【三】四段（赌徒）')
    say('='*100)
    n = len(r_te); ok4 = True
    for i in range(4):
        a = i*n//4; b = (i+1)*n//4
        m = bet[a:b]
        if m.sum() < 10:
            say(f'  段{i+1}  n={m.sum():>5}  样本太少'); ok4 = False; continue
        r = r_te[a:b][m]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  段{i+1}  n={m.sum():>5}  单笔 {r.mean()*100:+7.3f}%  t={t:+6.2f}')
        if r.mean() <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    if a2 is None:
        say('  ❌ 赌徒没下注 ⇒ 这个距离度量下任何状态的期望都没超过 1.5×成本')
    elif a1 is None:
        say(f'  C 赌徒 {a2*100:+.3f}%（B 硬编码无样本）；四段{"同正✅" if ok4 else "非全正❌"}')
    else:
        say(f'  C 赌徒 {a2*100:+.3f}%  vs  B 硬编码 {a1*100:+.3f}%  （差 {(a2-a1)*100:+.3f}%）')
        say(f'  四段 {"同正 ✅" if ok4 else "非全正 ❌"}')
        if a2 > a1 and ok4:
            say('  ✅ 入场确实能由模型决定（赌徒方式：算赔率，不预测）')
        elif a2 > 0 and ok4:
            say('  ⚠ 赌徒方式为正且四段同正，但不如硬编码 —— 可继续加特征')
        else:
            say('  ❌ 这个距离度量下入场仍学不出来')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_entry_odds 完成\n')

if __name__ == '__main__':
    main()
