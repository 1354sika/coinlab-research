# -*- coding: utf-8 -*-
r"""
gambler_entry_drift.py — 记忆窗口由「市场节奏」决定，不由「过去哪个窗口赚得多」决定

上一版（entry_adaptive）错在哪：
  用"过去 200 笔各窗口的收益"选窗口 —— 这是**后视**，而且慢
  （市场变了要 200 笔才反应过来）⇒ 结果更差 -0.163%

本版按赌徒的直觉：**看多久，取决于节奏快慢**
  · 波动大（节奏快）⇒ 旧数据很快失效 ⇒ 窗口短
  · 波动小（节奏慢）⇒ 数据稳 ⇒ 窗口长
  · 窗口天数 = 30 × (历史中位波动 / 当前波动)，夹在 [7, 365]
  · 极端波动（σ > 历史 90 分位）⇒ 收手不下注（怕）

窗口是**连续值**，不是 5 档 —— 这就是"粒度自由"。
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\entry_drift.txt'
NSYM = int(os.environ.get('ED_NSYM', '60'))

COST    = 0.0009
H       = 176
TRAIN   = 0.70
K_NEIGH = 400
SAFETY  = 1.5
STEP_TR = 100
STEP_TE = 50
DAY_MS  = 86_400_000
SIG_IDX = 8          # sig 是第 9 列
W_BASE  = 30.0       # 基准窗口天数
W_MIN, W_MAX = 7.0, 365.0

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
    say('gambler_entry_drift · 记忆窗口由市场节奏决定（波动大→看短，波动小→看长）')
    say('='*100)
    say(f'  基准窗口 {W_BASE:.0f}天  夹在 [{W_MIN:.0f}, {W_MAX:.0f}] | K={K_NEIGH} | 安全边际={SAFETY}× | 成本={COST*100:.3f}%')
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

    cut = int(len(X)*TRAIN)
    mu = X[:cut].mean(0); sd = X[:cut].std(0) + 1e-9
    Xn = (X - mu) / sd

    tr_idx = np.arange(0, cut, STEP_TR)
    te_idx = np.arange(cut, len(X), STEP_TE)
    say(f'  训练 {len(tr_idx):,}  测试 {len(te_idx):,}')

    sig_tr = X[tr_idx][:, SIG_IDX]
    sig_med = float(np.median(sig_tr))
    sig_p90 = float(np.percentile(sig_tr, 90))
    say(f'  训练段波动 中位 {sig_med*100:.3f}%  P90 {sig_p90*100:.3f}%')

    say('  建 KNN（只用训练段）...')
    nn = NearestNeighbors(n_neighbors=K_NEIGH, algorithm='kd_tree').fit(Xn[tr_idx])
    _, nb = nn.kneighbors(Xn[te_idx])
    nb_ret  = y[tr_idx][nb]
    nb_time = T[tr_idx][nb].astype(np.int64)
    say(f'  查完近邻 {time.time()-t0:.0f}s')

    t_te  = T[te_idx].astype(np.int64)
    sig_te = X[te_idx][:, SIG_IDX]
    win_days = np.clip(W_BASE * sig_med / np.maximum(sig_te, 1e-10), W_MIN, W_MAX)

    say('  按窗口过滤邻居 ...')
    age = t_te[:, None] - nb_time
    m   = age <= (win_days[:, None] * DAY_MS)
    cnt = m.sum(1)
    exp_all = np.where(cnt >= 20, (nb_ret * m).sum(1) / np.maximum(cnt, 1), np.nan)
    say(f'  过滤完 {time.time()-t0:.0f}s')

    thr = COST * SAFETY
    pause = sig_te > sig_p90
    bet = np.isfinite(exp_all) & (exp_all > thr) & (~pause)
    r_te = y[te_idx]

    say()
    say('='*100)
    say('【一】模型自己看多久（窗口天数分布）')
    say('='*100)
    say('  ' + '   '.join(f'P{q}:{np.percentile(win_days, q):.1f}天' for q in (10, 25, 50, 75, 90)))
    say(f'  窗口 <15天 {(win_days<15).mean()*100:5.1f}%  15~60天 {((win_days>=15)&(win_days<60)).mean()*100:5.1f}%  60~180天 {((win_days>=60)&(win_days<180)).mean()*100:5.1f}%  >180天 {(win_days>=180).mean()*100:5.1f}%')
    say(f'  极端波动收手（σ>P90）占比 {pause.mean()*100:.1f}%')

    say()
    say('='*100)
    say('【二】测试段对比')
    say('='*100)
    def summ(mask, name):
        if mask.sum() < 10:
            say(f'  {name:<30} 样本太少（{mask.sum()}）'); return None
        r = r_te[mask]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  {name:<30} {mask.sum():>6,} 笔  单笔 {r.mean()*100:+7.3f}%  中位 {np.median(r)*100:+7.3f}%  胜率 {(r>0).mean()*100:4.1f}%  t={t:+6.2f}')
        return r.mean()

    a0 = summ(np.ones(len(r_te), bool), 'A 全样（随机下注）')
    a1 = summ(bet & (~pause), 'C 节奏自适应窗口')
    a2 = summ(bet & pause,  '（对照）极端波动时强行下注')

    say()
    say('='*100)
    say('【三】四段（节奏自适应）')
    say('='*100)
    n = len(r_te); ok4 = True
    for i in range(4):
        aa = i*n//4; bb = (i+1)*n//4
        mm = bet[aa:bb]
        wd = np.median(win_days[aa:bb])
        if mm.sum() < 10:
            say(f'  段{i+1}  n={mm.sum():>5}  样本太少'); ok4 = False; continue
        r = r_te[aa:bb][mm]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  段{i+1}  n={mm.sum():>5}  窗口中位 {wd:5.1f}天  单笔 {r.mean()*100:+7.3f}%  t={t:+6.2f}')
        if r.mean() <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    say(f'  节奏自适应 {a1*100 if a1 is not None else float("nan"):+.3f}%  vs  全样 {a0*100:+.3f}%')
    say(f'  四段 {"同正 ✅" if ok4 else "非全正 ❌"}')
    if a1 is not None and a1 > a0 and ok4:
        say('  ✅ 记忆窗口由节奏决定这条路成立')
    elif a1 is not None and a1 > 0:
        say('  ⚠ 正期望但四段不稳')
    else:
        say('  ❌ 仍未成立')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_entry_drift 完成\n')

if __name__ == '__main__':
    main()
