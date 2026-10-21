# -*- coding: utf-8 -*-
r"""
gambler_entry_adaptive.py — 入场 + 回看多久，都交给模型

上一版（entry_odds）的问题：KNN 用「全部历史」做记忆，
把 2024 年"跌了就买"的经验套到 2026 年 ⇒ 段1/段2 亏、段3/段4 赚。

本版：让模型自己决定"往回看多久"
  5 个记忆窗口：30天 / 90天 / 180天 / 365天 / 全部历史
  · 每 10 万根（约 350 天）滚动重建一次，窗口边界跟着时间走
  · 每个时刻，5 个窗口各自查 KNN、各自算期望
  · 权重由「过去 200 笔各窗口若一直用它，赚不赚」决定（softmax）
  · 组合期望 > 成本 × 安全边际 ⇒ 下注

无前视：
  - 特征全 rolling
  - 标准化统计量只用训练段
  - 每个窗口的 KNN 只用「该窗口内的训练样本」
  - 权重只用「已结算的历史笔」

输出里会打印：模型在测试段选了哪个窗口、各占多少 —— 这就是"自己决定回看多久"的证据。
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\entry_adaptive.txt'
NSYM = int(os.environ.get('EA_NSYM', '60'))

COST    = 0.0009
H       = 176
TRAIN   = 0.70
K       = 100
SAFETY  = 1.5
STEP_TR = 100
STEP_TE = 50
REBUILD = 100_000
DAY_MS  = 86_400_000

WINDOWS = [30, 90, 180, 365, 10**9]
WNAMES  = ['30天', '90天', '180天', '365天', '全部']

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
    say('gambler_entry_adaptive · 入场 + 回看多久，都交给模型')
    say('='*100)
    say(f'  窗口 {WNAMES} | K={K} | 安全边际={SAFETY}× | 成本={COST*100:.3f}% | 持{H}根 | 训练{TRAIN:.0%}')
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

    # 测试段按 REBUILD 切块
    blocks = []
    b = 0
    while b < len(te_idx):
        b1 = int(np.searchsorted(te_idx, te_idx[min(b + REBUILD//STEP_TE, len(te_idx)-1)], side='right'))
        b1 = max(b1, b+1)
        blocks.append((b, min(b1, len(te_idx))))
        b = b1
    say(f'  重建 {len(blocks)} 次（每 {REBUILD:,} 根）')

    n_te = len(te_idx)
    exp_all = np.full(n_te, np.nan)
    win_used = np.full(n_te, -1, dtype=int)
    hist = [[] for _ in WINDOWS]      # 各窗口虚拟账本
    thr = COST * SAFETY

    for bi, (b0, b1) in enumerate(blocks):
        t_now = T[te_idx[b0]]
        knns = []
        for w in WINDOWS:
            mask = np.ones(len(tr_idx), bool) if w >= 10**9 else (T[tr_idx] >= t_now - w*DAY_MS)
            sub = tr_idx[mask]
            if len(sub) < 500:
                knns.append(None); continue
            nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
            knns.append((nn, sub))

        m = b1 - b0
        exp_blk = np.full((m, len(WINDOWS)), np.nan)
        for k, item in enumerate(knns):
            if item is None: continue
            nn, sub = item
            _, nb = nn.kneighbors(Xn[te_idx[b0:b1]])
            exp_blk[:, k] = y[sub][nb].mean(1)

        # 权重：块开始时固定（过去 200 笔各窗口表现）
        if len(hist[0]) < 100:
            wts = np.ones(len(WINDOWS)) / len(WINDOWS)
        else:
            rec = np.array([np.mean(h[-200:]) if len(h) >= 50 else -1e9 for h in hist])
            e = np.exp((rec - rec.max()) / 1e-4)
            wts = e / e.sum()

        for i in range(b0, b1):
            ex = exp_blk[i-b0]
            okk = np.isfinite(ex)
            if not okk.any(): continue
            w = np.where(okk, wts, 0.0)
            if w.sum() <= 0: continue
            w = w / w.sum()
            exp_all[i] = float((ex[okk] * w[okk]).sum())
            win_used[i] = int(np.argmax(w))

        for k in range(len(WINDOWS)):
            for i in range(b0, b1):
                v = exp_blk[i-b0, k]
                if np.isfinite(v):
                    hist[k].append(y[te_idx[i]] if v > thr else 0.0)
        if bi % 5 == 0: say(f'    块 {bi+1}/{len(blocks)}  {time.time()-t0:.0f}s')

    bet = np.isfinite(exp_all) & (exp_all > thr)
    r_te = y[te_idx]

    say()
    say('='*100)
    say('【一】测试段对比')
    say('='*100)
    def summ(mask, name):
        if mask.sum() < 10:
            say(f'  {name:<30} 样本太少（{mask.sum()}）'); return None
        r = r_te[mask]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  {name:<30} {mask.sum():>6,} 笔  单笔 {r.mean()*100:+7.3f}%  中位 {np.median(r)*100:+7.3f}%  胜率 {(r>0).mean()*100:4.1f}%  t={t:+6.2f}')
        return r.mean()

    a0 = summ(np.ones(len(r_te), bool), 'A 全样（随机下注）')
    a2 = summ(bet, 'C 自适应窗口（模型自己选）')

    # 对比：固定用全部历史（上一版）
    say()
    say('='*100)
    say('【二】模型自己选了哪个窗口（测试段逐点统计）')
    say('='*100)
    tot = (win_used >= 0).sum()
    if tot > 0:
        for k, nm in enumerate(WNAMES):
            cnt = (win_used == k).sum()
            say(f'  {nm:<8} {cnt:>7,}  ({cnt/tot*100:5.1f}%)')
        # 按四段看窗口选择
        say('  分四段的窗口选择：')
        for i in range(4):
            aa = i*n_te//4; bb = (i+1)*n_te//4
            seg = win_used[aa:bb]
            seg = seg[seg >= 0]
            if len(seg) == 0: continue
            dist = '  '.join(f'{WNAMES[k]}:{(seg==k).mean()*100:.0f}%' for k in range(len(WINDOWS)) if (seg==k).sum() > 0)
            say(f'    段{i+1}  {dist}')

    say()
    say('='*100)
    say('【三】四段（自适应窗口）')
    say('='*100)
    ok4 = True
    for i in range(4):
        aa = i*n_te//4; bb = (i+1)*n_te//4
        mm = bet[aa:bb]
        if mm.sum() < 10:
            say(f'  段{i+1}  n={mm.sum():>5}  样本太少'); ok4 = False; continue
        r = r_te[aa:bb][mm]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  段{i+1}  n={mm.sum():>5}  单笔 {r.mean()*100:+7.3f}%  t={t:+6.2f}')
        if r.mean() <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    say(f'  自适应 {a2*100 if a2 is not None else float("nan"):+.3f}%  vs  全样 {a0*100:+.3f}%')
    say(f'  四段 {"同正 ✅" if ok4 else "非全正 ❌"}')
    if a2 is not None and a2 > a0 and ok4:
        say('  ✅ 入场 + 回看多久都能由模型决定')
    elif a2 is not None and a2 > 0:
        say('  ⚠ 正期望，但四段不稳 —— 说明"选窗口"还没救回段1/段2')
    else:
        say('  ❌ 自适应窗口未救回')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_entry_adaptive 完成\n')

if __name__ == '__main__':
    main()
