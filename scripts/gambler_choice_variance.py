# -*- coding: utf-8 -*-
"""gambler_choice_variance · 选择可塑性诊断

问题：同样的「过去」，换一个「选择」，结果差多少？
  · 过去 = 状态向量（多尺度动量 + 波动 + 量比 + 区间位置），只用 t 及之前
  · 选择 = 5 种持有期 × 2 方向（买/卖），入场 t+1 开盘，出场 t+1+H 收盘
  · 结果 = 方向 × log(出场/入场) − maker 往返 0.04%
  · 诊断 = 两因素方差分解：选择效应 vs 状态效应 vs 交互
  · 无前视：状态只用过去；KMeans 只用训练段拟合；测试段分配
  · 对照：--shuffle 按天块重排 log 收益

判读：
  选择效应占比大 → 「选择」重要，三元组有东西可学
  状态效应占比大 → 「过去」决定结果，退化成两元组（已失败）
  两者都小        → 纯噪声

用法：
  python -X utf8 gambler_choice_variance.py
  python -X utf8 gambler_choice_variance.py --shuffle
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('GV_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('GV_NSYM', '60'))
BPD = int(os.environ.get('GV_BPD', '288'))
STEP = int(os.environ.get('GV_STEP', '12'))     # 决策间隔（12 根 = 60 分钟，避免重叠）
K = int(os.environ.get('GV_K', '30'))           # 状态桶数
COST = 0.0004                                   # maker 往返
SHUF = '--shuffle' in sys.argv
TAG = 'SHUF' if SHUF else 'REAL'
HOLDS = [3, 6, 12, 24, 48]
DIRS = [+1, -1]
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def shuffle_panel(C, rng):
    T, S = C.shape
    r = np.zeros_like(C)
    ok = np.isfinite(C[1:]) & np.isfinite(C[:-1]) & (C[1:] > 0) & (C[:-1] > 0)
    r[1:][ok] = np.log(C[1:][ok] / C[:-1][ok])
    r = np.nan_to_num(r)
    nb = T // BPD
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * BPD, (b + 1) * BPD) for b in order]
                         + [np.arange(nb * BPD, T)]).astype(int)
    v0 = C[0][np.isfinite(C[0])]
    c0 = np.full(S, v0.mean() if len(v0) else 1.0)
    return np.exp(np.log(np.maximum(c0, 1e-12)) + np.cumsum(r[idx], axis=0))


def kmeans_fit(X, K, iters=25, seed=20261020):
    rng = np.random.default_rng(seed)
    if len(X) > 40000:
        X = X[rng.choice(len(X), 40000, replace=False)]
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            m = lab == k
            if m.sum():
                C[k] = X[m].mean(0)
    return C


def kmeans_assign(X, C, chunk=50000):
    lab = np.empty(len(X), np.int32)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab[i:i + chunk] = d.argmin(1)
    return lab


def main():
    t0 = time.time()
    say('=' * 104)
    say(f'gambler_choice_variance · 同样的过去 × 换一个选择 × 结果差多少 [{TAG}]')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  动作 {len(HOLDS)}×{len(DIRS)}={len(HOLDS)*len(DIRS)}  成本 maker {COST:.2%}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close', 'volume'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    O = df.pivot_table(index='t', columns='symbol', values='open').sort_index()[syms].values.astype(np.float64)
    C = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms].values.astype(np.float64)
    V = df.pivot_table(index='t', columns='symbol', values='volume').sort_index()[syms].values.astype(np.float64)
    O[~np.isfinite(O) | (O <= 0)] = np.nan
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    tidx = df.pivot_table(index='t', columns='symbol', values='close').sort_index().index
    rng = np.random.default_rng(20261020)
    if SHUF:
        C = shuffle_panel(C, rng)
        O = C.copy()
        say('  ⚠ 已打乱：按天块重排 log 收益（总漂移不变）')
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    # ---- 状态特征（只用 t 及之前） ----
    feats = []
    for k in (1, 3, 6, 12, 24, 48):
        m = np.full((T, S), np.nan)
        m[k:] = C[k:] / C[:-k] - 1.0
        feats.append(np.nan_to_num(m))
    lr = np.zeros((T, S))
    ok = np.isfinite(C[1:]) & np.isfinite(C[:-1]) & (C[1:] > 0) & (C[:-1] > 0)
    lr[1:][ok] = np.log(C[1:][ok] / C[:-1][ok])
    vol = pd.DataFrame(lr).rolling(12, min_periods=6).std().values
    feats.append(np.nan_to_num(vol))
    vr = pd.DataFrame(V).rolling(12, min_periods=6).mean().values / \
         np.maximum(pd.DataFrame(V).rolling(288, min_periods=48).mean().values, 1e-12)
    feats.append(np.nan_to_num(vr))
    hi = pd.DataFrame(C).rolling(288, min_periods=48).max().values
    lo = pd.DataFrame(C).rolling(288, min_periods=48).min().values
    pos = (C - lo) / np.maximum(hi - lo, 1e-12)
    feats.append(np.nan_to_num(pos))
    say(f'  状态 {len(feats)} 维算完  {time.time()-t0:.0f}s')

    # ---- 决策点 + 结果（10 个动作） ----
    acts = [(h, d) for h in HOLDS for d in DIRS]
    starts = np.arange(BPD, T - 1 - max(HOLDS) - 1, STEP)
    rows_t, rows_s, Xs, Rs = [], [], [], []
    for s in range(S):
        c = C[:, s]
        o = O[:, s]
        for t in starts:
            if not (np.isfinite(o[t + 1]) and np.isfinite(c[t + 1 + max(HOLDS)])):
                continue
            r = []
            for h, d in acts:
                p_in = o[t + 1]
                p_out = c[t + 1 + h]
                if not (np.isfinite(p_in) and np.isfinite(p_out) and p_in > 0 and p_out > 0):
                    r.append(np.nan)
                else:
                    r.append(d * np.log(p_out / p_in) - COST)
            Rs.append(r)
            Xs.append([f[t, s] for f in feats])
            rows_t.append(t)
            rows_s.append(s)
    Rs = np.asarray(Rs, np.float64)
    Xs = np.asarray(Xs, np.float64)
    rows_t = np.asarray(rows_t)
    rows_s = np.asarray(rows_s)
    n = len(Rs)
    say(f'  决策点 {n:,}  {time.time()-t0:.0f}s')

    # ---- 训练/测试切分（按时间） ----
    split = int(n * 0.7)
    tr = np.arange(split)
    te = np.arange(split, n)

    # 标准化：训练段均值/标准差
    mu = np.nanmean(Xs[tr], 0)
    sd = np.nanstd(Xs[tr], 0) + 1e-12
    Z = (Xs - mu) / sd
    Z = np.clip(np.nan_to_num(Z), -5, 5)

    # KMeans：训练段拟合，测试段分配
    Cc = kmeans_fit(Z[tr], K)
    lab_tr = kmeans_assign(Z[tr], Cc)
    lab_te = kmeans_assign(Z[te], Cc)
    say(f'  KMeans {K} 桶：训练段分布 {np.bincount(lab_tr, minlength=K).min()}~{np.bincount(lab_tr, minlength=K).max()}  {time.time()-t0:.0f}s')

    def decomp(lab, R, name):
        B = K
        A = len(acts)
        M = np.full((B, A), np.nan)
        cnt = np.zeros((B, A), int)
        for b in range(B):
            m = lab == b
            if m.sum() < 30:
                continue
            sub = R[m]
            for a in range(A):
                v = sub[:, a]
                v = v[np.isfinite(v)]
                cnt[b, a] = len(v)
                if len(v) >= 30:
                    M[b, a] = v.mean()
        okb = np.isfinite(M).all(1)
        Mo = M[okb]
        if len(Mo) < 5:
            say(f'  [{name}] 可用桶不足')
            return
        grand = Mo.mean()
        rowm = Mo.mean(1); colm = Mo.mean(0)
        SS_tot = ((Mo - grand) ** 2).sum()
        SS_a = Mo.shape[0] * ((colm - grand) ** 2).sum()
        SS_b = Mo.shape[1] * ((rowm - grand) ** 2).sum()
        SS_e = max(SS_tot - SS_a - SS_b, 0.0)
        say(f'  [{name}] 可用桶 {len(Mo)}  总平方和 {SS_tot:.6f}')
        say(f'     选择效应 {SS_a/SS_tot:6.1%}   状态效应 {SS_b/SS_tot:6.1%}   交互 {SS_e/SS_tot:6.1%}')
        spread = (Mo.max(1) - Mo.min(1))
        say(f'     桶内「换选择」极差：均值 {spread.mean():.4%}  中位 {np.median(spread):.4%}')
        best = Mo.argmax(1)
        say(f'     最优动作分布：' + '；'.join(
            f'{acts[a][0]}根{"买" if acts[a][1]>0 else "卖"}×{int((best==a).sum())}'
            for a in range(len(acts)) if (best == a).sum() > 0))
        # 各动作平均
        say('     各动作平均净收益：' + '  '.join(
            f'{acts[a][0]}{"买" if acts[a][1]>0 else "卖"}={colm[a]:+.4%}' for a in range(len(acts))))

    say('\n【一】方差分解（桶均值）')
    decomp(lab_tr, Rs[tr], '训练段')
    decomp(lab_te, Rs[te], '测试段')

    # ---- 分季度：选择极差 + 各动作均值 ----
    say('\n【二】按自然季度（测试段）')
    qt_all = pd.PeriodIndex(pd.to_datetime(tidx[rows_t]), freq='Q')
    te_idx = np.flatnonzero(np.arange(n) >= split)
    for q in sorted(set(qt_all[te_idx])):
        sel = te_idx[qt_all[te_idx] == q]
        if len(sel) < 200:
            continue
        Rq = Rs[sel]
        labq = lab_te[sel - split]
        # 桶内极差（只算桶内样本 >= 30 的桶）
        sp = []
        for b in range(K):
            mb = labq == b
            if mb.sum() < 30:
                continue
            sub = Rq[mb]
            ma = np.array([np.nanmean(sub[:, a]) for a in range(len(acts))])
            if np.isfinite(ma).all():
                sp.append(ma.max() - ma.min())
        spread = np.mean(sp) if sp else np.nan
        say(f'  {q}  n={len(sel):,}  桶内换选择极差 {spread:+.4%}  ' +
            ' '.join(f'{acts[a][0]}{"买" if acts[a][1]>0 else "卖"}={np.nanmean(Rq[:,a]):+.4%}' for a in range(len(acts))))

    # ---- 总体各动作 ----
    say('\n【三】总体各动作（全部决策点）')
    for a, (h, d) in enumerate(acts):
        v = Rs[:, a][np.isfinite(Rs[:, a])]
        say(f'  持有{h:>2}根 {"买" if d>0 else "卖"}  n={len(v):,}  均值 {v.mean():+.4%}  中位 {np.median(v):+.4%}  胜率 {(v>0).mean():.1%}')

    say('\n【判据】选择效应占比 vs 状态效应占比 —— 由账本客观给出，不设门槛')
    say(f'TOTAL {time.time()-t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'gambler_choice_variance_{TAG}.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT) + '\n')
    print(f'\n已写 {p}')


if __name__ == '__main__':
    main()
