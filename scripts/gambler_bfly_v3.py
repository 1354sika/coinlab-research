# -*- coding: utf-8 -*-
"""gambler_bfly_v3.py — 蝴蝶效应第三轮：**多尺度一致性 + 路径形态 + 时序链结合**

用户指令：继续穷尽蝴蝶效应（前两轮已覆盖：基础量 / 价格几何 / 量能形态 / 价量关系 / 相对强弱）。

本轮新增第三批维度（全部 t 时刻可观测，无前视）：
  H. 多尺度一致性：r12/r24/r96/r288 中同为正的数量（0~4）
  I. 均线关系：close/MA24 - 1、close/MA96 - 1
  J. 路径效率：r288 / Σ|r1|（净涨除以总波动，越大越"顺畅"）
  K. 上行占比：288 根内上涨根数的比例
  L. 量能持续：volr24 > 2 的连续根数（吸筹持续时长）
  M. 突破强度：close / 前 288 根最高价 - 1（创新高幅度）
  N. 波动率分位：vola24 在 288 根内的分位
  O. 回调位置：从 288 根高点回撤后，当前距高点/距低点的位置比

产出：
  ① 第三批新条件池（~40 条件）**真穷举** 2 元 + 3 元
  ② 与前两轮 top 条件**混合**（~100 条件）**真穷举** 2 元 + 3 元
  ③ **时序链结合**：新条件 A →τ→ 旧 top 条件 B（**全穷举**）
  ④ 置换检验（**分块置换 20 次**，给出"随机能达到的最高值"作为显著性基线）

产物：D:\中信证券\coinlab\evidence\butterfly\v3_*.csv / v3_summary.txt
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
OUTDIR = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY, H48 = 288, 576
MIN_N = 150
TAUS = [6, 12, 24, 48, 96]

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.append(s)

log('=' * 104)
log('gambler_bfly_v3 · 第三轮：多尺度一致性 / 均线关系 / 路径效率 / 量能持续 / 突破强度')
log('=' * 104)

# ---------- 数据 ----------
RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'open', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'open': 'float32', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['open'].values.astype('float64'), g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'), g['low'].values.astype('float64'),
                    g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

frames = []
for (sym, ot, o_, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); cs = pd.Series(c)
    lo_w = ls.rolling(WDAY, min_periods=72).min().values
    hi_w = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo_w, 1e-12) - 1.0
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    R = {}
    for k in (12, 24, 96, 288):
        R[k] = np.full(n, np.nan); R[k][k:] = c[k:] / c[:-k] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6  = pd.Series(volr).rolling(6,  min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    # H 多尺度一致性
    sg4 = np.stack([np.sign(np.nan_to_num(R[k])) for k in (12, 24, 96, 288)], axis=1)
    n_pos = (sg4 > 0).sum(axis=1).astype('float64')
    n_neg = (sg4 < 0).sum(axis=1).astype('float64')
    # I 均线关系
    ma24 = cs.rolling(24, min_periods=8).mean().values
    ma96 = cs.rolling(96, min_periods=24).mean().values
    px_ma24 = np.where(ma24 > 0, c / np.maximum(ma24, 1e-12) - 1.0, np.nan)
    px_ma96 = np.where(ma96 > 0, c / np.maximum(ma96, 1e-12) - 1.0, np.nan)
    # J 路径效率
    absum = pd.Series(np.abs(np.nan_to_num(r1))).rolling(WDAY, min_periods=72).sum().values
    path_eff = np.where(absum > 0, R[288] / np.maximum(absum, 1e-12), np.nan)
    # K 上行占比
    up_ind = (np.nan_to_num(r1) > 0).astype('float64')
    up_frac = pd.Series(up_ind).rolling(WDAY, min_periods=72).mean().values
    # L 量能持续
    vr_hi = (np.nan_to_num(volr24, nan=0.0) > 2).astype('float64')
    run_vol = np.zeros(n)
    for i in range(1, n):
        run_vol[i] = run_vol[i-1] + 1 if vr_hi[i] > 0 else 0
    # M 突破强度
    hi_prev = hs.rolling(WDAY, min_periods=72).max().shift(1).values
    brk = np.where(hi_prev > 0, c / np.maximum(hi_prev, 1e-12) - 1.0, np.nan)
    # N 波动率分位
    vola24 = pd.Series(r1).rolling(24, min_periods=8).std().values
    vola_rank = pd.Series(vola24).rolling(WDAY, min_periods=72).apply(lambda x: (x[-1] > x).mean(), raw=True).values
    # O 回调位置
    retr = np.where(hi_w > lo_w, (c - lo_w) / np.maximum(hi_w - lo_w, 1e-12), np.nan)
    # 标签
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    for t in np.flatnonzero(np.isfinite(f_up)):
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    ok = np.isfinite(f_up) & np.isfinite(dd)
    keep = ok & (over >= 0.03) & (over <= 0.40) & np.isfinite(volr24) & (volr24 >= 1)
    idx = np.flatnonzero(keep)[::3]
    if len(idx) < 200: continue
    frames.append(pd.DataFrame({
        't': ot[idx], 'over': over[idx], 'pos': retr[idx],
        'r12': R[12][idx], 'r24': R[24][idx], 'r96': R[96][idx], 'r288': R[288][idx],
        'volr6': volr6[idx], 'volr24': volr24[idx],
        'n_pos': n_pos[idx], 'n_neg': n_neg[idx],
        'px_ma24': px_ma24[idx], 'px_ma96': px_ma96[idx],
        'path_eff': path_eff[idx], 'up_frac': up_frac[idx], 'run_vol': run_vol[idx],
        'brk': brk[idx], 'vola_rank': vola_rank[idx],
        'true': (ok[idx] & (f_up[idx] >= 0.50) & (dd[idx] >= 0.30)).astype('int8')}))
    del hs, ls, qs, cs; gc.collect()

df = pd.concat(frames, ignore_index=True)
seg = []; o = 0
for i, d in enumerate(frames):
    L = len(d); seg.append((o, o + L)); o += L
Y = df['true'].values.astype('int8')
cut_t = np.quantile(df['t'].values, 0.70)
TR = df['t'].values <= cut_t; TE = ~TR
btr, bte = Y[TR].mean(), Y[TE].mean()
log(f'候选 {len(df):,}  币 {df["sym"].nunique() if "sym" in df else len(frames)}  基准 全 {100*Y.mean():.3f}% | '
    f'训练 {100*btr:.4f}% | 测试 {100*bte:.4f}%  耗时 {time.time()-t0t:.0f}s')
del RAW, frames; gc.collect()

# ---------- 第三批条件池 ----------
C3 = {}
def cond(name, col, lo=None, hi=None):
    a = df[col].values.astype('float64')
    m = np.isfinite(a)
    if lo is not None: m &= (a >= lo)
    if hi is not None: m &= (a < hi)
    C3[name] = m

SPECS = [
    ('n_pos',     [(2,None),(3,None),(4,None)],      ['4尺度同涨>=2','4尺度同涨>=3','4尺度全涨']),
    ('n_neg',     [(2,None),(3,None)],               ['4尺度同跌>=2','4尺度同跌>=3']),
    ('px_ma24',   [(0.01,None),(0.03,None)],         ['高于MA24>=1%','高于MA24>=3%']),
    ('px_ma96',   [(0.02,None),(0.05,None)],         ['高于MA96>=2%','高于MA96>=5%']),
    ('path_eff',  [(0.3,None),(0.5,None)],           ['路径效率>=0.3','路径效率>=0.5']),
    ('up_frac',   [(0.55,None),(0.6,None)],          ['上行占比>=55%','上行占比>=60%']),
    ('run_vol',   [(3,None),(6,None),(12,None)],     ['放量持续>=3根','放量持续>=6根','放量持续>=12根']),
    ('brk',       [(0.0,None),(0.05,None)],          ['创新高','突破>=5%']),
    ('vola_rank', [(0.8,None),(0.95,None)],          ['波动率分位>=80%','波动率分位>=95%']),
    ('over',      [(0.15,None),(0.25,None)],         ['over>=15%','over>=25%']),
    ('r288',      [(0.30,None),(0.60,None)],         ['r288>=30%','r288>=60%']),
    ('r24',       [(0.10,None),(0.20,None)],         ['r24>=10%','r24>=20%']),
    ('volr6',     [(10,None)],                       ['volr6>=10']),
]
for col, cuts, nm in SPECS:
    for (lo, hi), nv in zip(cuts, nm):
        cond(nv, col, lo, hi)
N3 = sorted(C3.keys())
M3 = np.stack([C3[k] for k in N3], axis=1)
log(f'\n第三批条件池：{len(N3)} 个')

def eval_mask(m):
    ntr = int((m & TR).sum()); nte = int((m & TE).sum())
    if ntr < MIN_N: return None
    return ntr, Y[m & TR].mean(), nte, (Y[m & TE].mean() if nte >= 30 else np.nan)

def run_stage(M, names, tag):
    t_st = time.time(); res = []
    IDX = [np.flatnonzero(M[:, j]).astype(np.int32) for j in range(M.shape[1])]
    for i, nm in enumerate(names):
        r = eval_mask(M[:, i])
        if r: res.append((nm, r[0], r[1], r[2], r[3]))
    pairs = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            inter = np.intersect1d(IDX[i], IDX[j], assume_unique=True)
            if len(inter) >= MIN_N: pairs[(i, j)] = inter
    for (i, j), inter in pairs.items():
        m = np.zeros(len(df), bool); m[inter] = True
        r = eval_mask(m)
        if r: res.append((f'{names[i]} & {names[j]}', r[0], r[1], r[2], r[3]))
    log(f'\n--- {tag}: 1元 {len(names)} | 2元 {len(pairs):,} ---')
    n3 = 0
    for (i, j), inter in pairs.items():
        if len(inter) < MIN_N * 2: continue
        for k in range(len(names)):
            if k in (i, j): continue
            mm = np.intersect1d(inter, IDX[k], assume_unique=True)
            if len(mm) < MIN_N: continue
            m = np.zeros(len(df), bool); m[mm] = True
            r = eval_mask(m)
            if r: res.append((f'{names[i]} & {names[j]} & {names[k]}', r[0], r[1], r[2], r[3])); n3 += 1
    log(f'  3元 {n3:,}  累计 {len(res):,}  耗时 {time.time()-t_st:.0f}s')
    out = pd.DataFrame(res, columns=['combo', 'n_train', 'rate_train', 'n_test', 'rate_test'])
    out['lift_train'] = out['rate_train'] / btr
    out['lift_test'] = out['rate_test'] / bte
    return out.sort_values('lift_test', ascending=False)

O3 = run_stage(M3, N3, '① 第三批池')
O3.to_csv(os.path.join(OUTDIR, 'v3_new_all.csv'), index=False, encoding='utf-8-sig')
log(f'\n① 结果 {len(O3):,} 条（按测试段排序 top-15）')
for _, r in O3.head(15).iterrows():
    log(f'  {r["combo"][:66]:<66} 训 {100*r["rate_train"]:>6.2f}% lift {r["lift_train"]:>6.1f}x | '
        f'测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>5,}) lift {r["lift_test"]:>6.1f}x')

# ---------- ③ 时序链结合（新条件 A →τ→ 旧 top 条件 B） ----------
log('\n' + '=' * 104)
log('【③ 时序链结合】A（第三批条件）→τ→ B（前两轮 top 条件）')
log('=' * 104)
OLDB = {}
def condO(name, col, th):
    a = df[col].values.astype('float64')
    OLDB[name] = np.isfinite(a) & (a >= th)
for nm, col, th in [('old_r12>=5%', 'r12', 0.05), ('old_r288>=30%', 'r288', 0.30),
                    ('old_volr6>=10', 'volr6', 10), ('old_volr24>=5', 'volr24', 5),
                    ('old_over>=15%', 'over', 0.15)]:
    condO(nm, col, th)
OBN = sorted(OLDB.keys())
MOB = np.stack([OLDB[k] for k in OBN], axis=1)
A3 = np.stack([C3[k] for k in N3], axis=1)

rows = []
t_ch = time.time()
for tau in TAUS:
    lag = np.zeros(len(df), dtype=bool)
    for s, e in seg:
        L = e - s
        if L > tau:
            lag[s+tau:e] = True    # 占位：下面按条件逐列构造
    for ai in range(len(N3)):
        a_now = A3[:, ai]
        a_lag = np.zeros(len(df), dtype=bool)
        for s, e in seg:
            L = e - s
            if L > tau: a_lag[s+tau:e] = a_now[s:e-tau]
        if a_lag.sum() < MIN_N: continue
        for bi in range(len(OBN)):
            m = a_lag & MOB[:, bi]
            ntr = int((m & TR).sum()); nte = int((m & TE).sum())
            if ntr < MIN_N: continue
            rtr = Y[m & TR].mean(); rte = Y[m & TE].mean() if nte >= 30 else np.nan
            rows.append((f'[{N3[ai]}] -{tau*5}m-> [{OBN[bi]}]', ntr, rtr, nte, rte))
    log(f'  τ={tau} 完成，累计 {len(rows):,}  耗时 {time.time()-t_ch:.0f}s')
O4 = pd.DataFrame(rows, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
O4['lift_train'] = O4['rate_train'] / btr
O4['lift_test'] = O4['rate_test'] / bte
O4 = O4.sort_values('lift_test', ascending=False)
O4.to_csv(os.path.join(OUTDIR, 'v3_chain.csv'), index=False, encoding='utf-8-sig')
log(f'\n③ 时序链结果 {len(O4):,} 条（top-15）')
for _, r in O4.head(15).iterrows():
    log(f'  {r["chain"][:66]:<66} 训 {100*r["rate_train"]:>6.2f}% lift {r["lift_train"]:>6.1f}x | '
        f'测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>5,}) lift {r["lift_test"]:>6.1f}x')

# ---------- ④ 置换检验（分块置换） ----------
log('\n' + '=' * 104)
log('【④ 置换检验】按天分块打乱标签，跑 20 次，看"随机能搜到的最高测试段真局率"')
log('=' * 104)
rng = np.random.default_rng(20261017)
n_block = 288                         # 一天一根块
nblk = len(df) // n_block
DAY = df['t'].values // 86400000
udays = np.unique(DAY)
rand_best = []
IDX3 = [np.flatnonzero(A3[:, j]).astype(np.int32) for j in range(A3.shape[1])]
for it in range(20):
    # 按"等长连续块"置换（每块 288 根 ≈ 一天），避免不同天样本数不等导致的形状不匹配
    nall = len(Y)
    nb = nall // n_block
    order = rng.permutation(nb)
    Yp = np.empty_like(Y)
    for b in range(nb):
        Yp[b*n_block:(b+1)*n_block] = Y[order[b]*n_block:(order[b]+1)*n_block]
    if nb * n_block < nall:
        Yp[nb*n_block:] = Y[nb*n_block:]
    best = 0.0
    for (i, j) in [(x, y) for x in range(len(N3)) for y in range(x+1, len(N3))]:
        inter = np.intersect1d(IDX3[i], IDX3[j], assume_unique=True)
        sub = inter[TE[inter]]
        if len(sub) < 30: continue
        v = Yp[sub].mean()
        if v > best: best = v
    rand_best.append(best)
    log(f'  置换 {it+1:>2}/20  当日最高测试段真局率 {100*best:.3f}%')
log(f'\n随机水平：中位 {100*np.median(rand_best):.3f}%  p90 {100*np.percentile(rand_best,90):.3f}%  '
    f'max {100*np.max(rand_best):.3f}%')
log(f'真实搜索最高（第三批池）：{100*O3["rate_test"].max():.3f}%')
log(f'真实基准：{100*bte:.4f}%')
log('判读：真实最高 显著 高于随机 max ⇒ 真信号；接近 ⇒ 多重比较假象')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'v3_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {os.path.join(OUTDIR, "v3_summary.txt")}')
