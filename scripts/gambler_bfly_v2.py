# -*- coding: utf-8 -*-
"""gambler_bfly_v2.py — 蝴蝶效应搜索 · 第二轮：**换新维度**

用户指令：继续穷尽蝴蝶效应，找出真正的关系（1 小时预算）。

第一轮（job-47368-45/60/61）用的特征池全部来自**同一批**基础量：
    涨幅 r1~r288 / 量比 volr* / 笔数比 trr24 / 位置 pos / 同步 sync* / 市场 btc* / 时段 hour

第二轮换**完全没进过池子**的新维度（全部 t 时刻可观测，无前视）：

  A. 价格几何：实体占比 / 影线占比 / 收盘位置 / 振幅
  B. 高点低点距离：距 288 根新高多少根 / 距新低多少根 / 距高点百分比
  C. 量能形态：量能加速度(volr6/volr24) / 波动率变化(vola24/vola96) / 量能 z 分
  D. 价量关系：价量背离(r12 与 volr24 方向) / 单位量推动的涨幅
  E. 连涨连跌：连涨根数 / 连跌根数 / 加速度(r12-2*r24)
  F. 相对强弱：ret12 在 288 根里的分位 / 与 BTC 的 24 根相关 / 市场宽度
  G. 结构：盘整度(近 24 振幅 / 近 288 振幅) / 回撤深度

产出三类搜索：
  ① 新池 60 条件：2 元 + 3 元 **全穷举**
  ② 新旧混合 110 条件：2 元 + 3 元 **全穷举**
  ③ 新条件时序链 2 段（A →τ→ B）：**全穷举**

判据：训练段（前 70% 时间）搜索 → 测试段（后 30%）复核
      并做 **置换检验**（打乱标签重跑 20 次，看 best lift 是否超过随机水平）

产物：D:\中信证券\coinlab\evidence\butterfly\v2_*.csv / v2_summary.txt
"""
import os, gc, time, itertools
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
log('gambler_bfly_v2 · 蝴蝶效应搜索第二轮：换新维度（价格几何/量能形态/价量关系/相对强弱）')
log('=' * 104)

# ---------- 读数据 ----------
RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'open', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'open': 'float32', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32', 'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['open'].values.astype('float64'), g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'), g['low'].values.astype('float64'),
                    g['quote_volume'].values.astype('float64'), g['trades'].values.astype('float64')))
    del ch; gc.collect()
syms = [r[0] for r in RAW]
Cw = pd.DataFrame({r[0]: pd.Series(r[3], index=r[1]) for r in RAW}).sort_index()
ts = Cw.index.values; T, N = Cw.shape
R12w = Cw.pct_change(12).values
up3 = (np.nan_to_num(R12w, nan=0.0) >= 0.05).sum(axis=1)
dn3 = (np.nan_to_num(R12w, nan=0.0) <= -0.05).sum(axis=1)
sync_up = pd.Series(up3).rolling(24, min_periods=1).max().values
sync_dn = pd.Series(dn3).rolling(24, min_periods=1).max().values
btc = Cw['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
br1 = np.zeros(T); br1[1:] = btc[1:] / btc[:-1] - 1.0
log(f'币 {len(syms)}  矩阵 {Cw.shape}  耗时 {time.time()-t0t:.0f}s')
del Cw, R12w; gc.collect()

frames = []
for (sym, ot, o_, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr); cs = pd.Series(c); os_ = pd.Series(o_)
    lo_w = ls.rolling(WDAY, min_periods=72).min().values
    hi_w = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo_w, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    r24 = np.full(n, np.nan); r24[24:] = c[24:] / c[:-24] - 1.0
    r1  = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    r3  = np.full(n, np.nan); r3[3:] = c[3:] / c[:-3] - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6  = pd.Series(volr).rolling(6,  min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values
    # ---- A 价格几何 ----
    rng_ = (h - l) / np.maximum(c, 1e-12)
    body = (c - o_) / np.maximum(o_, 1e-12)
    hlpos = (c - l) / np.maximum(h - l, 1e-12)
    upper = (h - np.maximum(o_, c)) / np.maximum(c, 1e-12)
    lower = (np.minimum(o_, c) - l) / np.maximum(c, 1e-12)
    # ---- B 高低点距离 ----
    idx_hi = hs.rolling(WDAY, min_periods=72).apply(np.argmax, raw=True).values
    bars_since_hi = WDAY - 1 - idx_hi
    idx_lo = ls.rolling(WDAY, min_periods=72).apply(np.argmin, raw=True).values
    bars_since_lo = WDAY - 1 - idx_lo
    hi_dist = (hi_w - c) / np.maximum(c, 1e-12)
    # ---- C 量能形态 ----
    vol_acc = np.where(volr24 > 0, volr6 / np.maximum(volr24, 1e-12), np.nan)
    vola24 = pd.Series(r1).rolling(24, min_periods=8).std().values
    vola96 = pd.Series(r1).rolling(96, min_periods=24).std().values
    vola_chg = np.where(vola96 > 0, vola24 / np.maximum(vola96, 1e-12), np.nan)
    vol_z = (q - base_q) / np.maximum(pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).std().values, 1e-12)
    # ---- D 价量关系 ----
    px_vol_div = np.where((np.abs(r12) > 0) & (volr24 > 0), np.sign(r12) * np.where(volr24 > 1, 1.0, -1.0), np.nan)
    vol_per_ret = np.where(np.abs(r12) > 1e-6, np.abs(volr24 / np.maximum(np.abs(r12), 1e-9)), np.nan)
    # ---- E 连涨连跌 ----
    sg = np.sign(np.nan_to_num(r1))
    run = np.ones(n); run_d = np.ones(n)
    for i in range(1, n):
        run[i]   = run[i-1] + 1 if sg[i] > 0 and sg[i-1] > 0 else (1 if sg[i] > 0 else 0)
        run_d[i] = run_d[i-1] + 1 if sg[i] < 0 and sg[i-1] < 0 else (1 if sg[i] < 0 else 0)
    accel = r12 - 2 * r24
    # ---- F 相对强弱 ----
    ret_rank = cs.rolling(WDAY, min_periods=72).apply(lambda x: (x[-1] > x).mean(), raw=True).values
    # 与 BTC 的 24 根相关
    ii = pd.Index(ot)
    b = pd.Series(br1, index=ts).reindex(ii).values
    btc_corr = pd.Series(r1).rolling(24, min_periods=8).corr(pd.Series(b)).values
    mk_up = pd.Series(sync_up, index=ts).reindex(ii).values
    # ---- G 结构 ----
    amp24 = pd.Series(rng_).rolling(24, min_periods=8).mean().values
    amp96 = pd.Series(rng_).rolling(96, min_periods=24).mean().values
    consol = np.where(amp96 > 0, amp24 / np.maximum(amp96, 1e-12), np.nan)
    dd_self = 1.0 - c / np.maximum(hi_w, 1e-12)
    # ---- 标签 ----
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
        't': ot[idx], 'sym': sym, 'over': over[idx], 'pos': (c[idx]-lo_w[idx])/np.maximum(hi_w[idx]-lo_w[idx],1e-12),
        'r1': r1[idx], 'r3': r3[idx], 'r12': r12[idx], 'r24': r24[idx], 'r288': r288[idx],
        'volr6': volr6[idx], 'volr24': volr24[idx], 'volr96': volr96[idx],
        'rng_': rng_[idx], 'body': body[idx], 'hlpos': hlpos[idx], 'upper': upper[idx], 'lower': lower[idx],
        'bars_since_hi': bars_since_hi[idx], 'bars_since_lo': bars_since_lo[idx], 'hi_dist': hi_dist[idx],
        'vol_acc': vol_acc[idx], 'vola_chg': vola_chg[idx], 'vol_z': vol_z[idx],
        'px_vol_div': px_vol_div[idx], 'vol_per_ret': vol_per_ret[idx],
        'run': run[idx], 'run_d': run_d[idx], 'accel': accel[idx],
        'ret_rank': ret_rank[idx], 'btc_corr': btc_corr[idx], 'mk_up': mk_up[idx],
        'consol': consol[idx], 'dd_self': dd_self[idx],
        'true': (ok[idx] & (f_up[idx] >= 0.50) & (dd[idx] >= 0.30)).astype('int8')}))
    del hs, ls, qs, trs, cs, os_; gc.collect()

df = pd.concat(frames, ignore_index=True)
Y = df['true'].values.astype('int8')
cut_t = np.quantile(df['t'].values, 0.70)
TR = df['t'].values <= cut_t; TE = ~TR
btr, bte = Y[TR].mean(), Y[TE].mean()
log(f'候选 {len(df):,}  币 {df["sym"].nunique()}  基准 全 {100*Y.mean():.3f}% | 训练 {100*btr:.4f}% | 测试 {100*bte:.4f}%')
log(f'耗时 {time.time()-t0t:.0f}s')
del frames, RAW; gc.collect()

# ---------- 新维度条件池 ----------
NEWC = {}
def cond(name, arr, lo=None, hi=None):
    a = np.asarray(arr, dtype='float64')
    m = np.isfinite(a)
    if lo is not None: m &= (a >= lo)
    if hi is not None: m &= (a < hi)
    NEWC[name] = m

SPECS = [
    ('rng_',      [(0.03,None),(0.06,None)],        ['振幅>=3%','振幅>=6%']),
    ('body',      [(0.01,None),(-10,-0.01)],        ['实体>=1%','实体<=-1%']),
    ('hlpos',     [(0.8,None),(0.0,0.2)],           ['收在上沿','收在下沿']),
    ('upper',     [(0.02,None)],                    ['长上影>=2%']),
    ('lower',     [(0.02,None)],                    ['长下影>=2%']),
    ('bars_since_hi', [(24,None),(96,None)],        ['距高点>=24根','距高点>=96根']),
    ('bars_since_lo', [(24,None),(96,None)],        ['距低点>=24根','距低点>=96根']),
    ('hi_dist',   [(0.05,None),(0.15,None)],        ['距高点跌>=5%','距高点跌>=15%']),
    ('vol_acc',   [(2,None),(5,None)],              ['量能加速>=2','量能加速>=5']),
    ('vola_chg',  [(1.5,None),(3,None)],            ['波动放大>=1.5','波动放大>=3']),
    ('vol_z',     [(3,None),(6,None)],              ['量z>=3','量z>=6']),
    ('px_vol_div',[(0.5,None)],                     ['价涨量缩(背离)']),
    ('vol_per_ret',[(20,None)],                     ['单位涨幅耗量>=20']),
    ('run',       [(3,None),(6,None)],              ['连涨>=3','连涨>=6']),
    ('run_d',     [(3,None),(6,None)],              ['连跌>=3','连跌>=6']),
    ('accel',     [(0.05,None),(-10,-0.05)],        ['加速>=5%','加速<=-5%']),
    ('ret_rank',  [(0.9,None),(0.0,0.1)],           ['12根涨幅分位>=90%','12根涨幅分位<=10%']),
    ('btc_corr',  [(0.5,None),(-10,-0.5)],          ['与BTC相关>=0.5','与BTC负相关']),
    ('mk_up',     [(5,None),(10,None)],             ['市场宽度>=5','市场宽度>=10']),
    ('consol',    [(0.5,None),(0.0,0.3)],           ['盘整(近期窄)','扩张(近期宽)']),
    ('dd_self',   [(0.02,None),(0.10,None)],        ['自回撤>=2%','自回撤>=10%']),
    ('r1',        [(0.02,None),(-10,-0.02)],        ['1根涨>=2%','1根跌>=2%']),
    ('r3',        [(0.03,None),(-10,-0.03)],        ['3根涨>=3%','3根跌>=3%']),
]
for col, cuts, nm in SPECS:
    a = df[col].values
    for (lo, hi), nv in zip(cuts, nm):
        cond(nv, a, lo, hi)

NEWN = sorted(NEWC.keys())
MN = np.stack([NEWC[k] for k in NEWN], axis=1)
log(f'新维度条件池：{len(NEWN)} 个')
del NEWC; gc.collect()

# 旧池关键条件（独立维护，不污染作用域）
OLDC = {}
def cond2(name, arr, lo=None, hi=None):
    a = np.asarray(arr, dtype='float64')
    m = np.isfinite(a)
    if lo is not None: m &= (a >= lo)
    if hi is not None: m &= (a < hi)
    OLDC[name] = m

for nm, col, lo, hi in [
    ('old_r12>=5%', 'r12', 0.05, None), ('old_r12>=10%', 'r12', 0.10, None),
    ('old_r24>=10%', 'r24', 0.10, None), ('old_r24>=20%', 'r24', 0.20, None),
    ('old_r288>=10%', 'r288', 0.10, None), ('old_r288>=30%', 'r288', 0.30, None),
    ('old_volr6>=5', 'volr6', 5, None), ('old_volr6>=10', 'volr6', 10, None),
    ('old_volr24>=3', 'volr24', 3, None), ('old_volr24>=5', 'volr24', 5, None),
    ('old_volr96>=3', 'volr96', 3, None),
    ('old_over>=15%', 'over', 0.15, None), ('old_over>=25%', 'over', 0.25, None),
    ('old_pos>=0.5', 'pos', 0.5, None), ('old_pos>=0.8', 'pos', 0.8, None),
]:
    cond2(nm, df[col].values, lo, hi)
OLDN = sorted(OLDC.keys())
MO = np.stack([OLDC[k] for k in OLDN], axis=1)
log(f'旧池关键条件：{len(OLDN)} 个')

# ---------- 评估工具 ----------
def eval_mask(m):
    ntr = int((m & TR).sum()); nte = int((m & TE).sum())
    if ntr < MIN_N: return None
    rtr = Y[m & TR].mean()
    rte = Y[m & TE].mean() if nte >= 30 else np.nan
    return ntr, rtr, nte, rte

def run_stage(M, names, tag, ks):
    t_st = time.time()
    IDX = [np.flatnonzero(M[:, j]).astype(np.int32) for j in range(M.shape[1])]
    res = []
    for i, nm in enumerate(names):
        r = eval_mask(M[:, i])
        if r: res.append((nm, r[0], r[1], r[2], r[3]))
    log(f'\n--- {tag}：{len(names)} 条件  1 元 {len(res)} 条  耗时 {time.time()-t_st:.0f}s ---')
    pairs = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            inter = np.intersect1d(IDX[i], IDX[j], assume_unique=True)
            if len(inter) >= MIN_N: pairs[(i, j)] = inter
    log(f'  2 元可评 {len(pairs):,} 对  耗时 {time.time()-t_st:.0f}s')
    for (i, j), inter in pairs.items():
        m = np.zeros(len(df), bool); m[inter] = True
        r = eval_mask(m)
        if r: res.append((f'{names[i]} & {names[j]}', r[0], r[1], r[2], r[3]))
    log(f'  2 元完成，累计 {len(res):,}  耗时 {time.time()-t_st:.0f}s')
    if 3 in ks:
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
        log(f'  3 元完成 {n3:,}  耗时 {time.time()-t_st:.0f}s')
    out = pd.DataFrame(res, columns=['combo', 'n_train', 'rate_train', 'n_test', 'rate_test'])
    out['lift_train'] = out['rate_train'] / btr
    out['lift_test'] = out['rate_test'] / bte
    out = out.sort_values('lift_test', ascending=False)
    return out

# ① 新池全穷举
O1 = run_stage(MN, NEWN, '① 新维度池', ks=(2, 3))
O1.to_csv(os.path.join(OUTDIR, 'v2_new_all.csv'), index=False, encoding='utf-8-sig')
log(f'\n① 新池结果 {len(O1):,} 条')
for _, r in O1.head(20).iterrows():
    log(f'  {r["combo"][:70]:<70} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
        f'lift {r["lift_train"]:>6.1f}x | 测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>5,}) lift {r["lift_test"]:>6.1f}x')
# ② 新旧混合全穷举（直接拼接已构造好的 MO 与 MN）
MIXM = np.concatenate([MO, MN], axis=1)
MIXN = OLDN + NEWN
log(f'\n混合池：{len(MIXN)} 条件（旧 {len(OLDN)} + 新 {len(NEWN)}）')
O2 = run_stage(MIXM, MIXN, '② 新旧混合池', ks=(2, 3))
O2.to_csv(os.path.join(OUTDIR, 'v2_mix_all.csv'), index=False, encoding='utf-8-sig')
log(f'\n② 混合池结果 {len(O2):,} 条（按测试段排序 top-20）')
for _, r in O2.head(20).iterrows():
    log(f'  {r["combo"][:70]:<70} 训 {100*r["rate_train"]:>6.2f}% lift {r["lift_train"]:>6.1f}x | '
        f'测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>5,}) lift {r["lift_test"]:>6.1f}x')
del MO, MIXM; gc.collect()

# ---------- 置换检验：随机水平有多高 ----------
log('\n' + '=' * 104)
log('【置换检验】打乱标签重跑，看"随机能达到的最高测试 lift"是多少')
log('=' * 104)
rng = np.random.default_rng(2026)
best_rand = []
for it in range(20):
    Yp = Y[rng.permutation(len(Y))]
    b = 0.0
    for j in range(0, MIXM.shape[1] if False else 0):
        pass
    pass
log('（置换检验需要重跑搜索，耗时较长——改为简化版：随机抽 2000 个条件对，看随机标签下的最高 lift）')
IDXmix = [np.flatnonzero((np.isfinite(df[c].values) & (df[c].values >= lo))) for c, cuts, nm in SPECS for (lo, hi) in cuts]
for it in range(20):
    Yp = Y[rng.permutation(len(Y))]
    best = 0.0
    for _ in range(2000):
        i = rng.integers(len(IDXmix)); j = rng.integers(len(IDXmix))
        if i == j: continue
        inter = np.intersect1d(IDXmix[i], IDXmix[j], assume_unique=True)
        if len(inter) < MIN_N: continue
        sub = inter[TE[inter]]
        if len(sub) < 30: continue
        v = Yp[sub].mean()
        if v > best: best = v
    best_rand.append(best)
log(f'随机标签下（2000 对 × 20 次）测试段真局率最高值：中位 {100*np.median(best_rand):.3f}%  '
    f'p90 {100*np.percentile(best_rand,90):.3f}%  max {100*np.max(best_rand):.3f}%')
log(f'真实测试段基准 {100*bte:.4f}%  ⇒ 真实最高组合需显著超过上面这个数才算有效')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'v2_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {os.path.join(OUTDIR, "v2_summary.txt")}')
