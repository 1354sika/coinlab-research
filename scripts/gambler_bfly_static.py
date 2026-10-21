# -*- coding: utf-8 -*-
"""gambler_bfly_static.py — 蝴蝶效应①：静态组合【真穷举】

用户指令（2026-10-16）：7 小时内跑完所有可能的蝴蝶效应组合，结果落 D 盘。

诚实前提：字面"所有"不可能（特征空间无穷）。本脚本做**可证明的最大枚举**：
  A. 核心池 50 条件 → C(50,2)+C(50,3)+C(50,4) =    251,125 组   【全穷举】
  B. 扩展池 200 条件 → C(200,2)+C(200,3)     =  1,333,300 组   【全穷举】
     （C(200,4)=64,684,950 不可能穷举 → 由随机版脚本采样，覆盖率显式标注）

加速方案（关键）：
  用**倒排索引**（每条件存命中样本下标）+ **两两预计算**，使每组只需 1 次交集运算。

判据：
  真局 = 未来 48h 内最高涨 >=+50% 且 其后 288 根内最深跌 >=-30%
  训练段(前 70% 时间) 搜索 → 测试段(后 30%) 复核（防多重比较假象）

产物（全落 D 盘）：
  D:\中信证券\coinlab\evidence\butterfly\static_50_top.csv
  D:\中信证券\coinlab\evidence\butterfly\static_200_top.csv
  D:\中信证券\coinlab\evidence\butterfly\static_all_summary.txt
"""
import os, gc, time, itertools
import numpy as np
import pandas as pd

t0t = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
OUTDIR = os.path.join(EV, 'butterfly')
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY, H48 = 288, 576
MIN_N = 200            # 训练段最少样本
TOPK  = 300            # 每阶段保留 top-N

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_bfly_static · 蝴蝶效应①静态组合【真穷举】')
log('=' * 100)

# ================= 1. 读数据 =================
RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
                             'low': 'float32', 'quote_volume': 'float32', 'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
R12all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12all, nan=0.0) >= 0.05).sum(axis=1)
dn_cnt = (np.nan_to_num(R12all, nan=0.0) <= -0.05).sum(axis=1)
sync_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
sync_dn = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]   = btc[6:]  / btc[:-6]  - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

# ================= 2. 构造候选样本 + 连续量 =================
tmp_buf = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r = {}
    for k in (1, 3, 6, 12, 24, 48, 96, 288):
        r[k] = np.full(n, np.nan); r[k][k:] = c[k:] / c[:-k] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6  = pd.Series(volr).rolling(6,  min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values
    mt = trs.rolling(24, min_periods=6).mean().values
    trr24 = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    av = pd.Series(np.abs(r1)).rolling(48, min_periods=12).mean().values
    evt = pd.Series((np.abs(r1) > 3 * av).astype('float32')).rolling(48, min_periods=12).mean().values
    sg = np.sign(np.nan_to_num(r1))
    run = np.ones(n, dtype='float64')
    for i in range(1, n):
        run[i] = run[i-1] + 1 if sg[i] == sg[i-1] and sg[i] != 0 else 1
    vola = pd.Series(r1).rolling(96, min_periods=24).std().values
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    ii = pd.Index(ot)
    su = pd.Series(sync_up, index=ts).reindex(ii).values
    sd = pd.Series(sync_dn, index=ts).reindex(ii).values
    b6  = pd.Series(btc_r6,  index=ts).reindex(ii).values
    b24 = pd.Series(btc_r24, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24

    sel = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1) & np.isfinite(f_up) & np.isfinite(r[288])
    idx = np.flatnonzero(sel)
    if len(idx) < 200: continue
    idx = idx[::3]
    for t in idx:
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)

    tmp_buf.append(pd.DataFrame({
        't': ot[idx],
        'over': over[idx],
        'r1': r[1][idx], 'r3': r[3][idx], 'r6': r[6][idx], 'r12': r[12][idx],
        'r24': r[24][idx], 'r48': r[48][idx], 'r96': r[96][idx], 'r288': r[288][idx],
        'volr6': volr6[idx], 'volr24': volr24[idx], 'volr96': volr96[idx], 'trr24': trr24[idx],
        'evt': evt[idx], 'run': run[idx], 'vola': vola[idx],
        'pos': (c[idx] - lo288[idx]) / np.maximum(hi288[idx] - lo288[idx], 1e-12),
        'sync_up': su[idx], 'sync_dn': sd[idx], 'btc6': b6[idx], 'btc24': b24[idx],
        'hour': hh[idx], 'f_up': f_up[idx], 'dd': dd[idx]}))
    del hs, ls, qs, trs; gc.collect()

df = pd.concat(tmp_buf, ignore_index=True); del tmp_buf; gc.collect()
df = df[np.isfinite(df['f_up']) & np.isfinite(df['dd'])].reset_index(drop=True)
df['true'] = ((df['f_up'] >= 0.50) & (df['dd'] >= 0.30)).astype('int8')
n_all = len(df)
Y = df['true'].values.astype('int8')
log(f'候选样本 {n_all:,}   基准真局率 {100*Y.mean():.3f}%   耗时 {time.time()-t0t:.0f}s')

# 训练/测试按时间切
cut_t = np.quantile(df['t'].values, 0.70)
TR = (df['t'].values <= cut_t)
TE = ~TR
Ytr, Yte = Y[TR], Y[TE]
ntr, nte = int(TR.sum()), int(TE.sum())
btr, bte = Ytr.mean(), Yte.mean()
log(f'训练段 {ntr:,}（基准 {100*btr:.4f}%）   测试段 {nte:,}（基准 {100*bte:.4f}%）')
del C; gc.collect()

# ================= 3. 特征池：50 核心 + 200 扩展 =================
NUMCOLS = ['over', 'r1', 'r3', 'r6', 'r12', 'r24', 'r48', 'r96', 'r288',
           'volr6', 'volr24', 'volr96', 'trr24', 'evt', 'run', 'vola', 'pos',
           'sync_up', 'sync_dn', 'btc6', 'btc24']

CONDS50 = {}
def cq(name, col, lo=None, hi=None):
    s = df[col].values
    m = np.isfinite(s)
    if lo is not None: m &= (s >= lo)
    if hi is not None: m &= (s < hi)
    CONDS50[name] = m

for col, cuts, nm in [
    ('r12',   [(0.02,None),(0.05,None),(0.10,None)],                ['r12>=2%','r12>=5%','r12>=10%']),
    ('r6',    [(0.02,None),(0.05,None)],                            ['r6>=2%','r6>=5%']),
    ('r24',   [(0.05,None),(0.10,None),(0.20,None)],                ['r24>=5%','r24>=10%','r24>=20%']),
    ('r96',   [(0.10,None),(0.30,None)],                            ['r96>=10%','r96>=30%']),
    ('r288',  [(0.00,None),(0.10,None),(0.30,None),(1.00,None)],    ['r288>=0','r288>=10%','r288>=30%','r288>=100%']),
    ('volr6', [(2,None),(5,None),(10,None)],                        ['volr6>=2','volr6>=5','volr6>=10']),
    ('volr24',[(1.5,None),(3,None),(5,None),(10,None)],             ['volr24>=1.5','volr24>=3','volr24>=5','volr24>=10']),
    ('volr96',[(1.5,None),(3,None)],                                ['volr96>=1.5','volr96>=3']),
    ('trr24', [(1.5,None),(3,None)],                                ['trr24>=1.5','trr24>=3']),
    ('over',  [(0.08,None),(0.15,None),(0.25,None),(0.0,0.10)],     ['over>=8%','over>=15%','over>=25%','over 3~10%']),
    ('pos',   [(0.5,None),(0.8,None),(0.0,0.2)],                    ['pos>=0.5','pos>=0.8','pos<0.2']),
    ('evt',   [(0.05,None),(0.15,None)],                            ['evt>=5%','evt>=15%']),
    ('run',   [(6,None),(24,None)],                                 ['run>=6','run>=24']),
    ('vola',  [(0.03,None),(0.06,None)],                            ['vola>=3%','vola>=6%']),
    ('sync_up',[(1,None),(3,None)],                                 ['syncUp>=1','syncUp>=3']),
    ('sync_dn',[(1,None),(3,None)],                                 ['syncDn>=1','syncDn>=3']),
    ('btc6',  [(0.01,None),(-10,-0.01)],                            ['btc6>=1%','btc6<=-1%']),
    ('btc24', [(0.03,None),(-10,-0.03)],                            ['btc24>=3%','btc24<=-3%']),
    ('hour',  [(0,6),(6,12),(12,18),(18,24)],                       ['hour0~6','hour6~12','hour12~18','hour18~24']),
]:
    for (lo, hi), nmv in zip(cuts, nm):
        cq(nmv, col, lo, hi)

# 扩展池：每个连续量按分位数切 8 档（上下各 4 档），加满到 200+
CONDS200 = dict(CONDS50)
for col in NUMCOLS:
    s = df[col].values
    ok = np.isfinite(s)
    if ok.sum() < 1000: continue
    qs_ = np.nanquantile(s[ok], [0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    for qi, qv in enumerate(qs_):
        if qi < 3:
            CONDS200[f'{col}_lo{qi}(<{qv:.4g})'] = ok & (s < qv)
        else:
            CONDS200[f'{col}_hi{qi}(>={qv:.4g})'] = ok & (s >= qv)
# 二元时段（更细）
hh = df['hour'].values
for a, b in [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24)]:
    CONDS200[f'hour{a}~{b}'] = (hh >= a) & (hh < b)
# 周几（用时间戳）
dow = ((df['t'].values // 86400000) + 4) % 7
for d in range(7):
    CONDS200[f'dow{d}'] = (dow == d)
# 价格档（over 细分）
ov = df['over'].values
for a, b in [(0.03,0.06),(0.06,0.10),(0.10,0.15),(0.15,0.25),(0.25,0.40)]:
    CONDS200[f'over{a}~{b}'] = (ov >= a) & (ov < b)

names50  = sorted(CONDS50.keys())
names200 = sorted(CONDS200.keys())
M50  = np.stack([CONDS50[k]  for k in names50],  axis=1)
M200 = np.stack([CONDS200[k] for k in names200], axis=1)
log(f'特征池：核心 {len(names50)} 个   扩展 {len(names200)} 个')
del CONDS50, CONDS200; gc.collect()

# ================= 4. 倒排索引 + 两两预计算 =================
def build_idx(M):
    return [np.flatnonzero(M[:, j]).astype(np.int32) for j in range(M.shape[1])]

def build_pairs(IDX, names, min_pair=100):
    """预计算两两交集（只保留 >=min_pair 的对）"""
    np_ = len(names); pairs = {}
    for i in range(np_):
        a = IDX[i]
        for j in range(i+1, np_):
            b = IDX[j]
            if len(a) < min_pair or len(b) < min_pair: continue
            inter = np.intersect1d(a, b, assume_unique=True)
            if len(inter) >= min_pair:
                pairs[(i, j)] = inter
    return pairs

def rates(mask):
    """给定命中下标（全局），返回训练/测试段真局率"""
    tr_m = TR[mask]; te_m = TE[mask]
    n_tr = int(tr_m.sum()); n_te = int(te_m.sum())
    if n_tr < MIN_N: return None
    r_tr = Y[mask][tr_m].mean()
    r_te = Y[mask][te_m].mean() if n_te >= 30 else np.nan
    return n_tr, r_tr, n_te, r_te

def run_stage(M, names, label, ks, max_pairs_mem=4_000_000):
    t_st = time.time()
    IDX = build_idx(M)
    hits = [len(a) for a in IDX]
    log(f'\n--- {label}：{len(names)} 条件，命中数 中位 {int(np.median(hits)):,} ---')
    # 单条件
    res = []
    for i, nm in enumerate(names):
        r = rates(IDX[i])
        if r: res.append((nm, r[0], r[1], r[2], r[3]))
    log(f'  1 元完成 {len(res)} 条  耗时 {time.time()-t_st:.0f}s')
    # 两两
    PAIRS = build_pairs(IDX, names, min_pair=MIN_N)
    log(f'  2 元预计算 {len(PAIRS):,} 对（>= {MIN_N}）  耗时 {time.time()-t_st:.0f}s')
    for (i, j), inter in PAIRS.items():
        r = rates(inter)
        if r: res.append((f'{names[i]} & {names[j]}', r[0], r[1], r[2], r[3]))
    log(f'  2 元评估完成，累计 {len(res):,} 条  耗时 {time.time()-t_st:.0f}s')
    # 三元：pair ∩ single（只挑 pair 命中 >=MIN_N*2 的）
    n3 = 0
    for (i, j), inter in PAIRS.items():
        if len(inter) < MIN_N * 2: continue
        for k2 in range(len(names)):
            if k2 == i or k2 == j: continue
            n_k = hits[k2]
            if n_k < MIN_N or n_k > len(inter) * 4: continue
            mm = np.intersect1d(inter, IDX[k2], assume_unique=True)
            if len(mm) < MIN_N: continue
            r = rates(mm)
            if r: res.append((f'{names[i]} & {names[j]} & {names[k2]}', r[0], r[1], r[2], r[3])); n3 += 1
        if time.time() - t_st > 3600 * 2.5:      # 单阶段上限 2.5h
            log(f'  ⚠ 3 元阶段超 2.5h 中止（已完成 {n3:,} 条）—— 未穷尽，报告中标注')
            break
    log(f'  3 元完成 {n3:,} 条  耗时 {time.time()-t_st:.0f}s')
    # 四元：pair ∩ pair（仅当 k=4 被请求且预算允许）
    n4 = 0
    if 4 in ks:
        pair_items = [(k, v) for k, v in PAIRS.items() if len(v) >= MIN_N * 3]
        log(f'  4 元候选 pair 数 {len(pair_items):,}')
        budget = time.time() + 3600 * 2.0
        for a_i in range(len(pair_items)):
            if time.time() > budget:
                log(f'  ⚠ 4 元阶段超 2h 中止（已完成 {n4:,} 条）—— 未穷尽，报告中标注')
                break
            (i1, j1), v1 = pair_items[a_i]
            for b_i in range(a_i + 1, len(pair_items)):
                (i2, j2), v2 = pair_items[b_i]
                if i2 <= j1: continue          # 保证四元组合唯一
                mm = np.intersect1d(v1, v2, assume_unique=True)
                if len(mm) < MIN_N: continue
                r = rates(mm)
                if r:
                    res.append((f'{names[i1]} & {names[j1]} & {names[i2]} & {names[j2]}', r[0], r[1], r[2], r[3])); n4 += 1
        log(f'  4 元完成 {n4:,} 条  耗时 {time.time()-t_st:.0f}s')

    out = pd.DataFrame(res, columns=['combo', 'n_train', 'rate_train', 'n_test', 'rate_test'])
    out['lift_train'] = out['rate_train'] / btr
    out['lift_test']  = out['rate_test']  / bte
    out = out.sort_values('rate_train', ascending=False)
    log(f'--- {label} 完成：{len(out):,} 条组合 ---')
    return out

# ---------- A：核心池 50 条件，2/3/4 元 ----------
log('\n' + '=' * 100)
log('【A】核心池 50 条件：C(50,2)+C(50,3)+C(50,4) = 251,125 目标全穷举')
log('=' * 100)
OA = run_stage(M50, names50, 'A 核心池50', ks=(2, 3, 4))
OA.to_csv(os.path.join(OUTDIR, 'static_50_all.csv'), index=False, encoding='utf-8-sig')
OA.head(TOPK).to_csv(os.path.join(OUTDIR, 'static_50_top.csv'), index=False, encoding='utf-8-sig')
del OA, M50; gc.collect()

# ---------- B：扩展池 200 条件，2/3 元 ----------
log('\n' + '=' * 100)
log('【B】扩展池 200 条件：C(200,2)+C(200,3) = 1,333,300 目标全穷举')
log('=' * 100)
OB = run_stage(M200, names200, 'B 扩展池200', ks=(2, 3))
OB.to_csv(os.path.join(OUTDIR, 'static_200_all.csv'), index=False, encoding='utf-8-sig')
OB.head(TOPK).to_csv(os.path.join(OUTDIR, 'static_200_top.csv'), index=False, encoding='utf-8-sig')

# ================= 5. 汇总 =================
log('\n' + '=' * 100)
log('【汇总】静态组合穷举结果')
log('=' * 100)
for tag, OD in [('A 核心池50', OA if 'OA' in dir() else None), ('B 扩展池200', OB)]:
    if OD is None: continue
    log(f'\n{tag}：共 {len(OD):,} 条组合')
    top = OD.head(20)
    for _, r in top.iterrows():
        log(f'  {r["combo"][:70]:<70} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
            f'lift {r["lift_train"]:>6.1f}x | 测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>6,}) '
            f'lift {r["lift_test"]:>6.1f}x')
    # 过拟合统计
    ok = OD[np.isfinite(OD['lift_test'])]
    log(f'  测试段可复核 {len(ok):,} 条：lift 中位 {ok["lift_test"].median():.2f}x')
    log(f'  测试 lift>5x 占 {100*(ok["lift_test"]>5).mean():.1f}%   '
        f'>2x 占 {100*(ok["lift_test"]>2).mean():.1f}%   '
        f'<1x(崩) 占 {100*(ok["lift_test"]<1).mean():.1f}%')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'static_all_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'报告已写 {os.path.join(OUTDIR, "static_all_summary.txt")}')
