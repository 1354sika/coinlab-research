# -*- coding: utf-8 -*-
"""gambler_randsearch.py — 全方位随机组合搜索（回答"有没有试过随机的蝴蝶效应测试"）

现状（这是真实的方法论缺口）：
    此前 pure / pure2 / pure3 / inter / inter2 全部是我**人工挑选的 4 个维度**
    （volr24 / r12 / over288 / r288），从未系统枚举组合。

本脚本做三件事：
  ① 构造特征池：~56 个二值条件（全部 t 时刻可观测，无前视）
  ② 随机抽 2~4 个条件求交集，跑 N=60,000 次
  ③ **训练段（前 70% 时间）搜索 → 测试段（后 30%）复核**
     —— 这是防"多重比较假象"的必需步骤：随机搜 6 万次必然能找到漂亮组合

输出：训练段 top-20 组合 + 各自测试段表现；并统计"训练强/测试崩"的比例（过拟合程度）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_randsearch.txt')
WDAY = 288
H48 = WDAY * 2
N_TRIAL = 60000
MIN_N = 200          # 组合内最少样本（防噪声）

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log(f'gambler_randsearch · 全方位随机组合搜索（{N_TRIAL:,} 次，训练段搜索→测试段复核）')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
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
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

# ---------- 构造候选样本 + 特征池 ----------
FE = {}      # 条件名 → bool 数组
VAL = {}     # 连续量（供分档）
ROWS = []    # 逐样本记录：时间戳、标签
TSL = []

def add_cond(name, arr):
    FE[name] = arr

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
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
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
    b6 = pd.Series(btc_r6, index=ts).reindex(ii).values
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

df = pd.concat(tmp_buf, ignore_index=True)
del tmp_buf; gc.collect()
df = df[np.isfinite(df['f_up']) & np.isfinite(df['dd'])]
df['true'] = ((df['f_up'] >= 0.50) & (df['dd'] >= 0.30)).astype('int8')
log(f'候选样本 {len(df):,}   基准真局率 {100*df["true"].mean():.3f}%')

# ---------- 特征池：把连续量切成二值条件 ----------
CONDS = {}
def cq(name, series, lo=None, hi=None):
    m = np.ones(len(series), dtype=bool)
    if lo is not None: m &= (series.values >= lo)
    if hi is not None: m &= (series.values < hi)
    CONDS[name] = m

for col, cuts, nm in [
    ('r12',   [(0.02,None),(0.05,None),(0.10,None)],                     ['r12>=2%','r12>=5%','r12>=10%']),
    ('r6',    [(0.02,None),(0.05,None)],                                 ['r6>=2%','r6>=5%']),
    ('r24',   [(0.05,None),(0.10,None),(0.20,None)],                     ['r24>=5%','r24>=10%','r24>=20%']),
    ('r96',   [(0.10,None),(0.30,None)],                                 ['r96>=10%','r96>=30%']),
    ('r288',  [(0.00,None),(0.10,None),(0.30,None),(1.00,None)],         ['r288>=0','r288>=10%','r288>=30%','r288>=100%']),
    ('volr6', [(2,None),(5,None),(10,None)],                             ['volr6>=2','volr6>=5','volr6>=10']),
    ('volr24',[(1.5,None),(3,None),(5,None),(10,None)],                  ['volr24>=1.5','volr24>=3','volr24>=5','volr24>=10']),
    ('volr96',[(1.5,None),(3,None)],                                     ['volr96>=1.5','volr96>=3']),
    ('trr24', [(1.5,None),(3,None)],                                     ['trr24>=1.5','trr24>=3']),
    ('over',  [(0.08,None),(0.15,None),(0.25,None),(0.0,0.10)],          ['over>=8%','over>=15%','over>=25%','over 3~10%']),
    ('pos',   [(0.5,None),(0.8,None),(0.0,0.2)],                         ['pos>=0.5','pos>=0.8','pos<0.2']),
    ('evt',   [(0.05,None),(0.15,None)],                                 ['evt>=5%','evt>=15%']),
    ('run',   [(6,None),(24,None)],                                      ['run>=6','run>=24']),
    ('vola',  [(0.03,None),(0.06,None)],                                 ['vola>=3%','vola>=6%']),
    ('sync_up',[(1,None),(3,None)],                                      ['syncUp>=1','syncUp>=3']),
    ('sync_dn',[(1,None),(3,None)],                                      ['syncDn>=1','syncDn>=3']),
    ('btc6',  [(0.01,None),(-10,-0.01)],                                 ['btc6>=1%','btc6<=-1%']),
    ('btc24', [(0.03,None),(-10,-0.03)],                                 ['btc24>=3%','btc24<=-3%']),
    ('hour',  [(0,6),(6,12),(12,18),(18,24)],                            ['hour0~6','hour6~12','hour12~18','hour18~24']),
]:
    for (lo, hi), nmv in zip(cuts, nm):
        cq(nmv, df[col], lo, hi)
    # 反向条件（低档）
for col, nmv in [('r12','r12<2%'),('volr24','volr24<1.5'),('over','over<8%')]:
    pass
conds_names = sorted(CONDS.keys())
M = np.stack([CONDS[k] for k in conds_names], axis=1)      # (n, C) bool
log(f'特征池 {len(conds_names)} 个条件')

# ---------- 训练/测试切分（按时间，防多重比较） ----------
cut_t = np.quantile(df['t'].values, 0.70)
tr = df['t'].values <= cut_t
te = ~tr
Y = df['true'].values.astype('int8')
base_tr = Y[tr].mean(); base_te = Y[te].mean()
log(f'训练段 {tr.sum():,}（基准 {100*base_tr:.3f}%）  测试段 {te.sum():,}（基准 {100*base_te:.3f}%）')

rng = np.random.default_rng(42)
res = []
ncond = len(conds_names)
for it in range(N_TRIAL):
    k = int(rng.integers(2, 5))
    pick = rng.choice(ncond, size=k, replace=False)
    m = M[:, pick[0]].copy()
    for p in pick[1:]:
        m &= M[:, p]
        if int(m[tr].sum()) < MIN_N: break
    ntr = int(m[tr].sum())
    if ntr < MIN_N: continue
    rtr = Y[tr & m].mean()
    res.append((rtr, ntr, tuple(pick)))
res.sort(key=lambda x: -x[0])
log(f'有效组合 {len(res):,}（样本>= {MIN_N}）')

log('')
log('=' * 100)
log('【训练段 top-20 组合 → 测试段复核】')
log('=' * 100)
log(f'  {"组合":<64} {"训练真局率":>10} {"训练n":>8} {"测试真局率":>10} {"测试n":>8} {"衰减比":>8}')
kept = 0
decs = []
for rtr, ntr, pick in res[:20]:
    m = M[:, pick[0]].copy()
    for p in pick[1:]: m &= M[:, p]
    nte = int(m[te].sum())
    if nte < 20:
        log(f'  {" & ".join(conds_names[p] for p in pick):<64} {100*rtr:>9.3f}% {ntr:>8,} {"(测试样本不足)":>10}')
        continue
    rte = Y[te & m].mean()
    name = " & ".join(conds_names[p] for p in pick)
    log(f'  {name[:62]:<64} {100*rtr:>9.3f}% {ntr:>8,} {100*rte:>9.3f}% {nte:>8,} {rte/base_te:>7.2f}x')
    decs.append(rte / base_te)
    kept += 1

log('')
log('=' * 100)
log('【过拟合诊断：训练段 top-100 组合在测试段的衰减】')
log('=' * 100)
rats = []
for rtr, ntr, pick in res[:100]:
    m = M[:, pick[0]].copy()
    for p in pick[1:]: m &= M[:, p]
    nte = int(m[te].sum())
    if nte < 20: continue
    rte = Y[te & m].mean()
    rats.append(rte / base_te)
if rats:
    rats = np.array(rats)
    log(f'  top-100 中可在测试段复核的 {len(rats)} 组：')
    log(f'    测试段 lift 中位 {np.median(rats):.2f}x   均值 {rats.mean():.2f}x   '
        f'最大 {rats.max():.2f}x   最小 {rats.min():.2f}x')
    log(f'    测试段 lift <1（比基准差）的占比 {100.0*(rats < 1).mean():.1f}%')
    log(f'    测试段 lift >2 的占比 {100.0*(rats > 2).mean():.1f}%')
log('')
log('【判读】')
log('=' * 100)
log('  · top-20 若测试段普遍衰减到 1~2x ⇒ 训练段的"发现"多为多重比较噪声')
log('  · 若个别组合测试段仍 >3x 且样本>=100 ⇒ 值得进一步验证（但须做时间+币双留出）')
log('  · 全程无前视：所有条件在 t 时刻可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
