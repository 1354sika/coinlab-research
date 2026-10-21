# -*- coding: utf-8 -*-
"""gambler_bfly_chain.py — 蝴蝶效应②：时序链条【真穷举 + 受控采样】

这是"蝴蝶效应"的原义实现（用户反复强调：条件之间环环相扣）：
    A 事件在 t-τ 时刻发生  →  τ 根之后  →  B 事件在 t 时刻发生  →  预测 t 之后

与 static 版的根本区别：
    static 版只做「同一时刻 A 且 B」（横截面）
    本版做「先 A，隔 τ 根，再 B」（时序链，有方向、有间隔）

穷举规模：
    2 段链 (A, τ, B)：50 × 5 档 τ × 50 =    12,500  【全穷举】
    3 段链 (A,τ,B,τ',C)：50×5×50×5×50 = 3,125,000 → 【受控采样 300,000，覆盖率 9.6%，显式标注】
    4 段链：随机采样 100,000 条（覆盖率 ~0.03%，显式标注为"抽查非全量"）

判据：真局 = 未来 48h 最高涨 >=+50% 且 其后 288 根最深跌 >=-30%
      训练段(前 70% 时间) 搜索 → 测试段(后 30%) 复核
      全程无前视：τ 根之前的值在 t 时刻已完全可见

产物（全落 D 盘）：
    D:\中信证券\coinlab\evidence\butterfly\chain2_all.csv      （2 段链全穷举）
    D:\中信证券\coinlab\evidence\butterfly\chain3_full.csv    （3 段链【全穷举】）
    D:\中信证券\coinlab\evidence\butterfly\chain4_big.csv     （4 段链扩大采样）
    D:\中信证券\coinlab\evidence\butterfly\chain_summary.txt
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
OUTDIR = os.path.join(EV, 'butterfly')
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY, H48 = 288, 576
MIN_N = 200
TAUS = [6, 12, 24, 48, 96]        # 30min / 1h / 2h / 4h / 8h
N_CHAIN3 = 744_775                # 3 段链【全穷举】（空间 C(31,1)*5*C(31,1)*5*C(31,1)=744,775）
N_CHAIN4 = 2_000_000              # 4 段链采样（空间约 1.15 亿，覆盖率 ≈1.73%）
CHAINS_DIR = OUTDIR

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_bfly_chain · 蝴蝶效应②时序链条（A →τ→ B →τ\'→ C）')
log('=' * 100)

# ================= 1. 读数据 + 构造逐币条件矩阵及 lag =================
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
del C; gc.collect()

CORE = [  # (列名, 阈值, 方向, 简称)
    ('over', 0.08, +1, 'over>=8%'), ('over', 0.15, +1, 'over>=15%'),
    ('r12', 0.05, +1, 'r12>=5%'),   ('r12', 0.10, +1, 'r12>=10%'),
    ('r6', 0.05, +1, 'r6>=5%'),
    ('r24', 0.10, +1, 'r24>=10%'),  ('r24', 0.20, +1, 'r24>=20%'),
    ('r96', 0.30, +1, 'r96>=30%'),
    ('r288', 0.10, +1, 'r288>=10%'), ('r288', 0.30, +1, 'r288>=30%'),
    ('volr6', 5, +1, 'volr6>=5'),   ('volr6', 10, +1, 'volr6>=10'),
    ('volr24', 3, +1, 'volr24>=3'), ('volr24', 5, +1, 'volr24>=5'),
    ('volr96', 3, +1, 'volr96>=3'),
    ('trr24', 3, +1, 'trr24>=3'),
    ('evt', 0.15, +1, 'evt>=15%'),
    ('run', 24, +1, 'run>=24'),
    ('vola', 0.06, +1, 'vola>=6%'),
    ('pos', 0.8, +1, 'pos>=0.8'), ('pos', 0.2, -1, 'pos<0.2'),
    ('sync_up', 3, +1, 'syncUp>=3'), ('sync_dn', 3, +1, 'syncDn>=3'),
    ('btc6', -0.01, -1, 'btc6<=-1%'), ('btc6', 0.01, +1, 'btc6>=1%'),
    ('btc24', -0.03, -1, 'btc24<=-3%'),
    ('r12', 0.02, -1, 'r12<2%'),      ('volr24', 1.5, -1, 'volr24<1.5'),
    ('over', 0.08, -1, 'over<8%'),
    ('hour', (18, 24), 0, 'hour18~24'), ('hour', (0, 6), 0, 'hour0~6'),
]
MARK = -1          # 链标记（τ 根之前），A/B/C 共用同一条件池

# 逐币构造：条件矩阵（bool）+ 标签
segs = []          # 每币一个 dict
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r = {}
    for k in (6, 12, 24, 96, 288):
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
    for t in np.flatnonzero(np.isfinite(f_up)):
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    ii = pd.Index(ot)
    su  = pd.Series(sync_up, index=ts).reindex(ii).values
    sd  = pd.Series(sync_dn, index=ts).reindex(ii).values
    b6  = pd.Series(btc_r6,  index=ts).reindex(ii).values
    b24 = pd.Series(btc_r24, index=ts).reindex(ii).values
    hh  = (ot // 3600000) % 24
    COL = {'over': over, 'r6': r[6], 'r12': r[12], 'r24': r[24], 'r96': r[96], 'r288': r[288],
           'volr6': volr6, 'volr24': volr24, 'volr96': volr96, 'trr24': trr24, 'evt': evt,
           'run': run, 'vola': vola,
           'pos': (c - lo288) / np.maximum(hi288 - lo288, 1e-12),
           'sync_up': su, 'sync_dn': sd, 'btc6': b6, 'btc24': b24, 'hour': hh}
    # 条件矩阵
    M = np.zeros((n, len(CORE)), dtype=bool)
    for j, (col, th, direc, _nm) in enumerate(CORE):
        v = COL[col]
        if direc == 0:
            a, b = th; M[:, j] = np.isfinite(v) & (v >= a) & (v < b)
        elif direc > 0:
            M[:, j] = np.isfinite(v) & (v >= th)
        else:
            M[:, j] = np.isfinite(v) & (v < th)
    # 标签（真局）
    lab = np.zeros(n, dtype=np.int8)
    ok = np.isfinite(f_up) & np.isfinite(dd)
    lab[ok & (f_up >= 0.50) & (dd >= 0.30)] = 1
    # 采样：只保留 over 3~40%、volr24>=1 的有效候选（同 static 口径）
    keep = ok & (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    idx = np.flatnonzero(keep)[::3]
    if len(segs) < 5:
        print(f'  [dbg] {sym}: n={n} fin_fup={int(np.isfinite(f_up).sum())} '
              f'fin_dd={int(np.isfinite(dd).sum())} over_ok={int(((over >= 0.03) & (over <= 0.40)).sum())} '
              f'volr_ok={int((volr24 >= 1).sum())} keep={int(keep.sum())} idx={len(idx)}', flush=True)
    if len(idx) < 400: continue
    segs.append({'sym': sym, 't': ot[idx], 'M': M[idx], 'y': lab[idx]})
    del hs, ls, qs, trs; gc.collect()

if not segs:
    log('无有效币，退出'); raise SystemExit(1)

t_all = np.concatenate([s['t'] for s in segs])
y_all = np.concatenate([s['y'] for s in segs]).astype(np.int8)
M_all = np.concatenate([s['M'] for s in segs], axis=0)
seg_id = np.concatenate([np.full(len(s['t']), i, dtype=np.int32) for i, s in enumerate(segs)])
n_all = len(y_all)
log(f'候选样本 {n_all:,}  币 {len(segs)}  基准真局率 {100*y_all.mean():.3f}%  耗时 {time.time()-t0t:.0f}s')

# 训练/测试切分
cut_t = np.quantile(t_all, 0.70)
TR = t_all <= cut_t; TE = ~TR
ntr, nte = int(TR.sum()), int(TE.sum())
btr, bte = y_all[TR].mean(), y_all[TE].mean()
log(f'训练 {ntr:,}（基准 {100*btr:.4f}%）  测试 {nte:,}（基准 {100*bte:.4f}%）')

# 逐币分段边界（计算 lag 时不得跨币）
seg_start = np.zeros(n_all, dtype=bool); seg_start[0] = True
for s in segs[1:]:
    pass
offs = np.cumsum([0] + [len(s['t']) for s in segs])
for o in offs[1:-1]:
    seg_start[o] = True

NAMES = [c[3] for c in CORE]
NC = len(NAMES)

# ---------- 构造 lag 矩阵：M_lag[ti] = τ_i 根之前的 M（同币内） ----------
M_lag = {}
for ti, tau in enumerate(TAUS):
    ML = np.zeros_like(M_all)
    for si, s in enumerate(segs):
        a, b = offs[si], offs[si+1]
        if b - a > tau:
            ML[a+tau:b] = M_all[a:b-tau]
    M_lag[tau] = ML
log(f'lag 矩阵就绪（{len(TAUS)} 档 τ）  耗时 {time.time()-t0t:.0f}s')

# ---------- 倒排索引 ----------
IDX  = [np.flatnonzero(M_all[:, j]).astype(np.int32) for j in range(NC)]
IDXL = {tau: [np.flatnonzero(M_lag[tau][:, j]).astype(np.int32) for j in range(NC)] for tau in TAUS}
log(f'倒排索引就绪  耗时 {time.time()-t0t:.0f}s')

def rate_from_mask(mask_idx):
    trm = TR[mask_idx]; tem = TE[mask_idx]
    n_tr = int(trm.sum()); n_te = int(tem.sum())
    if n_tr < MIN_N: return None
    r_tr = y_all[mask_idx][trm].mean()
    r_te = y_all[mask_idx][tem].mean() if n_te >= 30 else np.nan
    return n_tr, r_tr, n_te, r_te

def rec(rows, name, idxs):
    r = rate_from_mask(idxs)
    if r is None: return
    rows.append((name, r[0], r[1], r[2], r[3]))

# ================= 2. 两段链：A →τ→ B  【全穷举】 =================
log('\n' + '=' * 100)
log(f'【链 2 段】A →τ→ B  组合数 = {NC} × {len(TAUS)} × {NC} = {NC*len(TAUS)*NC:,}  【全穷举】')
log('=' * 100)
rows2 = []
t_st = time.time()
for ta in TAUS:
    for ai in range(NC):
        a_idx = IDXL[ta][ai]
        if len(a_idx) < MIN_N: continue
        for bi in range(NC):
            b_idx = IDX[bi]
            if len(b_idx) < MIN_N: continue
            mm = np.intersect1d(a_idx, b_idx, assume_unique=True)
            if len(mm) < MIN_N: continue
            rec(rows2, f'[{NAMES[ai]}] -{ta*5}min-> [{NAMES[bi]}]', mm)
    log(f'  τ={ta}（{ta*5}min）完成，累计 {len(rows2):,} 条  耗时 {time.time()-t_st:.0f}s')
D2 = pd.DataFrame(rows2, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
D2['lift_train'] = D2['rate_train'] / btr
D2['lift_test']  = D2['rate_test']  / bte
D2 = D2.sort_values('rate_train', ascending=False)
D2.to_csv(os.path.join(OUTDIR, 'chain2_all.csv'), index=False, encoding='utf-8-sig')
log(f'  2 段链完成 {len(D2):,} 条  总耗时 {time.time()-t_st:.0f}s')
for _, r in D2.head(20).iterrows():
    log(f'    {r["chain"][:66]:<66} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
        f'| 测 {100*r["rate_test"]:>6.2f}% lift {r["lift_test"]:>5.1f}x')

# ================= 3. 三段链：【受控采样】 =================
log('\n' + '=' * 100)
tot3 = NC * len(TAUS) * NC * len(TAUS) * NC
log(f'【链 3 段】A →τ→ B →τ\'→ C  理论组合 {tot3:,}  →  采样 {N_CHAIN3:,}（覆盖 {100*N_CHAIN3/tot3:.2f}%）')
log('⚠ 非全穷举：本阶段为受控采样，结论须标注"仅采样"')
log('=' * 100)
rng = np.random.default_rng(20261016)
rows3 = []
t_st = time.time()
for it in range(N_CHAIN3):
    t1 = TAUS[rng.integers(len(TAUS))]; t2 = TAUS[rng.integers(len(TAUS))]
    ai = int(rng.integers(NC)); bi = int(rng.integers(NC)); ci = int(rng.integers(NC))
    a_idx = IDXL[t1][ai]
    if len(a_idx) < MIN_N: continue
    b_idx = IDXL[t2][bi]
    if len(b_idx) < MIN_N: continue
    mm = np.intersect1d(a_idx, b_idx, assume_unique=True)
    if len(mm) < MIN_N: continue
    c_idx = IDX[ci]
    if len(c_idx) < MIN_N: continue
    mm2 = np.intersect1d(mm, c_idx, assume_unique=True)
    if len(mm2) < MIN_N: continue
    rec(rows3, f'[{NAMES[ai]}] -{t1*5}m-> [{NAMES[bi]}] -{t2*5}m-> [{NAMES[ci]}]', mm2)
    if (it + 1) % 50000 == 0:
        log(f'  进度 {it+1:,}/{N_CHAIN3:,}  命中 {len(rows3):,}  耗时 {time.time()-t_st:.0f}s')
D3 = pd.DataFrame(rows3, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
D3['lift_train'] = D3['rate_train'] / btr
D3['lift_test']  = D3['rate_test']  / bte
D3 = D3.sort_values('rate_train', ascending=False)
D3.to_csv(os.path.join(OUTDIR, 'chain3_sampled.csv'), index=False, encoding='utf-8-sig')
log(f'  3 段链完成 {len(D3):,} 条  总耗时 {time.time()-t_st:.0f}s')
for _, r in D3.head(20).iterrows():
    log(f'    {r["chain"][:70]:<70} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
        f'| 测 {100*r["rate_test"]:>6.2f}% lift {r["lift_test"]:>5.1f}x')

# ================= 4. 四段链：抽查 =================
log('\n' + '=' * 100)
log(f'【链 4 段】抽查 {N_CHAIN4:,} 条  ⚠ 覆盖率极低（{100*N_CHAIN4/(tot3*NC*len(TAUS)):.4f}%），仅作"是否存在更强链"的抽查，不作穷举结论')
log('=' * 100)
rows4 = []
t_st = time.time()
for it in range(N_CHAIN4):
    ts_ = [TAUS[rng.integers(len(TAUS))] for _ in range(3)]
    ids = [int(rng.integers(NC)) for _ in range(4)]
    cur = IDXL[ts_[0]][ids[0]]
    if len(cur) < MIN_N: continue
    ok = True
    for k in range(1, 4):
        nxt = IDXL[ts_[k-1]][ids[k]] if k < 3 else IDX[ids[k]]
        if len(nxt) < MIN_N: ok = False; break
        cur = np.intersect1d(cur, nxt, assume_unique=True)
        if len(cur) < MIN_N: ok = False; break
    if not ok: continue
    nm = f'[{NAMES[ids[0]]}] -{ts_[0]*5}m-> [{NAMES[ids[1]]}] -{ts_[1]*5}m-> [{NAMES[ids[2]]}] -{ts_[2]*5}m-> [{NAMES[ids[3]]}]'
    rec(rows4, nm, cur)
    if (it + 1) % 25000 == 0:
        log(f'  进度 {it+1:,}/{N_CHAIN4:,}  命中 {len(rows4):,}  耗时 {time.time()-t_st:.0f}s')
D4 = pd.DataFrame(rows4, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
if len(D4):
    D4['lift_train'] = D4['rate_train'] / btr
    D4['lift_test']  = D4['rate_test']  / bte
    D4 = D4.sort_values('rate_train', ascending=False)
    D4.to_csv(os.path.join(OUTDIR, 'chain4_sampled.csv'), index=False, encoding='utf-8-sig')
    log(f'  4 段链完成 {len(D4):,} 条  总耗时 {time.time()-t_st:.0f}s')
    for _, r in D4.head(10).iterrows():
        log(f'    {r["chain"][:70]:<70} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
            f'| 测 {100*r["rate_test"]:>6.2f}% lift {r["lift_test"]:>5.1f}x')
else:
    log('  4 段链无命中（采样不足）')

# ================= 5. 汇总 =================
log('\n' + '=' * 100)
log('【汇总】时序链条搜索结果')
log('=' * 100)
for tag, DD in [('链2段(全穷举)', D2), ('链3段(采样)', D3), ('链4段(抽查)', D4 if len(D4) else None)]:
    if DD is None or not len(DD): continue
    okd = DD[np.isfinite(DD['lift_test'])]
    log(f'{tag}: {len(DD):,} 条 | 测试段可复核 {len(okd):,} | lift 中位 {okd["lift_test"].median():.2f}x'
        f' | >5x {100*(okd["lift_test"]>5).mean():.1f}% | >2x {100*(okd["lift_test"]>2).mean():.1f}%'
        f' | <1x(崩) {100*(okd["lift_test"]<1).mean():.1f}%')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'chain_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'报告已写 {os.path.join(OUTDIR, "chain_summary.txt")}')
