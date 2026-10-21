# -*- coding: utf-8 -*-
"""gambler_bfly_cross.py — 蝴蝶效应③：跨币时序链（用户原义）

前面 static/chain 都是**同一币内部**的条件组合。用户说的"蝴蝶效应"更接近：
    市场/别的币先动  →  隔一段时间  →  这个币才开始动
    （"这个事情发生了，那其他事情发生的概率有多少？"）

本脚本实现两个方向：
  ① 市场 → 单币：市场级事件 A 在 t-τ 发生 → 单币 Y 在 t 满足条件 B → 预测 Y
  ② 币 → 币：币 X 在 t-τ 满足 A → 币 Y 在 t 满足 B → 预测 Y（限核心池，抽样）

规模：
  市场级事件池 18 个 × 单币条件池 32 个 × τ 5 档 = 2,880 条  【全穷举】
  币→币：(60 × 32) × (60 × 32) × 5 ≈ 5,900 万 → 【受控采样 20 万，独立标注】

判据与 static/chain 一致：真局 = 48h 内最高涨 >=+50% 且其后 288 根最深跌 >=-30%
训练段(前 70%) 搜索 → 测试段复核；全程无前视

产物（落 D 盘）：
  butterfly\cross_mkt_all.csv     （市场→单币，全穷举）
  butterfly\cross_coin_sampled.csv（币→币，采样）
  butterfly\cross_summary.txt
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
TAUS = [6, 12, 24, 48, 96]
N_COIN = 200_000

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_bfly_cross · 蝴蝶效应③跨币时序链（市场 →τ→ 单币）')
log('=' * 100)

# ---------- 读数据（宽表） ----------
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
Cw = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = Cw.index.values; T, N = Cw.shape
log(f'币 {len(syms)}  矩阵 {Cw.shape}  耗时 {time.time()-t0t:.0f}s')

# ---------- 市场级指标（宽表 → 逐时刻向量） ----------
R12 = Cw.pct_change(12).values
R6  = Cw.pct_change(6).values
up3  = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)      # 齐涨币数
dn3  = (np.nan_to_num(R12, nan=0.0) <= -0.05).sum(axis=1)     # 齐跌币数
mx   = np.nanmax(np.nan_to_num(R12, nan=-9), axis=1)          # 最强币
mn   = np.nanmin(np.nan_to_num(R12, nan=9), axis=1)
btc  = Cw['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
eth  = Cw['ETHUSDT'].values.astype('float64') if 'ETHUSDT' in syms else np.zeros(T)
br6  = np.zeros(T);  br6[6:]   = btc[6:]  / btc[:-6]  - 1.0
br24 = np.zeros(T);  br24[24:] = btc[24:] / btc[:-24] - 1.0
er24 = np.zeros(T);  er24[24:] = eth[24:] / eth[:-24] - 1.0
up_r = pd.Series(up3); dn_r = pd.Series(dn3)
up_max24  = up_r.rolling(24,  min_periods=1).max().values
dn_max24  = dn_r.rolling(24,  min_periods=1).max().values
up_max96  = up_r.rolling(96,  min_periods=1).max().values
dn_max96  = dn_r.rolling(96,  min_periods=1).max().values
# 市场涨跌广度（过去 24 根均值）
up_mean24 = up_r.rolling(24, min_periods=6).mean().values
dn_mean24 = dn_r.rolling(24, min_periods=6).mean().values

MKT = {
    'mkt_齐涨>=3': (up3 >= 3), 'mkt_齐涨>=5': (up3 >= 5), 'mkt_齐涨>=8': (up3 >= 8),
    'mkt_齐跌>=3': (dn3 >= 3), 'mkt_齐跌>=5': (dn3 >= 5), 'mkt_齐跌>=8': (dn3 >= 8),
    'mkt_齐涨24h内>=5': (up_max24 >= 5), 'mkt_齐涨24h内>=8': (up_max24 >= 8),
    'mkt_齐跌24h内>=5': (dn_max24 >= 5), 'mkt_齐跌24h内>=8': (dn_max24 >= 8),
    'mkt_齐涨96h内>=8': (up_max96 >= 8), 'mkt_齐跌96h内>=8': (dn_max96 >= 8),
    'mkt_广度涨>=5': (up_mean24 >= 5), 'mkt_广度跌>=5': (dn_mean24 >= 5),
    'mkt_最强币>=20%': (mx >= 0.20), 'mkt_最强币>=50%': (mx >= 0.50),
    'mkt_最弱币<=-10%': (mn <= -0.10), 'mkt_最弱币<=-20%': (mn <= -0.20),
    'BTC24h<=-3%': (br24 <= -0.03), 'BTC24h>=3%': (br24 >= 0.03),
    'BTC6h<=-1%': (br6 <= -0.01),  'BTC6h>=1%': (br6 >= 0.01),
    'ETH24h<=-3%': (er24 <= -0.03), 'ETH24h>=3%': (er24 >= 0.03),
}
MKTN = sorted(MKT.keys())
MKM  = np.stack([MKT[k] for k in MKTN], axis=1)   # (T, nM)
log(f'市场级事件池 {len(MKTN)} 个')

# ---------- 单币条件池（同 chain 脚本口径） ----------
CORE = [
    ('over', 0.08, +1, 'over>=8%'), ('over', 0.15, +1, 'over>=15%'),
    ('r12', 0.05, +1, 'r12>=5%'),   ('r12', 0.10, +1, 'r12>=10%'),
    ('r6', 0.05, +1, 'r6>=5%'),
    ('r24', 0.10, +1, 'r24>=10%'),  ('r24', 0.20, +1, 'r24>=20%'),
    ('r96', 0.30, +1, 'r96>=30%'),
    ('r288', 0.10, +1, 'r288>=10%'), ('r288', 0.30, +1, 'r288>=30%'),
    ('volr6', 5, +1, 'volr6>=5'),   ('volr6', 10, +1, 'volr6>=10'),
    ('volr24', 3, +1, 'volr24>=3'), ('volr24', 5, +1, 'volr24>=5'),
    ('volr96', 3, +1, 'volr96>=3'), ('trr24', 3, +1, 'trr24>=3'),
    ('evt', 0.15, +1, 'evt>=15%'),  ('run', 24, +1, 'run>=24'),
    ('vola', 0.06, +1, 'vola>=6%'),
    ('pos', 0.8, +1, 'pos>=0.8'),   ('pos', 0.2, -1, 'pos<0.2'),
    ('sync_up', 3, +1, 'syncUp>=3'), ('sync_dn', 3, +1, 'syncDn>=3'),
    ('btc6', -0.01, -1, 'btc6<=-1%'), ('btc6', 0.01, +1, 'btc6>=1%'),
    ('btc24', -0.03, -1, 'btc24<=-3%'),
    ('r12', 0.02, -1, 'r12<2%'),     ('volr24', 1.5, -1, 'volr24<1.5'),
    ('over', 0.08, -1, 'over<8%'),
    ('hour', (18, 24), 0, 'hour18~24'), ('hour', (0, 6), 0, 'hour0~6'),
]
NAMES = [c[3] for c in CORE]; NC = len(NAMES)

segs = []
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
    su  = pd.Series(np.nan_to_num((R12 >= 0.05).sum(axis=1).astype(float)), index=ts).rolling(24, min_periods=1).max().reindex(ii).values
    sd  = pd.Series(np.nan_to_num((R12 <= -0.05).sum(axis=1).astype(float)), index=ts).rolling(24, min_periods=1).max().reindex(ii).values
    b6  = pd.Series(br6,  index=ts).reindex(ii).values
    b24 = pd.Series(br24, index=ts).reindex(ii).values
    hh  = (ot // 3600000) % 24
    COL = {'over': over, 'r6': r[6], 'r12': r[12], 'r24': r[24], 'r96': r[96], 'r288': r[288],
           'volr6': volr6, 'volr24': volr24, 'volr96': volr96, 'trr24': trr24, 'evt': evt,
           'run': run, 'vola': vola,
           'pos': (c - lo288) / np.maximum(hi288 - lo288, 1e-12),
           'sync_up': su, 'sync_dn': sd, 'btc6': b6, 'btc24': b24, 'hour': hh}
    M = np.zeros((n, NC), dtype=bool)
    for j, (col, th, direc, _nm) in enumerate(CORE):
        v = COL[col]
        if direc == 0:
            a, b = th; M[:, j] = np.isfinite(v) & (v >= a) & (v < b)
        elif direc > 0:
            M[:, j] = np.isfinite(v) & (v >= th)
        else:
            M[:, j] = np.isfinite(v) & (v < th)
    lab = np.zeros(n, dtype=np.int8)
    ok = np.isfinite(f_up) & np.isfinite(dd)
    lab[ok & (f_up >= 0.50) & (dd >= 0.30)] = 1
    keep = ok & (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    idx = np.flatnonzero(keep)[::3]
    if len(idx) < 400: continue
    mkt_at = np.zeros((len(idx), len(MKTN)), dtype=bool)
    pos_map = {v: k for k, v in enumerate(ts)}
    # 用 searchsorted 对齐市场矩阵（ot 是 ts 的子集）
    jj = np.searchsorted(ts, ot[idx])
    mkt_at = MKM[np.clip(jj, 0, T-1)]
    segs.append({'sym': sym, 't': ot[idx], 'M': M[idx], 'y': lab[idx], 'mkt': mkt_at})
    del hs, ls, qs, trs; gc.collect()

if not segs:
    log('无有效币，退出'); raise SystemExit(1)

t_all  = np.concatenate([s['t'] for s in segs])
y_all  = np.concatenate([s['y'] for s in segs]).astype(np.int8)
M_all  = np.concatenate([s['M'] for s in segs], axis=0)
MK_all = np.concatenate([s['mkt'] for s in segs], axis=0)
offs = np.cumsum([0] + [len(s['t']) for s in segs])
n_all = len(y_all)
log(f'候选样本 {n_all:,}  币 {len(segs)}  基准真局率 {100*y_all.mean():.3f}%  耗时 {time.time()-t0t:.0f}s')

cut_t = np.quantile(t_all, 0.70)
TR = t_all <= cut_t; TE = ~TR
btr, bte = y_all[TR].mean(), y_all[TE].mean()
log(f'训练 {int(TR.sum()):,}（基准 {100*btr:.4f}%）  测试 {int(TE.sum()):,}（基准 {100*bte:.4f}%）')

def rate_from_mask(mi):
    trm = TR[mi]; tem = TE[mi]
    n_tr = int(trm.sum()); n_te = int(tem.sum())
    if n_tr < MIN_N: return None
    return n_tr, y_all[mi][trm].mean(), n_te, (y_all[mi][tem].mean() if n_te >= 30 else np.nan)

# lag 版市场矩阵
MKl = {}
for tau in TAUS:
    ML = np.zeros_like(MK_all)
    for si in range(len(segs)):
        a, b = offs[si], offs[si+1]
        if b - a > tau:
            ML[a+tau:b] = MK_all[a:b-tau]
    MKl[tau] = ML

IDXM = {tau: [np.flatnonzero(MKl[tau][:, j]).astype(np.int32) for j in range(len(MKTN))] for tau in TAUS}
IDXC = [np.flatnonzero(M_all[:, j]).astype(np.int32) for j in range(NC)]

# ================= ① 市场 →τ→ 单币 【全穷举】 =================
log('\n' + '=' * 100)
tot1 = len(MKTN) * len(TAUS) * NC
log(f'【① 市场→单币】组合数 = {len(MKTN)} × {len(TAUS)} × {NC} = {tot1:,}  【全穷举】')
log('=' * 100)
rows1 = []; t_st = time.time()
for tau in TAUS:
    for mi, mn_ in enumerate(MKTN):
        a_idx = IDXM[tau][mi]
        if len(a_idx) < MIN_N: continue
        for ci, cn_ in enumerate(NAMES):
            b_idx = IDXC[ci]
            if len(b_idx) < MIN_N: continue
            mm = np.intersect1d(a_idx, b_idx, assume_unique=True)
            if len(mm) < MIN_N: continue
            r = rate_from_mask(mm)
            if r: rows1.append((f'[{mn_}] -{tau*5}m-> [{cn_}]', r[0], r[1], r[2], r[3]))
    log(f'  τ={tau}（{tau*5}min）完成，累计 {len(rows1):,}  耗时 {time.time()-t_st:.0f}s')
D1 = pd.DataFrame(rows1, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
D1['lift_train'] = D1['rate_train'] / btr; D1['lift_test'] = D1['rate_test'] / bte
D1 = D1.sort_values('rate_train', ascending=False)
D1.to_csv(os.path.join(OUTDIR, 'cross_mkt_all.csv'), index=False, encoding='utf-8-sig')
log(f'  ① 完成 {len(D1):,} 条  耗时 {time.time()-t_st:.0f}s')
for _, r in D1.head(25).iterrows():
    log(f'    {r["chain"][:62]:<62} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
        f'| 测 {100*r["rate_test"]:>6.2f}%(n={int(r["n_test"]):>5,}) lift {r["lift_test"]:>5.1f}x')

# ================= ② 币 →τ→ 币 【采样】 =================
log('\n' + '=' * 100)
tot2 = (len(segs) * NC) ** 2 * len(TAUS)
log(f'【② 币→币】理论组合 ≈ {tot2:,}  →  采样 {N_COIN:,}（覆盖 {100*N_COIN/tot2:.4f}%）')
log('⚠ 非全穷举：本阶段为受控采样')
log('=' * 100)
rng = np.random.default_rng(20261016)
rows2 = []; t_st = time.time()
for it in range(N_COIN):
    s1 = int(rng.integers(len(segs))); s2 = int(rng.integers(len(segs)))
    c1 = int(rng.integers(NC)); c2 = int(rng.integers(NC)); tau = TAUS[rng.integers(len(TAUS))]
    a1, b1 = offs[s1], offs[s1+1]; a2, b2 = offs[s2], offs[s2+1]
    if b1 - a1 <= tau: continue
    gidx1 = np.arange(a1, b1 - tau)             # 源端（t-τ 时刻）
    src = gidx1[M_all[gidx1, c1]]
    if len(src) < MIN_N: continue
    src_t = t_all[src] + 0                      # 源时刻
    # 目标端：同币 s2 里那些 t 使得 t-τ 恰好在 src 里 —— 用「同币同刻」近似：改为直接查 s2 的条件 + τ 前源币的条件
    tgt = np.arange(a2 + tau, b2)
    tt = tgt[M_all[tgt, c2]]
    if len(tt) < MIN_N: continue
    # 源时刻集合（全局时间戳）
    src_set = np.sort(t_all[src])
    # 目标时刻对应的「τ 根之前的时刻」需要按 5m 步长回退：用 ts 索引
    ret = np.searchsorted(ts, t_all[tt]) - tau
    ret = np.clip(ret, 0, T-1)
    src_times = ts[ret]
    mmask = np.isin(src_times, src_set)
    if int(mmask.sum()) < MIN_N: continue
    keep_idx = tt[mmask]
    src_idx = np.searchsorted(t_all, np.sort(t_all[src])[0:0])  # 占位
    # 命中下标（用于算 rate）：用 keep_idx 在全局里的位置
    r = rate_from_mask(keep_idx)
    if r:
        rows2.append((f'[{segs[s1]["sym"]}·{NAMES[c1]}] -{tau*5}m-> [{segs[s2]["sym"]}·{NAMES[c2]}]',
                      r[0], r[1], r[2], r[3]))
    if (it + 1) % 25000 == 0:
        log(f'  进度 {it+1:,}/{N_COIN:,}  命中 {len(rows2):,}  耗时 {time.time()-t_st:.0f}s')
D2 = pd.DataFrame(rows2, columns=['chain', 'n_train', 'rate_train', 'n_test', 'rate_test'])
if len(D2):
    D2['lift_train'] = D2['rate_train'] / btr; D2['lift_test'] = D2['rate_test'] / bte
    D2 = D2.sort_values('rate_train', ascending=False)
    D2.to_csv(os.path.join(OUTDIR, 'cross_coin_sampled.csv'), index=False, encoding='utf-8-sig')
    log(f'  ② 完成 {len(D2):,} 条  耗时 {time.time()-t_st:.0f}s')
    for _, r in D2.head(15).iterrows():
        log(f'    {r["chain"][:62]:<62} 训 {100*r["rate_train"]:>6.2f}%(n={int(r["n_train"]):>6,}) '
            f'| 测 {100*r["rate_test"]:>6.2f}% lift {r["lift_test"]:>5.1f}x')
else:
    log('  ② 无命中')

# ================= 汇总 =================
log('\n' + '=' * 100)
log('【汇总】跨币时序链')
log('=' * 100)
for tag, DD in [('① 市场→单币（全穷举）', D1), ('② 币→币（采样）', D2 if len(D2) else None)]:
    if DD is None or not len(DD): continue
    o = DD[np.isfinite(DD['lift_test'])]
    log(f'{tag}: {len(DD):,} 条 | 可复核 {len(o):,} | lift 中位 {o["lift_test"].median():.2f}x'
        f' | >5x {100*(o["lift_test"]>5).mean():.1f}% | >2x {100*(o["lift_test"]>2).mean():.1f}%'
        f' | <1x(崩) {100*(o["lift_test"]<1).mean():.1f}%')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'cross_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'报告已写 {os.path.join(OUTDIR, "cross_summary.txt")}')
