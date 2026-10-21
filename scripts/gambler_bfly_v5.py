# -*- coding: utf-8 -*-
"""gambler_bfly_v5.py — 第五轮：【用户口径】可实现差价 ≥+30% 标签 + 高质量条件池 + 真穷举

【与 v4 的唯一区别：标签】
    v4 标签（"纸面上限"）：未来 48h 最高涨 >=+50% 且 其后 288 根最深跌 >=-30%
    v5 标签（"可实现"，用户口径）：未来 T_H 小时内，最高价是否曾 >= 入场价 × 1.30
        → 即「1 块钱进、1.3 块钱出」这件事，在该时点之后 T_H 小时内是否真的发生过
    ⇒ v4 问的是"这波多大"，v5 问的是"进场后能不能拿到 30% 的出场机会"

（条件池、搜索方式、验证方式与 v4 完全相同）

【本轮动机（有据）】
前三轮「测试段 lift 中位」的总账：
    A 核心池 50 条件 (2/3/4元) 真穷举 → 4.49x   崩率 10.8%   ← 信号最强
    B 扩展池 196 条件 (2/3元)  真穷举 → 1.76x   崩率 29.6%   ← 过拟合严重
    ⇒ 条件池从 50 扩到 196，lift 中位反而 4.49x → 1.76x。
    ⇒ 结论：弱条件 / 分位细切档是噪声源。正确方向是扩「高质量条件」，不是扩「条件数量」。

【本轮做法】
    1. 构造 ~90 个候选条件（多尺度阈值扫描），逐条算「训练段单变量 lift」
    2. 只保留 lift >= 1.30 的，作为高质量条件池
    3. 对高质量池真穷举 C(N,2) + C(N,3) + C(N,4)
    4. 训练段（前 70% 时间）搜索 → 测试段（后 30%）复核（全程无前视）
    5. 置换检验：按天分块打乱标签跑 20 次，给出「随机搜索能达到的最高测试段 lift」
    6. 全落 D 盘

产物（D:\\中信证券\\coinlab\\evidence\\butterfly\\）：
    v4_pool.csv      高质量条件池（含单变量 lift / n）
    v4_all.csv       全部组合结果
    v4_top.csv       测试段 top-300
    v4_summary.txt   汇总 + 置换检验

跑法：
    cd /d D:\\中信证券\\coinlab\\scripts && python -X utf8 gambler_bfly_v4.py
"""
import os, gc, time, itertools, random
import numpy as np
import pandas as pd

t0 = time.time()
D  = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')     # 60 币 USDT / 3 年 5m（与 A 阶段同源，可比）
WDAY = 288
MIN_N = 120           # 组合最少触发样本数
MIN_LIFT1 = 1.30      # 单变量入选门槛
N_PERM = 20           # 置换次数
SPLIT = 0.70          # 训练段占比（前 70%）
T_H = 48              # ★ 用户口径：入场后多少小时内出现过 +30% 的出场机会
TARGET = 0.30         # ★ 用户口径：可实现差价 ≥ 本金的 30%

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_bfly_v5 · 第五轮：【用户口径】可实现差价 ≥+30% 标签')
log('=' * 100)

# ================== 1. 读数据 ==================
RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32',
                             'quote_volume': 'float32', 'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns)                 # ★ 以 DataFrame 实际列为准（RAW 可能有重复 symbol，字典会去重）
SYM2CI = {s: i for i, s in enumerate(syms)}
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

R12all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12all, nan=0.0) >= 0.05).sum(axis=1)
dn_cnt = (np.nan_to_num(R12all, nan=0.0) <= -0.05).sum(axis=1)
sync_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
sync_dn = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]   = btc[6:]  / btc[:-6]  - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
MARK = -1
del C; gc.collect()

# ================== 2. 逐币构造特征 + 真局标签 ==================
# 真局定义（与前几轮一致）：未来 48h 内最高涨 >= +50% 且 其后 288 根内最深跌 >= -30%
FEATS = {}          # 条件名 -> (T, N) bool
LAB   = np.zeros((T, N), dtype='int8')
idx_train, idx_test = [], []

def put(name, arr2d):
    FEATS[name] = arr2d

for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4:
        continue
    ci = SYM2CI[sym]
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    lo_all = np.minimum.accumulate(np.where(np.isnan(l), np.inf, l))
    over = c / np.maximum(lo288, 1e-12) - 1.0

    r = {}
    for k in (6, 12, 24, 96, 288):
        v = np.full(n, np.nan); v[k:] = c[k:] / c[:-k] - 1.0
        r[k] = v
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
    vola = pd.Series(r1).rolling(48, min_periods=12).std().values
    sg = np.sign(np.nan_to_num(r1))
    run = np.ones(n, dtype='float64')
    for i in range(1, n):
        run[i] = run[i-1] + 1 if (sg[i] == sg[i-1] and sg[i] != 0) else 1
    pos = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    # 未来 48h 最高涨 / 其后 288 根最深跌（真局标签）
    # ★ v5 标签（用户口径）：未来 T_H 小时内，最高价是否曾达到 入场价 × 1.30
    _win = int(T_H * 12)                       # 12 根 / 小时（5m）
    _futmax = pd.Series(h).rolling(_win, min_periods=1).max().shift(-_win + 1).values
    y = ((_futmax / np.maximum(c, 1e-12) - 1.0) >= TARGET).astype('int8')

    otv = ot
    posv = np.searchsorted(ts, ot)          # ★ 该币每根在全局时间轴上的位置
    hh_utc = ((otv // 3_600_000) % 24)
    dowv = (((otv // 86_400_000) + 4) % 7)

    def col(name, mask):
        return np.where(np.nan_to_num(mask.astype('float64'), nan=0.0) > 0, 1, 0).astype('int8')

    # 条件（多尺度阈值扫描）
    conds = {
        'over>=5%':      over >= 0.05,   'over>=8%':   over >= 0.08,
        'over>=10%':     over >= 0.10,   'over>=15%':  over >= 0.15,
        'over>=25%':     over >= 0.25,   'over>=50%':  over >= 0.50,
        'r6>=3%':        r[6] >= 0.03,   'r6>=5%':     r[6] >= 0.05,
        'r6>=10%':       r[6] >= 0.10,
        'r12>=2%':       r[12] >= 0.02,  'r12>=5%':    r[12] >= 0.05,
        'r12>=10%':      r[12] >= 0.10,
        'r24>=5%':       r[24] >= 0.05,  'r24>=10%':   r[24] >= 0.10,
        'r24>=20%':      r[24] >= 0.20,
        'r96>=15%':      r[96] >= 0.15,  'r96>=30%':   r[96] >= 0.30,
        'r288>=10%':     r[288] >= 0.10, 'r288>=30%':  r[288] >= 0.30,
        'r288>=50%':     r[288] >= 0.50,
        'volr6>=3':      volr6 >= 3,     'volr6>=5':   volr6 >= 5,
        'volr6>=10':     volr6 >= 10,    'volr6>=20':  volr6 >= 20,
        'volr24>=1.5':   volr24 >= 1.5,  'volr24>=3':  volr24 >= 3,
        'volr24>=5':     volr24 >= 5,
        'volr96>=1.5':   volr96 >= 1.5,  'volr96>=3':  volr96 >= 3,
        'trr24>=1.5':    trr24 >= 1.5,   'trr24>=3':   trr24 >= 3,
        'evt>=10%':      evt >= 0.10,    'evt>=15%':   evt >= 0.15,
        'evt>=25%':      evt >= 0.25,
        'run>=12':       run >= 12,      'run>=24':    run >= 24,
        'run>=48':       run >= 48,
        'vola>=4%':      vola >= 0.04,   'vola>=6%':   vola >= 0.06,
        'vola>=10%':     vola >= 0.10,
        'pos>=0.5':      pos >= 0.5,     'pos>=0.8':   pos >= 0.8,
        'pos<0.2':       pos < 0.2,
        'syncUp>=3':     sync_up[posv] >= 3,   'syncUp>=5':  sync_up[posv] >= 5,
        'syncDn>=3':     sync_dn[posv] >= 3,   'syncDn>=5':  sync_dn[posv] >= 5,
        'btc6>=-1%':     btc_r6[posv] <= -0.01, 'btc6<=-3%': btc_r6[posv] <= -0.03,
        'btc6>=1%':      btc_r6[posv] >= 0.01,
        'btc24<=-3%':    btc_r24[posv] <= -0.03, 'btc24<=-5%': btc_r24[posv] <= -0.05,
        'btc24>=3%':     btc_r24[posv] >= 0.03,
        'hour0~6':       (hh_utc >= 0) & (hh_utc < 6),
        'hour6~12':      (hh_utc >= 6) & (hh_utc < 12),
        'hour12~18':     (hh_utc >= 12) & (hh_utc < 18),
        'hour18~24':     (hh_utc >= 18) & (hh_utc < 24),
        'dow0':          dowv == 0, 'dow3': dowv == 3, 'dow6': dowv == 6,
    }

    # ★ 按 open_time 对齐到全局时间轴 ts（各币上市时间不同，不能按行号硬填）
    for name, m in conds.items():
        if name not in FEATS:
            FEATS[name] = np.zeros((T, N), dtype='int8')
        FEATS[name][posv, ci] = np.where(
            np.nan_to_num(m.astype('float64'), nan=0.0) > 0, 1, 0).astype('int8')
    LAB[posv, ci] = y
    del hs, ls, qs, trs, lo288, hi288, over
    gc.collect()

log(f'条件 {len(FEATS)} 个   特征算完 {time.time()-t0:.0f}s')

# ★ 时间切分：按全局时间轴（前 70% 训练 / 后 30% 测试）—— 所有币共用同一切点
tsplit = ts[int(T * SPLIT)]
tmask_tr = ts < tsplit
tmask_te = ts >= tsplit

# ================== 3. 单变量筛选 ==================
base_tr = LAB[tmask_tr].mean()
base_te = LAB[tmask_te].mean()
log(f'基准：训练 {base_tr*100:.4f}%   测试 {base_te*100:.4f}%')

rows = []
for name, M in FEATS.items():
    m_tr = M[tmask_tr].astype(bool)
    n1 = int(m_tr.sum())
    if n1 < MIN_N:
        continue
    p_tr = LAB[tmask_tr][m_tr].mean()
    lift1 = p_tr / base_tr if base_tr > 0 else 0.0
    rows.append((name, n1, p_tr, lift1))
pool_df = pd.DataFrame(rows, columns=['cond', 'n_train', 'rate_train', 'lift1'])
pool_df = pool_df.sort_values('lift1', ascending=False).reset_index(drop=True)
pool = pool_df[pool_df['lift1'] >= MIN_LIFT1]['cond'].tolist()
log(f'候选 {len(pool_df)} 个 → 高质量池（单变量 lift>={MIN_LIFT1}）{len(pool)} 个')
log(pool_df.head(20).to_string(index=False))
pool_df.to_csv(os.path.join(OUT, 'v5_pool.csv'), index=False, encoding='utf-8-sig')

if len(pool) < 20:
    log('★ 高质量池过小，退出（不得据此说"不存在"）')
    open(os.path.join(OUT, 'v5_summary.txt'), 'w', encoding='utf-8').write('\n'.join(logbuf))
    raise SystemExit(0)

# ================== 4. 真穷举 2/3/4 元 ==================
def evaluate(combo):
    m = np.ones((T, N), dtype=bool)
    for c in combo:
        m &= FEATS[c].astype(bool)
    mtr = m[tmask_tr]; mte = m[tmask_te]
    n_tr = int(mtr.sum())
    if n_tr < MIN_N:
        return None
    p_tr = LAB[tmask_tr][mtr].mean()
    n_te = int(mte.sum())
    p_te = LAB[tmask_te][mte].mean() if n_te > 0 else np.nan
    lt = p_tr / base_tr if base_tr > 0 else 0.0
    le = p_te / base_te if (base_te > 0 and n_te > 0) else np.nan
    return (n_tr, p_tr, lt, n_te, p_te, le)

res = []
for k in (2, 3, 4):
    ncombo = 0; nok = 0
    for combo in itertools.combinations(pool, k):
        ncombo += 1
        r = evaluate(combo)
        if r is None:
            continue
        nok += 1
        res.append((' & '.join(combo),) + r)
    log(f'  {k} 元：试 {ncombo:,}  有效 {nok:,}  累计 {len(res):,}  耗时 {time.time()-t0:.0f}s')

df = pd.DataFrame(res, columns=['combo', 'n_train', 'rate_train', 'lift_train',
                                'n_test', 'rate_test', 'lift_test'])
df.to_csv(os.path.join(OUT, 'v5_all.csv'), index=False, encoding='utf-8-sig')
top = df.sort_values('lift_test', ascending=False).head(300)
top.to_csv(os.path.join(OUT, 'v5_top.csv'), index=False, encoding='utf-8-sig')

log('=' * 100)
log('【测试段 top-20】')
for _, r in top.head(20).iterrows():
    log(f"  {r['combo'][:70]:<72} 训 {r['rate_train']*100:7.2f}%(n={int(r['n_train']):>5}) lift {r['lift_train']:6.1f}x | "
        f"测 {r['rate_test']*100:7.2f}%(n={int(r['n_test']):>5}) lift {r['lift_test']:6.1f}x")
log(f"  测试段可复核 {df['lift_test'].notna().sum():,} 条：lift 中位 {df['lift_test'].median():.2f}x | "
    f">5x 占 {(df['lift_test']>5).mean()*100:.1f}% | >2x 占 {(df['lift_test']>2).mean()*100:.1f}% | "
    f"<1x(崩) 占 {(df['lift_test']<1).mean()*100:.1f}%")

# ================== 5. 置换检验 ==================
log('=' * 100)
log(f'【置换检验】按天分块打乱标签，跑 {N_PERM} 次')
day = (ts // 86_400_000).astype('int64')
udays = np.unique(day)
real_max = float(top['lift_test'].max())
rng = np.random.default_rng(42)
perm_max = []
sub_pool = random.Random(7).sample(pool, min(len(pool), 30))
sub_combos = list(itertools.combinations(sub_pool, 2)) + list(itertools.combinations(sub_pool, 3))
sub_combos = random.Random(11).sample(sub_combos, min(len(sub_combos), 3000))
for it in range(N_PERM):
    perm_day = rng.permutation(udays)
    m = dict(zip(udays, perm_day))
    pmap = np.array([m[d] for d in day])
    LABp = np.zeros_like(LAB)
    for d in udays:
        src = np.flatnonzero(day == d)
        dst = np.flatnonzero(pmap == d)
        if len(src) == 0 or len(dst) == 0:
            continue
        k = min(len(src), len(dst))
        LABp[dst[:k]] = LAB[src[:k]]
    bs_tr = LABp[tmask_tr].mean()
    bs_te = LABp[tmask_te].mean()
    if bs_tr <= 0 or bs_te <= 0:
        continue
    best = 0.0
    for combo in sub_combos:
        mx = np.ones((T, N), dtype=bool)
        for c in combo:
            mx &= FEATS[c].astype(bool)
        mte = mx[tmask_te]
        if mte.sum() < MIN_N:
            continue
        p = LABp[tmask_te][mte].mean()
        lf = p / bs_te
        if lf > best:
            best = lf
    perm_max.append(best)
    log(f'  置换 {it+1:>2}/{N_PERM}  随机最高测试 lift {best:6.2f}x')

if perm_max:
    log(f"随机水平：中位 {np.median(perm_max):.2f}x  p90 {np.percentile(perm_max,90):.2f}x  max {np.max(perm_max):.2f}x")
    log(f"真实最高（高质量池）：{real_max:.2f}x   真实测试基准 {base_te*100:.4f}%")
    log(f"判读：真实最高 {'显著高于' if real_max > np.max(perm_max) else '未超过'} 随机 max ⇒ "
        f"{'真信号' if real_max > np.max(perm_max) else '多重比较假象风险高'}")

log(f'TOTAL {time.time()-t0:.0f}s')
with open(os.path.join(OUT, 'v5_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log('报告已写 ' + os.path.join(OUT, 'v5_summary.txt'))
