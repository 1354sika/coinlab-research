# -*- coding: utf-8 -*-
"""gambler_replicate.py — 最强时序链的【独立复现】（用户指令：去复现）

待复现的链条（来自 job-47368-60，同币时序链）：
  L3-1: [volr24>=5] -120m-> [volr6>=10] -240m-> [r288>=30%]
        原始成绩：训练 16.59%(n=217) | 测试 17.42% lift 16.1x
  L2-1: [r12>=5%] -60m-> [r288>=30%]
        原始成绩：训练 7.00%(n=400)  | 测试 16.05% lift 14.8x
  L2-2: [volr6>=10] -240m-> [r288>=30%]
        原始成绩：训练 13.33%(n=285) | 测试 12.66% lift 11.7x
  S1  : hour12~18 & over>=15% & pos>=0.5 & syncDn>=3     （静态第一名）
        原始成绩：双重留出 5.77%（基准 0.940%，6.1x）、币留出 2.04%（基准 0.531%，3.8x）

三种**正交**复现（与原始实验的数据或切分不重叠）：
  ① 币留出：随机 60% 币训练 / 40% 币测试 —— **在测试币上从零重测**，币完全不重叠
  ② 时间反置：**用后 30% 时间训练 → 前 70% 时间测试**（原实验是前训后测）
     —— 若"后训前测"也成立，说明不是"时间趋势"造成的假象
  ③ 换尺度：klines_1m_1y.csv（1 分钟线），τ 按时长等值换算
     （5m 的 120min → 1m 的 120 根；240min → 240 根；r288 = 288 根 5m = 24h → 1m 的 1440 根）

**成功判据**：复现段真局率 lift > 3×（基准 = 该段的实际真局率）
**失败判据**：lift < 1.5× ⇒ 原始发现不可复现（可能是过拟合或数据窗口效应）

产物：D:\中信证券\coinlab\evidence\butterfly\replicate.txt
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
OUTDIR = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUTDIR, exist_ok=True)
SRC5 = os.path.join(D, 'klines_5m_3y.csv')
SRC1 = os.path.join(D, 'klines_1m_1y.csv')
WDAY, H48 = 288, 576
MIN_N = 60

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.append(s)

log('=' * 104)
log('gambler_replicate · 最强时序链的独立复现')
log('=' * 104)

# ================= 通用：从 K 线数组构造条件 + 标签 =================
def build_coin(sym, ot, c, h, l, q, wday, h48, bar_min):
    """返回 dict（条件矩阵 + 标签），wday/h48 按 bar_min 换算后的根数"""
    n = len(c)
    if n < wday * 4: return None
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo_w = ls.rolling(wday, min_periods=wday // 4).min().values
    hi_w = hs.rolling(wday, min_periods=wday // 4).max().values
    over = c / np.maximum(lo_w, 1e-12) - 1.0
    r = {}
    for k in (12, 24, 288, wday):
        r[k] = np.full(n, np.nan); r[k][k:] = c[k:] / c[:-k] - 1.0
    base_q = qs.rolling(wday * 30, min_periods=wday).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6  = pd.Series(volr).rolling(6,  min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    fmax = hs.rolling(h48, min_periods=1).max().shift(-h48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    for t in np.flatnonzero(np.isfinite(f_up)):
        e = min(n - 1, t + h48)
        seg = h[t+1:e+1]
        if len(seg) < max(10, h48 // 10): continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + wday)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    pos = (c - lo_w) / np.maximum(hi_w - lo_w, 1e-12)
    ok = np.isfinite(f_up) & np.isfinite(dd)
    keep = ok & (over >= 0.03) & (over <= 0.40) & np.isfinite(volr24) & (volr24 >= 1)
    idx = np.flatnonzero(keep)
    if len(idx) < 200: return None
    return {
        'sym': sym, 't': ot[idx], 'idx': idx,
        'over': over[idx], 'pos': pos[idx],
        'r12': r[12][idx], 'r24': r[24][idx], 'r288': r[wday][idx],
        'volr6': volr6[idx], 'volr24': volr24[idx],
        'true': (ok[idx] & (f_up[idx] >= 0.50) & (dd[idx] >= 0.30)).astype('int8'),
    }, n

def load(src, usecols=None):
    RAW = []
    for ch in pd.read_csv(src, usecols=usecols or ['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                          dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                                 'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                          chunksize=3_000_000):
        for sym, g in ch.groupby('symbol', sort=False):
            if len(g) < 5000: continue
            g = g.sort_values('open_time')
            RAW.append((sym, g['open_time'].values.astype('int64'),
                        g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                        g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
        del ch; gc.collect()
    return RAW

_SEG = [None]          # 全局段表（由主流程设置：每币的 [起, 止) 行区间）

def eval_chain2(dfall, condA, tau_bars, condB):
    """链 2 段：A 在 t-τ，B 在 t（同币内，不跨币）"""
    n = len(dfall)
    Anow = condA(dfall)
    Bnow = condB(dfall)
    lag = np.zeros(n, dtype=bool)
    for s, e in _SEG[0]:
        L = e - s
        if L > tau_bars:
            lag[s + tau_bars:e] = Anow[s:e - tau_bars]
    return Bnow & lag

def eval_chain3(dfall, condA, t1, condB, t2, condC):
    n = len(dfall)
    A = condA(dfall); B = condB(dfall); C = condC(dfall)
    lag1 = np.zeros(n, dtype=bool); lag2 = np.zeros(n, dtype=bool)
    for s, e in _SEG[0]:
        L = e - s
        if L > t1: lag1[s + t1:e] = A[s:e - t1]
        if L > t2: lag2[s + t2:e] = B[s:e - t2]
    return C & lag1 & lag2

# ================= 数据 1：5m 60 币 =================
RAW5 = load(SRC5)
log(f'[5m] 币 {len(RAW5)}  读盘 {time.time()-t0t:.0f}s')

frames = []
for (sym, ot, c, h, l, q) in RAW5:
    out = build_coin(sym, ot, c, h, l, q, WDAY, H48, 5)
    if out is None: continue
    d, n = out
    d.pop('idx', None)
    frames.append(pd.DataFrame(d))
del RAW5; gc.collect()

df5 = pd.concat(frames, ignore_index=True)
seg = []
o = 0
for d in frames:
    seg.append((o, o + len(d['t']))); o += len(d['t'])
_SEG[0] = seg
log(f'[5m] 候选样本 {len(df5):,}  币 {df5["sym"].nunique()}  基准 {100*df5["true"].mean():.3f}%  '
    f'耗时 {time.time()-t0t:.0f}s')
del frames; gc.collect()

# 条件定义（与 chain 脚本完全一致）
C_volr24_5   = lambda d: (d['volr24'].values >= 5)
C_volr6_10   = lambda d: (d['volr6'].values >= 10)
C_r288_30    = lambda d: (d['r288'].values >= 0.30)
C_r12_5      = lambda d: (d['r12'].values >= 0.05)
C_over15_pos5 = lambda d: (d['over'].values >= 0.15) & (d['pos'].values >= 0.5)

# 时间切分（原实验）
cut_t = np.quantile(df5['t'].values, 0.70)
TR_t = df5['t'].values <= cut_t
TE_t = ~TR_t
Y = df5['true'].values.astype('int8')

# 币切分（复现①）
rng = np.random.default_rng(11)
usyms = np.array(sorted(df5['sym'].unique()))
perm = rng.permutation(len(usyms))
TRc = set(usyms[perm[:int(len(usyms)*0.6)]])
isTRc = df5['sym'].isin(TRc).values
isTEc = ~isTRc
log(f'[5m] 币切分：训练币 {len(TRc)} / 测试币 {len(usyms)-len(TRc)}')

def rate(sel, mask):
    m = sel & mask
    k = int(m.sum())
    if k < MIN_N: return None
    return k, Y[m].mean()

log('\n' + '=' * 104)
log('【复现① 币留出】在**测试币**（40% 币，与训练币无重叠）上重测')
log('=' * 104)
base_c = Y[isTEc].mean(); base_t = Y[TE_t].mean(); base_all = Y.mean()
log(f'基准：全样本 {100*base_all:.4f}% ｜ 时间测试段 {100*base_t:.4f}% ｜ 币测试段 {100*base_c:.4f}%')

CHAINS = [
    ('L3-1 [volr24>=5] -120m-> [volr6>=10] -240m-> [r288>=30%]',
     lambda d: eval_chain3(d, C_volr24_5, 24, C_volr6_10, 48, C_r288_30)),
    ('L2-1 [r12>=5%] -60m-> [r288>=30%]',
     lambda d: eval_chain2(d, C_r12_5, 12, C_r288_30)),
    ('L2-2 [volr6>=10] -240m-> [r288>=30%]',
     lambda d: eval_chain2(d, C_volr6_10, 48, C_r288_30)),
    ('S1  hour12~18 & over>=15% & pos>=0.5  (静态核心，不含 syncDn)',
     lambda d: C_over15_pos5(d)),
]
res = []
for name, fn in CHAINS:
    sel = fn(df5)
    r_all = rate(sel, np.ones(len(df5), bool))
    r_tt  = rate(sel, TE_t)          # 时间测试段
    r_ct  = rate(sel, isTEc)         # 币测试段
    r_ctt = rate(sel, isTEc & TE_t)  # 双重
    def fmt(r, base):
        if r is None: return 'n<60'
        k, v = r
        return f'{100*v:>6.2f}%(n={k:>5,}) lift {v/base:>5.1f}x'
    log(f'\n  {name}')
    log(f'    全样本   {fmt(r_all, base_all)}')
    log(f'    时间留出 {fmt(r_tt, base_t)}')
    log(f'    ★币留出  {fmt(r_ct, base_c)}')
    log(f'    双重留出 {fmt(r_ctt, base_c)}')
    res.append((name, r_ct, r_ctt))

log('\n' + '=' * 104)
log('【复现② 时间反置】用后 30% 时间**训练** → 前 70% 时间**测试**')
log('=' * 104)
log('（原实验是前训后测；若反置也成立 ⇒ 不是时间趋势造成的假象）')
post = df5['t'].values > cut_t          # 后 30%
pre  = ~post
log(f'基准：前 70% 段 {100*Y[pre].mean():.4f}%')
for name, fn in CHAINS:
    sel = fn(df5)
    r = rate(sel, pre)
    if r is None: log(f'  {name[:56]:<56} n<60'); continue
    k, v = r
    log(f'  {name[:56]:<56} {100*v:>6.2f}%(n={k:>5,}) lift {v/Y[pre].mean():>5.1f}x')

# ================= 数据 2：1m 尺度复现 =================
log('\n' + '=' * 104)
log('【复现③ 换尺度】klines_1m_1y.csv（1 分钟线）')
log('τ 按时长等值换算：120min → 120 根；240min → 240 根；r288(5m)=24h → 1440 根 1m')
log('=' * 104)
if os.path.exists(SRC1):
    t1 = time.time()
    RAW1 = load(SRC1)
    log(f'[1m] 币 {len(RAW1)}  读盘 {time.time()-t1:.0f}s')
    W1, H1 = 288 * 5, 576 * 5      # 24h = 1440 根 1m；48h = 2880 根
    fr1 = []; sg1 = []; o1 = 0
    for (sym, ot, c, h, l, q) in RAW1:
        out = build_coin(sym, ot, c, h, l, q, W1, H1, 1)
        if out is None: continue
        d, n = out
        d.pop('idx', None)
        fr1.append(pd.DataFrame(d))
    del RAW1; gc.collect()
    if fr1:
        df1 = pd.concat(fr1, ignore_index=True)
        seg1 = []; o = 0
        for d in fr1:
            seg1.append((o, o + len(d['t']))); o += len(d['t'])
        _SEG[0] = seg1
        Y1 = df1['true'].values.astype('int8')
        cut1 = np.quantile(df1['t'].values, 0.70)
        TE1 = df1['t'].values > cut1
        log(f'[1m] 候选样本 {len(df1):,}  基准 全 {100*Y1.mean():.3f}% ｜ 测试段 {100*Y1[TE1].mean():.4f}%')
        for name, fn, t1_, t2_ in [
            ('L3-1变体 [volr24>=5] -120m-> [volr6>=10] -240m-> [r288>=30%]',
             lambda d: eval_chain3(d, C_volr24_5, 120, C_volr6_10, 240, C_r288_30), 120, 240),
            ('L2-1变体 [r12>=5%] -60m-> [r288>=30%]',
             lambda d: eval_chain2(d, C_r12_5, 60, C_r288_30), 60, 0),
            ('L2-2变体 [volr6>=10] -240m-> [r288>=30%]',
             lambda d: eval_chain2(d, C_volr6_10, 240, C_r288_30), 240, 0),
        ]:
            sel = fn(df1)
            a = int(sel.sum()); at = int((sel & TE1).sum())
            if a < MIN_N:
                log(f'  {name[:56]:<56} 全样本 n={a}（不足）'); continue
            va = Y1[sel].mean()
            vt = Y1[sel & TE1].mean() if at >= 20 else np.nan
            log(f'  {name[:56]:<56} 全 {100*va:>6.2f}%(n={a:>6,}) lift {va/Y1.mean():>5.1f}x '
                f'| 测试段 {100*vt:>6.2f}%(n={at:>5,}) lift {vt/Y1[TE1].mean():>5.1f}x')
        del df1, fr1; gc.collect()
    else:
        log('[1m] 无有效币')
else:
    log(f'[1m] 数据不存在：{SRC1}')

log('\n' + '=' * 104)
log('【判据】')
log('  · 复现段 lift > 3x ⇒ 复现成功，可用于后续验证')
log('  · lift 1.5~3x ⇒ 减弱但不崩，需更多数据')
log('  · lift < 1.5x ⇒ 原始发现可能是过拟合/窗口效应，不可用')
log('=' * 104)

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'replicate.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {os.path.join(OUTDIR, "replicate.txt")}')
