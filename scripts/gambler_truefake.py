# -*- coding: utf-8 -*-
"""gambler_truefake.py — 局的真假分层：模型抓到的到底是真局还是自然波动？
用户问题（2026-10-16）：
  "我们的实验是不是真假结合的？有真局，也有假局，也有普通散户博弈？"

★ 此前所有实验的"局"定义都是纯数学的（未来 N 根内涨 >= X%），
  从未区分：真局（拉高-出货结构）/ 弱局 / 自然波动。
  若模型在"真局"上的精确率 ≈ 在"自然波动"上的精确率，说明它根本分不出来。

本脚本：先按**事后可见的操盘证据**把局分 3 档，再看模型（同 ops85 口径）在各档上的表现。
分档（事后，仅用于验证，不作为特征）：
  真局   gain>=100% 且 峰后回撤>=50%   ← 强操盘证据（拉高后崩盘）
  弱局   gain 20~100%                  ← 中等拉升，可能自然
  微局   gain 10~20%                   ← 普通波动
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_truefake.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_truefake · 局的真假分层：模型抓到的是真局还是自然波动？')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96',
         'll_d1','ll_d3','ll_d6','ll_d12','ll_d24','ll_d48','ll_d96','ll_d288',
         'll_rank','near_ll_run','ll_vol',
         'seg_ret','evt_rate','n_trans','dur_state','run_len','accel',
         'btc_r6','btc_r12','btc_r24','btc_lead','mkt_up','mkt_dn',
         'hour_sin','hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

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
Q = pd.DataFrame({r[0]: pd.Series(r[5], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

R1all = C.pct_change(12).values
Rnall = np.nan_to_num(R1all, nan=0.0)
up_cnt = (Rnall >= 0.05).sum(axis=1)
dn_cnt = (Rnall <= -0.05).sum(axis=1)
mkt_up_g = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
mkt_dn_g = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r12 = np.zeros(T); btc_r12[12:] = btc[12:] / btc[:-12] - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0

def build(sym, ot, c, h, l, q, tr):
    """返回该币的特征字典 F、起点列表 starts（每个含 t0,i1,gain,dd24,grade）、状态标签 yE"""
    n = len(c)
    W = np.lib.stride_tricks.sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= W.min(axis=1) + 1e-12
    hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    cand = np.flatnonzero(is_low & np.isfinite(hmax))
    last = -10 ** 9
    starts = []
    for t0 in cand:
        if t0 - last < WMAX: continue
        Hh = hmax[t0]; base = max(l[t0], 1e-12)
        gain = Hh / base - 1.0
        if gain < 0.10: continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 96)
        if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (Hh - base) * 0.5):
            continue
        # 峰后 24h（96 根）回撤
        if i1 + 3 < n:
            dd24 = 1.0 - l[i1:min(n, i1 + 97)].min() / max(h[i1], 1e-12)
        else:
            dd24 = np.nan
        # 分档
        if gain >= 1.0 and np.isfinite(dd24) and dd24 >= 0.50:
            grade = '真局(涨>=100%且崩>=50%)'
        elif gain >= 0.20:
            grade = '弱局(涨20~100%)'
        else:
            grade = '微局(涨10~20%)'
        starts.append((t0, i1, gain, dd24, grade))
        last = i1
    if not starts: return None, None, None

    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F = {}
    for k in (6,12,24,48,96,288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    for k in (6,12,24,96):
        F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
    for k in (1,3,6,12,24,48,96,288):
        lo = pd.Series(l).rolling(k, min_periods=1).min().values
        F['ll_d%d' % k] = c / np.maximum(lo, 1e-12) - 1.0
    lo288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WMAX, min_periods=72).max().values
    F['ll_rank'] = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    near = (c <= lo288 * 1.20).astype('float32')
    chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (near[1:] != near[:-1]).astype('float32')
    ix = np.arange(n)
    F['near_ll_run'] = (ix - np.maximum.accumulate(np.where(chg2 > 0, ix, 0)) + 1).astype('float64')
    mv288 = rmean(q, WMAX)
    F['ll_vol'] = np.where(mv288 > 0, q / np.maximum(mv288, 1e-12), np.nan)
    loS = lo288
    F['seg_ret'] = c / np.maximum(loS, 1e-12) - 1.0
    av = pd.Series(np.abs(r1)).rolling(48, min_periods=12).mean().values
    F['evt_rate'] = pd.Series((np.abs(r1) > 3 * av).astype('float32')).rolling(48, min_periods=12).mean().values
    sg = np.sign(np.nan_to_num(r1)); chg = np.zeros(n); chg[1:] = (sg[1:] != sg[:-1]).astype(float)
    F['n_trans'] = pd.Series(chg).rolling(48, min_periods=12).sum().values
    st = np.where(r1 >= 0.02, 2, np.where(r1 <= -0.02, 0, 1))
    dur = np.ones(n, dtype='float64')
    for i in range(1, n):
        dur[i] = dur[i-1] + 1 if st[i] == st[i-1] else 1
    F['dur_state'] = dur
    run = np.ones(n, dtype='float64')
    for i in range(1, n):
        run[i] = run[i-1] + 1 if sg[i] == sg[i-1] and sg[i] != 0 else 1
    F['run_len'] = run
    r6 = F['ret6'].copy(); r24 = F['ret24'].copy()
    F['accel'] = r6 - r24 / 4.0
    idx = pd.Index(ot)
    b6 = pd.Series(btc_r6, index=ts).reindex(idx).values
    b12 = pd.Series(btc_r12, index=ts).reindex(idx).values
    b24 = pd.Series(btc_r24, index=ts).reindex(idx).values
    mu = pd.Series(mkt_up_g, index=ts).reindex(idx).values
    md = pd.Series(mkt_dn_g, index=ts).reindex(idx).values
    F['btc_r6'], F['btc_r12'], F['btc_r24'] = b6, b12, b24
    F['btc_lead'] = b6 - r6
    F['mkt_up'], F['mkt_dn'] = mu, md
    hh = (ot // 3600000) % 24
    F['hour_sin'] = np.sin(2 * np.pi * hh / 24.0)
    F['hour_cos'] = np.cos(2 * np.pi * hh / 24.0)

    # 标签：t 落在某局起点 t0 之后 EARLY 根内
    yE = np.zeros(n, dtype='int8')
    grade_of_t = np.array(['无'] * n, dtype=object)
    for (t0, i1, gain, dd24, grade) in starts:
        lo = t0; hi = min(n, t0 + EARLY + 1)
        yE[lo:hi] = 1
        grade_of_t[lo:hi] = grade
    return F, starts, (yE, grade_of_t)

X, Y, TT, SS, GG = [], [], [], [], []
for (sym, ot, c, h, l, q, tr) in RAW:
    F, starts, lab = build(sym, ot, c, h, l, q, tr)
    if F is None: continue
    yE, gT = lab
    n = len(c)
    m = np.isfinite(F['ret288']) & np.isfinite(F['ll_d288'])
    idx = np.flatnonzero(m)
    if len(idx) < 500: continue
    keep = np.zeros(len(idx), dtype=bool); keep[::4] = True
    keep |= (yE[idx] == 1)          # 正例全留，负例降采样 1/4（省内存、提正例密度）
    idx = idx[keep]
    X.append(np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32'))
    Y.append(yE[idx]); TT.append(ot[idx]); SS.append(np.full(len(idx), sym, dtype=object))
    GG.append(gT[idx])
    del F; gc.collect()

X = np.concatenate(X); Y = np.concatenate(Y)
TT = np.concatenate(TT); SS = np.concatenate(SS); GG = np.concatenate(GG)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
log(f'总样本 {len(Y):,}  正例率 {100*Y.mean():.3f}%（{int(Y.sum()):,}）')

# 各档的样本量与基础比例
log('')
log('=' * 100)
log('【A】样本按"局档"分布（仅起点后 1h 内才带档标签）')
log('=' * 100)
for g in ['真局(涨>=100%且崩>=50%)', '弱局(涨20~100%)', '微局(涨10~20%)', '无']:
    m = (GG == g)
    log(f'  {g:<28} {int(m.sum()):>10,}  ({100*m.mean():>6.2f}%)')

# 时间留出
cut = np.quantile(TT, 0.70)
tr = TT <= cut; te = ~tr
te_idx = np.flatnonzero(te)
log('')
log('=' * 100)
log(f'【B】时间留出（前 70% → 后 30%）  训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}  基准 {100*Y[te].mean():.3f}%')
log('=' * 100)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                     min_samples_leaf=40, l2_regularization=1.0, random_state=0)
clf.fit(X[tr], Y[tr])
p = clf.predict_proba(X[te])[:, 1]
auc = roc_auc_score(Y[te], p) if len(np.unique(Y[te])) > 1 else float('nan')
log(f'  AUC = {auc:.4f}')

log('')
log('  ★ 核心问题：模型报出的高分点，落在哪些档？')
for thr in (0.3, 0.5, 0.7):
    sel_local = (p >= thr)
    if sel_local.sum() == 0:
        log(f'    thr={thr:<5} 无报出'); continue
    sel = te_idx[sel_local]        # 局部掩码 → 全局索引
    gg = GG[sel]; yy = Y[sel]
    log(f'    thr={thr:<5} 报出 {int(sel_local.sum()):>7,}  其中正例 {int(yy.sum()):>6,}（精确率 {100*yy.mean():>5.1f}%）')
    for g in ['真局(涨>=100%且崩>=50%)', '弱局(涨20~100%)', '微局(涨10~20%)', '无']:
        c1 = int((gg == g).sum())
        if c1: log(f'        {g:<28} {c1:>7,}  ({100.0*c1/len(gg):>5.1f}%)')
    # 各档内部召回
    for g in ['真局(涨>=100%且崩>=50%)', '弱局(涨20~100%)', '微局(涨10~20%)']:
        mg_local = (GG[te_idx] == g)
        if mg_local.sum() == 0: continue
        rec = (p[mg_local] >= thr).mean()
        log(f'        ↳ {g} 内部召回 {100*rec:>5.1f}%  (档内样本 {int(mg_local.sum()):,})')

log('')
log('=' * 100)
log('【C】判读')
log('=' * 100)
log('  · 若"真局"内部召回 ≈ "微局"内部召回 ⇒ 模型分不出真假，只是抓"会涨"')
log('  · 若"真局"内部召回 显著 高于"微局" ⇒ 模型确实偏向真操盘局')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
