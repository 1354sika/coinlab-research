# -*- coding: utf-8 -*-
"""gambler_retrain.py — 重训：把"币自身时间结构"特征加进模型（用户指示 2026-10-16）

依据（job-47368-27 selfbfly 实测，2,082,130 样本）：
  距上次起局 0~24h  → 后 24h 再起局率 51.72%（基准 0.234%）  **lift 220.6x**
  距上次起局 24~72h → 0.446% (1.90x)   72~168h → 0.121% (0.52x)   >168h → 0.014% (0.06x)
  前 24h 大波动根数(|r|>=5%) >20 → 11.56% (49.3x)；6~20 → 5.06% (21.6x)
  前 24h 涨幅 >+30% → 8.55% (36.5x)；前 24h 量比 5~20x → 2.87% (12.2x)
  ⇒ 局是**成簇**发生的；"上一次起局距今多久"是最强的自身时间特征。

本脚本把这 6 个自身时间特征并入原特征集，重训两阶段（Stage1 有无局 / Stage2 微弱真），
  并**重点考察真档能否被识别**（前三轮全部失败：召回 0~4%）。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_retrain.txt')
WDAY = 288
EARLY = 12
GAME = 0.50
C3 = ['微局(10~20%)', '弱局(20~50%)', '真局(>=50%)']

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_retrain · 加入"币自身时间结构"特征后重训（承 selfbfly 的 220x 发现）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96',
         'll_d1','ll_d3','ll_d6','ll_d12','ll_d24','ll_d48','ll_d96','ll_d288',
         'll_rank','near_ll_run','ll_vol','seg_ret','evt_rate','n_trans','dur_state','run_len','accel',
         'btc_r6','btc_r12','btc_r24','btc_lead','mkt_up','mkt_dn','hour_sin','hour_cos']
FEAT_SELF = ['gap_since_game','had_game24','p_nbig24','p_vol24','p_ret24','p_maxup24']
FEAT_ALL = FEAT + FEAT_SELF

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
ts = C.index.values
T, N = C.shape
R1all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R1all, nan=0.0) >= 0.05).sum(axis=1)
mkt_up_g = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r12 = np.zeros(T); btc_r12[12:] = btc[12:] / btc[:-12] - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

X, Y, TT = [], [], []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 2: continue
    # ---------- 自身时间结构（只用 t 及之前）----------
    hs = pd.Series(h); ls = pd.Series(l); cs = pd.Series(c); qs = pd.Series(q)
    fwd_all = hs.rolling(WDAY, min_periods=1).max().shift(-WDAY).values
    up_fwd = fwd_all / c - 1.0
    gap = np.full(n, 1e9)          # 距上次起局多少根
    last = -10 ** 9
    for t in range(n):
        gap[t] = (t - last) if last > -10**8 else 1e9
        if np.isfinite(up_fwd[t]) and up_fwd[t] >= GAME and t - last > WDAY:
            last = t
    gap_c = np.minimum(gap, WDAY * 7)
    had24 = (gap <= WDAY).astype('float64')
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    p_nbig = pd.Series((np.abs(r1) >= 0.05).astype('float64')).rolling(WDAY).sum().values
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    p_vol = np.where(base_q > 0, qs.rolling(WDAY).sum().values / np.maximum(base_q * WDAY, 1e-12), np.nan)
    c0 = cs.shift(WDAY).values
    p_ret = c / np.maximum(c0, 1e-12) - 1.0
    p_maxup = hs.rolling(WDAY).max().values / np.maximum(c0, 1e-12) - 1.0

    # 标签：t 落在局起点后 EARLY 根内（局定义 = 未来 288 根内涨 >= GAME，峰后崩 >= 30%）
    fmax = hs.rolling(WDAY, min_periods=1).max().shift(-WDAY).values
    cand = np.flatnonzero(np.isfinite(fmax) & (fmax / c - 1.0 >= 0.10))
    last_i = -10 ** 9
    y = np.zeros(n, dtype='int8')
    for t0 in cand:
        if t0 - last_i < WDAY: continue
        base = max(l[t0], 1e-12)
        gain = fmax[t0] / base - 1.0
        i1 = t0 + int(np.argmax(h[t0:min(n, t0 + WDAY)]))
        k2 = min(n, i1 + 97)
        if k2 - i1 < 5: continue
        dd = 1.0 - l[i1:k2].min() / max(h[i1], 1e-12)
        if gain >= 0.50 and dd >= 0.30: y[t0:t0+EARLY+1] = 3      # 真局
        elif gain >= 0.20:             y[t0:t0+EARLY+1] = 2      # 弱局
        elif gain >= 0.10:             y[t0:t0+EARLY+1] = 1      # 微局
        last_i = i1

    F = {}
    for k in (6,12,24,48,96,288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    for k in (6,12,24,96):
        F['hh%d' % k] = hs.rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
    for k in (1,3,6,12,24,48,96,288):
        lo = ls.rolling(k, min_periods=1).min().values
        F['ll_d%d' % k] = c / np.maximum(lo, 1e-12) - 1.0
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    F['ll_rank'] = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    near = (c <= lo288 * 1.20).astype('float32')
    chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (near[1:] != near[:-1]).astype('float32')
    ix = np.arange(n)
    F['near_ll_run'] = (ix - np.maximum.accumulate(np.where(chg2 > 0, ix, 0)) + 1).astype('float64')
    mv288 = rmean(q, WDAY)
    F['ll_vol'] = np.where(mv288 > 0, q / np.maximum(mv288, 1e-12), np.nan)
    F['seg_ret'] = c / np.maximum(lo288, 1e-12) - 1.0
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
    F['accel'] = F['ret6'] - F['ret24'] / 4.0
    ii = pd.Index(ot)
    b6 = pd.Series(btc_r6, index=ts).reindex(ii).values
    F['btc_r6'] = b6
    F['btc_r12'] = pd.Series(btc_r12, index=ts).reindex(ii).values
    F['btc_r24'] = pd.Series(btc_r24, index=ts).reindex(ii).values
    F['btc_lead'] = b6 - F['ret6']
    F['mkt_up'] = pd.Series(mkt_up_g, index=ts).reindex(ii).values
    F['mkt_dn'] = pd.Series(mkt_up_g, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24
    F['hour_sin'] = np.sin(2 * np.pi * hh / 24.0)
    F['hour_cos'] = np.cos(2 * np.pi * hh / 24.0)
    # 自身时间结构
    F['gap_since_game'] = gap_c.astype('float64')
    F['had_game24'] = had24
    F['p_nbig24'] = p_nbig
    F['p_vol24'] = p_vol
    F['p_ret24'] = p_ret
    F['p_maxup24'] = p_maxup

    m = np.isfinite(F['ret288']) & np.isfinite(F['ll_d288'])
    idx = np.flatnonzero(m)
    if len(idx) < 500: continue
    keep = np.zeros(len(idx), dtype=bool); keep[::8] = True
    keep |= (y[idx] > 0) | (had24[idx] > 0)      # 正例 + "刚有局"的样本全留
    idx = idx[keep]
    X.append(np.stack([F[k][idx] for k in FEAT_ALL], axis=1).astype('float32'))
    Y.append(y[idx]); TT.append(ot[idx])
    del F; gc.collect()

X = np.nan_to_num(np.concatenate(X), nan=0.0, posinf=0.0, neginf=0.0)
Y = np.concatenate(Y); TT = np.concatenate(TT)
log(f'总样本 {len(Y):,}')
for i, nm in enumerate(['无'] + C3):
    log(f'  {nm:<16} {int((Y==i).sum()):>9,}  ({100.0*(Y==i).mean():.3f}%)')

cut = np.quantile(TT, 0.70)
tr = TT <= cut; te = ~tr
log(f'训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')

def run(FEATS, tag):
    cols = [FEAT_ALL.index(k) for k in FEATS]
    Xs = X[:, cols]
    yb = (Y > 0).astype('int8')
    c1 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=6,
                                        min_samples_leaf=25, l2_regularization=1.0, random_state=0)
    c1.fit(Xs[tr], yb[tr])
    p1 = c1.predict_proba(Xs[te])[:, 1]
    auc = roc_auc_score(yb[te], p1) if len(np.unique(yb[te])) > 1 else float('nan')
    mpos = (Y > 0); tr2 = tr & mpos
    cnt = np.array([(Y[tr2] == k).sum() for k in (1, 2, 3)], dtype='float64')
    w = np.array([1.0 / max(cnt[k-1], 1.0) for k in Y[tr2]])
    c2 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=6,
                                        min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    c2.fit(Xs[tr2], Y[tr2], sample_weight=w)
    log('')
    log('=' * 100)
    log(f'【{tag}】 Stage1(有无局) AUC = {auc:.4f}')
    for thr in (0.3, 0.5):
        sel = (p1 >= thr)
        if sel.sum() == 0: continue
        cls2 = c2.classes_[np.argmax(c2.predict_proba(Xs[te][sel]), axis=1)]
        truth = Y[te][sel]
        line = f'  thr={thr} 报出 {int(sel.sum()):>7,}：'
        for k, nm in ((1, C3[0]), (2, C3[1]), (3, C3[2])):
            m = (cls2 == k)
            if m.sum() == 0: line += f' [{nm} 0]'; continue
            line += f' [{nm} {int(m.sum()):,}→真{int((truth[m]==k).sum()):,}={100.0*(truth[m]==k).mean():.0f}%]'
        log(line)
        for k, nm in ((1, C3[0]), (2, C3[1]), (3, C3[2])):
            mt = (truth == k)
            if mt.sum() == 0: continue
            log(f'      ↳ 真实{nm} {int(mt.sum()):>5,}，判对 {int((cls2[mt]==k).sum()):>5,}（召回 {100.0*(cls2[mt]==k).mean():>5.1f}%）')
    return auc

run(FEAT_ALL, '全部特征（原 34 + 自身时间 6）')
run(FEAT, '对照：只用原 34 特征（不加自身时间结构）')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
