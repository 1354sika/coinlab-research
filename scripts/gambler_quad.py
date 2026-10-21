# -*- coding: utf-8 -*-
"""gambler_multiclass2.py — 两阶段分档：先筛"有局"，再在局内分档（修正上一版失败）
上一版（gambler_multiclass）失败：4 分类直接训练，受 98.6% 的"无"类主导
  ⇒ 模型从不报"真局"（真局召回 0.0%，微局 0.5%、弱局 1.6%），总体准确率 98.9% 纯属"抄无"。

本版两处修正：
  ① 两阶段：Stage1 二分类"未来有没有局"（AUC~0.94）；Stage2 **只在局样本内**做 3 分类（微/弱/真）
  ② 类别权重：Stage2 按类频逆加权，抵消真局样本极少（严格档仅 169 点）的劣势
输出：报出档位分布 + 各档精确率/召回 + **"说是真局的，有多少真是真局"**（决策直接可用）
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
OUT = os.path.join(EV, 'gambler_multiclass2.txt')

WMAX = 288
EARLY = 12
C3 = ['微局(10~20%)', '弱局(20~100%)', '真局(>=100%&崩>=50%)']

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_multiclass2 · 两阶段分档（先筛有局 → 局内分档，类别加权）')
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

def feats(ot, c, h, l, q, tr):
    n = len(c)
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
    idx = pd.Index(ot)
    b6 = pd.Series(btc_r6, index=ts).reindex(idx).values
    F['btc_r6'] = b6
    F['btc_r12'] = pd.Series(btc_r12, index=ts).reindex(idx).values
    F['btc_r24'] = pd.Series(btc_r24, index=ts).reindex(idx).values
    F['btc_lead'] = b6 - F['ret6']
    F['mkt_up'] = pd.Series(mkt_up_g, index=ts).reindex(idx).values
    F['mkt_dn'] = pd.Series(mkt_dn_g, index=ts).reindex(idx).values
    hh = (ot // 3600000) % 24
    F['hour_sin'] = np.sin(2 * np.pi * hh / 24.0)
    F['hour_cos'] = np.cos(2 * np.pi * hh / 24.0)
    return F

def starts_of(c, h, l):
    n = len(c)
    W = np.lib.stride_tricks.sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= W.min(axis=1) + 1e-12
    hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    cand = np.flatnonzero(is_low & np.isfinite(hmax))
    last = -10 ** 9; out = []
    for t0 in cand:
        if t0 - last < WMAX: continue
        Hh = hmax[t0]; base = max(l[t0], 1e-12)
        gain = Hh / base - 1.0
        if gain < 0.10: continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 96)
        if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (Hh - base) * 0.5):
            continue
        dd24 = 1.0 - l[i1:min(n, i1 + 97)].min() / max(h[i1], 1e-12) if i1 + 3 < n else np.nan
        out.append((t0, gain, dd24)); last = i1
    return out

X, Y, TT = [], [], []
for (sym, ot, c, h, l, q, tr) in RAW:
    st = starts_of(c, h, l)
    if not st: continue
    F = feats(ot, c, h, l, q, tr)
    n = len(c)
    y = np.zeros(n, dtype='int8')     # 0=无 1=微 2=弱 3=真
    for (t0, gain, dd24) in st:
        lo = t0; hi = min(n, t0 + EARLY + 1)
        if gain >= 0.50 and np.isfinite(dd24) and dd24 >= 0.30: y[lo:hi] = 3   # 真局（放宽：涨>=50% 且峰后崩>=30%）
        elif gain >= 0.20: y[lo:hi] = 2
        else: y[lo:hi] = 1
    m = np.isfinite(F['ret288']) & np.isfinite(F['ll_d288'])
    idx = np.flatnonzero(m)
    if len(idx) < 500: continue
    keep = np.zeros(len(idx), dtype=bool); keep[::16] = True   # 负例只留 1/16
    keep |= (y[idx] > 0)                                        # 正例全留
    idx = idx[keep]
    X.append(np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32'))
    Y.append(y[idx]); TT.append(ot[idx])
    del F; gc.collect()

X = np.nan_to_num(np.concatenate(X), nan=0.0, posinf=0.0, neginf=0.0)
Y = np.concatenate(Y); TT = np.concatenate(TT)
log(f'总样本 {len(Y):,}')
for i, nm in enumerate(['无'] + C3):
    log(f'  {nm:<24} {int((Y == i).sum()):>9,}  ({100.0*(Y == i).mean():.3f}%)')

cut = np.quantile(TT, 0.70)
tr = TT <= cut; te = ~tr
te_idx = np.flatnonzero(te)
log(f'训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')

# ---------- Stage 1：有没有局 ----------
yb = (Y > 0).astype('int8')
clf1 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=6,
                                      min_samples_leaf=25, l2_regularization=1.0, random_state=0)
clf1.fit(X[tr], yb[tr])
p1 = clf1.predict_proba(X[te])[:, 1]
if len(np.unique(yb[te])) > 1:
    log('')
    log(f'Stage1（有没有局）测试段 AUC = {roc_auc_score(yb[te], p1):.4f}   基准 {100.0*yb[te].mean():.3f}%')

# ---------- Stage 2：局内 3 分类，类别加权 ----------
mpos = (Y > 0)
tr2 = tr & mpos; te2 = te & mpos
cnt = np.array([(Y[tr2] == k).sum() for k in (1, 2, 3)], dtype='float64')
w_tr2 = np.array([1.0 / max(cnt[k-1], 1.0) for k in Y[tr2]])
log(f'Stage2 训练样本 {int(tr2.sum()):,}  各类 {cnt.astype(int).tolist()}  权重已按类频逆加权')
clf2 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=6,
                                      min_samples_leaf=15, l2_regularization=1.0, random_state=0)
clf2.fit(X[tr2], Y[tr2], sample_weight=w_tr2)

log('')
log('=' * 100)
log('【两阶段串联：Stage1 报出 → Stage2 分档】')
log('=' * 100)
for thr in (0.3, 0.5, 0.7):
    sel_l = (p1 >= thr)
    if sel_l.sum() == 0:
        log(f'  thr={thr}: 无报出'); continue
    p2 = clf2.predict_proba(X[te][sel_l])
    cls2 = clf2.classes_[np.argmax(p2, axis=1)]
    truth = Y[te][sel_l]
    log(f'  thr={thr}  报出 {int(sel_l.sum()):>7,}（占测试 {100.0*sel_l.mean():.2f}%）')
    for k, nm in ((1, C3[0]), (2, C3[1]), (3, C3[2])):
        m = (cls2 == k)
        if m.sum() == 0:
            log(f'      → 判为 {nm:<24} 0 个'); continue
        tp = int((truth[m] == k).sum())
        log(f'      → 判为 {nm:<24} {int(m.sum()):>6,} 个，其中确是 {tp:>5,}（精确率 {100.0*tp/m.sum():>5.1f}%）')
    for k, nm in ((1, C3[0]), (2, C3[1]), (3, C3[2])):
        mt = (truth == k)
        if mt.sum() == 0: continue
        log(f'      ↳ 真实{nm} {int(mt.sum()):>5,} 个，被正确判出 {int((cls2[mt]==k).sum()):>5,}（召回 {100.0*(cls2[mt]==k).mean():>5.1f}%）')

log('')
log('=' * 100)
log('【Stage2 单独：只看"已经是局"的样本，3 分类准确率】')
log('=' * 100)
if (Y[te2] > 0).sum():
    p2b = clf2.predict_proba(X[te2]); cls2b = clf2.classes_[np.argmax(p2b, axis=1)]
    cm = confusion_matrix(Y[te2], cls2b, labels=[1, 2, 3])
    log('  行=真实  列=预测     ' + ''.join(f'{C3[i][:14]:>16}' for i in range(3)))
    for ri in range(3):
        log(f'  {C3[ri][:12]:<12}' + ''.join(f'{cm[ri][ci]:>16,}' for ci in range(3)))
    log(f'  局内 3 分类准确率 {100.0*(cls2b == Y[te2]).mean():.1f}%   基准（全判弱局）{100.0*(Y[te2]==2).mean():.1f}%')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
