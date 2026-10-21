# -*- coding: utf-8 -*-
"""gambler_ops85.py — 冲 85% 精确率（严格无泄漏 + 三重留出）
用户要求（2026-10-16）："我们的目标仍然是百分之 85"

当前（gambler_ops_final，严格无泄漏）：AUC 0.9266~0.9402；
币留出 thr=0.8 精确率 85.0% 但召回仅 0.9%。目标：把 85% 精确率的召回抬起来。

本脚本按已验证的最强方向加强特征：
  A "贴低点"系列细化（regime2 里 ll12/ll96 的重要性是第三名的 20 倍）
      ll_d{1,3,6,12,24,48,96,288}  距各尺度最低点的距离
      ll_rank    当前价在 288 根 [低点,高点] 中的分位
      near_ll_run 已贴住"近 288 根低位 20%"的持续根数
      ll_vol     贴低点期间的量比（缩量=蓄势）
  B 跨币领先（lead2 实测 BTC 跌→小币跌 5 分钟 lift 265x）
      btc_r6/r12/r24   BTC 过去 6/12/24 根涨跌（t 时刻可观测）
      btc_lead         BTC 过去 6 根涨幅 − 本币过去 6 根涨幅（落后度）
      mkt_up/mkt_dn    过去 2h 全市场 1h 涨/跌 >=5% 的币数
评估：A 时间留出 / B 币留出 / C 双重留出；每档报出"达到 85% 精确率的最低阈值 + 报出数 + 召回"
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
OUT = os.path.join(EV, 'gambler_ops85.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_ops85 · 冲 85% 精确率（严格无泄漏 + 三重留出）')
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
if 'BTCUSDT' in syms:
    btc = C['BTCUSDT'].values.astype('float64')
else:
    btc = np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r12 = np.zeros(T); btc_r12[12:] = btc[12:] / btc[:-12] - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0

REC = []
for (sym, ot, c, h, l, q, tr) in RAW:
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
        if Hh / base - 1.0 < 0.10: continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 96)
        if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (Hh - base) * 0.5):
            continue
        starts.append((t0, i1)); last = i1
    if not starts: continue

    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F = {}
    for k in (6,12,24,48,96,288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    for k in (6,12,24,96):
        F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
    # A 贴低点系列
    for k in (1,3,6,12,24,48,96,288):
        lo = pd.Series(l).rolling(k, min_periods=1).min().values
        F['ll_d%d' % k] = c / np.maximum(lo, 1e-12) - 1.0
    lo288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WMAX, min_periods=72).max().values
    F['ll_rank'] = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    near = (c <= lo288 * 1.20).astype('float32')
    chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (near[1:] != near[:-1]).astype('float32')
    _ix = np.arange(n)
    F['near_ll_run'] = (_ix - np.maximum.accumulate(np.where(chg2 > 0, _ix, 0)) + 1).astype('float64')
    v96 = rmean(q, 96)
    F['ll_vol'] = np.where(v96 > 0, q / np.maximum(v96, 1e-12), 1.0)
    F['seg_ret'] = c / np.maximum(lo288, 1e-12) - 1.0
    vol = rmean(np.abs(np.nan_to_num(r1)), 24)
    is_evt = np.abs(np.nan_to_num(r1)) >= np.maximum(3.0*vol, 0.01)
    F['evt_rate'] = pd.Series(is_evt.astype('float32')).rolling(48, min_periods=8).mean().values
    sg = np.sign(np.nan_to_num(r1))
    chg = np.zeros(n, dtype='float32'); chg[1:] = (sg[1:] != sg[:-1]).astype('float32')
    F['n_trans'] = pd.Series(chg).rolling(48, min_periods=8).sum().values
    st = np.zeros(n, dtype='int8')
    v24 = rmean(q, 24); vrr = np.where(v24 > 0, q / np.maximum(v24, 1e-12), 1.0)
    st[(np.nan_to_num(F['ret24']) > 0.03) & (vrr > 1.5)] = 1
    st[(np.nan_to_num(F['ret24']) < -0.03) & (vrr > 1.5)] = 2
    chg3 = np.zeros(n, dtype='float32'); chg3[1:] = (st[1:] != st[:-1]).astype('float32')
    F['dur_state'] = (_ix - np.maximum.accumulate(np.where(chg3 > 0, _ix, 0)) + 1).astype('float64')
    F['run_len'] = (_ix - np.maximum.accumulate(np.where(chg > 0, _ix, 0)) + 1).astype('float64')
    F['accel'] = F['ret6'] - F['ret24'] / 4.0
    # B 跨币领先
    pos = np.clip(np.searchsorted(ts, ot, side='left'), 0, T - 1)
    F['btc_r6'] = btc_r6[pos]; F['btc_r12'] = btc_r12[pos]; F['btc_r24'] = btc_r24[pos]
    F['btc_lead'] = btc_r6[pos] - np.nan_to_num(F['ret6'], nan=0.0)
    F['mkt_up'] = mkt_up_g[pos]; F['mkt_dn'] = mkt_dn_g[pos]
    hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
    F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)

    posE = np.zeros(n, dtype=bool)
    for (t0, i1) in starts: posE[t0:min(n, t0+EARLY+1)] = True
    yO = posE.astype('int8')          # 标签 = 纯未来，与特征零重叠

    idxs = np.arange(288, n - 12, 6)
    X = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in FEAT], axis=1).astype('float32')
    y = yO[idxs]
    keep = np.flatnonzero(y == 1); kneg = np.flatnonzero(y == 0)[::8]
    sel = np.concatenate([keep, kneg])
    if len(sel) < 30: continue
    REC.append((sym, X[sel], y[sel], ot[idxs][sel]))
    del F; gc.collect()

log(f'样本组 {len(REC)}   耗时 {time.time()-t0t:.0f}s')
X = np.concatenate([r[1] for r in REC]); Y = np.concatenate([r[2] for r in REC])
Tt = np.concatenate([r[3] for r in REC])
S = np.concatenate([[r[0]] * len(r[1]) for r in REC])
log(f'总样本 {len(X):,}  正例率 {100*Y.mean():.3f}%（{int(Y.sum()):,}）')

def fit_eval(tr, te, tag):
    if Y[tr].sum() < 40 or Y[te].sum() < 20:
        log(f'  {tag}: 正例不足（训 {int(Y[tr].sum())} 测 {int(Y[te].sum())}）'); return
    m_ = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.07, min_samples_leaf=30,
                                        l2_regularization=0.5, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[te])[:, 1]; yy = Y[te]
    auc = roc_auc_score(yy, p)
    log(f'  {tag:<24s} 测 {int(te.sum()):>7,d}  基准 {100*yy.mean():5.2f}%  AUC={auc:.4f}')
    best85 = None
    for thr in np.arange(0.05, 0.995, 0.005):
        s = p >= thr
        if s.sum() < 5: break
        if yy[s].mean() >= 0.85:
            best85 = (thr, int(s.sum()), 100*yy[s].mean(), 100*s.sum()/max(yy.sum(), 1))
            break
    if best85:
        log(f'      ★ 85%精确率 @P>={best85[0]:.3f}  报出 {best85[1]:,}  覆盖测试正例 {best85[3]:.1f}%')
    else:
        log('      ✗ 无阈值达 85%')
    for thr in (0.5, 0.7, 0.8, 0.9):
        s = p >= thr; nn = int(s.sum())
        if nn == 0: continue
        log(f'      thr={thr:.1f} 报出 {nn:>6,d} 精确率 {100*yy[s].mean():5.1f}% 召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

cut = np.quantile(Tt, 0.70)
uniq = sorted(set(S)); odd = set(uniq[0::2]); even = set(uniq[1::2])
trB = np.array([s in odd for s in S]); teB = np.array([s in even for s in S])

log('')
log('=' * 100); log('【A】时间留出'); log('=' * 100)
fit_eval(Tt < cut, Tt >= cut, 'A 时间留出')
log('')
log('=' * 100); log('【B】币留出'); log('=' * 100)
fit_eval(trB, teB, 'B 币留出')
log('')
log('=' * 100); log('【C】双重留出'); log('=' * 100)
fit_eval(trB & (Tt < cut), teB & (Tt >= cut), 'C 时间+币')

log('')
log('=' * 100); log('【D】特征重要性 top-15（测试段抽样 20000）'); log('=' * 100)
try:
    from sklearn.inspection import permutation_importance
    m_ = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.07, min_samples_leaf=30,
                                        l2_regularization=0.5, random_state=42)
    m_.fit(X[Tt < cut], Y[Tt < cut])
    rng = np.random.RandomState(0)
    sub = rng.choice(int((Tt >= cut).sum()), size=min(20000, int((Tt >= cut).sum())), replace=False)
    Xs = X[Tt >= cut][sub]; ys = Y[Tt >= cut][sub]
    pi = permutation_importance(m_, Xs, ys, n_repeats=3, random_state=0, scoring='roc_auc')
    order = np.argsort(pi.importances_mean)[::-1]
    for k in order[:15]:
        log(f'    {FEAT[k]:<14s} {pi.importances_mean[k]:+.5f}')
except Exception as e:
    log(f'  重要性失败：{e}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
