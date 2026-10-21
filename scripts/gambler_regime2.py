# -*- coding: utf-8 -*-
"""gambler_regime2.py — 以"局"为目标：能不能提前识别局的起点？
用户要求（2026-10-16）：
  "最重要的还是找到预测信号，或是影响概率的因素，算子，变量，以实现高概率的预测"

背景：gambler_multi3 证明"下跌条件"抓到的是"普通暴跌"，落在局内仅 5.8%。
      原因是目标设成了"未来跌 10%"，而不是"这里是某个局的起点"。
本脚本把目标换回**局本身**：

  正例 = 某个局的起点 t0 以及 t0 之后 EARLY 根（1h）内的时刻
  负例 = 其他所有时刻
  特征 = 全部只用 t 及之前（无前视）

输出：
  A 逐点：AUC / 85% 精确率可达性 / 各阈值报出
  B 局级：覆盖率（多少局被提前报出）+ 报出时已涨幅度
  C 特征重要性（置换重要性，抽样算）—— 回答"哪些算子/变量有用"
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_regime2.txt')

GAIN = 0.20
WMAX = 288
EARLY = 12          # 局起点后 1h 内算"早期"

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_regime2 · 以"局"为目标：能否提前识别局的起点？')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['seg_ret', 'up_frac24', 'evt_rate', 'n_trans', 'dur_state', 'run_len', 'accel', 'hour_sin', 'hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

REC = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000:
            continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64'); ot = g['open_time'].values

        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        last = -10 ** 9
        starts = []
        for t0 in cand:
            if t0 - last < WMAX:
                continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            if H / base - 1.0 < GAIN:
                continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            thr = base + (H - base) * 0.5
            k2 = min(n - 1, i1 + 96)
            if not (k2 > i1 and l[i1:k2 + 1].min() <= thr):
                continue
            starts.append((t0, i1))
            last = i1
        if not starts:
            continue

        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F = {}
        for k in (6, 12, 24, 48, 96, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2, k // 4)).max().values / c - 1.0
            F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2, k // 4)).min().values / c - 1.0
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)
        vol = rmean(np.abs(np.nan_to_num(r1)), 24)
        is_evt = np.abs(np.nan_to_num(r1)) >= np.maximum(3.0 * vol, 0.01)
        F['evt_rate'] = pd.Series(is_evt.astype('float32')).rolling(48, min_periods=8).mean().values
        sg = np.sign(np.nan_to_num(r1))
        chg = np.zeros(n, dtype='float32'); chg[1:] = (sg[1:] != sg[:-1]).astype('float32')
        F['n_trans'] = pd.Series(chg).rolling(48, min_periods=8).sum().values
        st = np.zeros(n, dtype='int8')
        v24 = rmean(q, 24); vrr = np.where(v24 > 0, q / np.maximum(v24, 1e-12), 1.0)
        st[(np.nan_to_num(F['ret24']) > 0.03) & (vrr > 1.5)] = 1
        st[(np.nan_to_num(F['ret24']) < -0.03) & (vrr > 1.5)] = 2
        chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (st[1:] != st[:-1]).astype('float32')
        _ix = np.arange(n)
        F['dur_state'] = (_ix - np.maximum.accumulate(np.where(chg2 > 0, _ix, 0)) + 1).astype('float64')
        F['run_len'] = (_ix - np.maximum.accumulate(np.where(chg > 0, _ix, 0)) + 1).astype('float64')
        F['accel'] = F['ret6'] - F['ret24'] / 4.0
        hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
        F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)

        # 正例：局起点后 EARLY 根内
        pos = np.zeros(n, dtype=bool)
        for (t0, i1) in starts:
            pos[t0:min(n, t0 + EARLY + 1)] = True

        idx_all = np.arange(WMAX, n - 12, 6)
        X = np.stack([F[k][idx_all] for k in FEAT], axis=1).astype('float32')
        m = np.isfinite(X).all(axis=1)
        ia = idx_all[m]; X = X[m]
        y = pos[ia].astype('int8')
        keep = np.flatnonzero(y == 1)
        keep_neg = np.flatnonzero(y == 0)[::8]
        sel = np.concatenate([keep, keep_neg])
        REC.append((sym, X[sel], y[sel], ot[ia[sel]], ia[sel], starts, c, n))
    del ch; gc.collect()

log(f'币 {len(REC)}   耗时 {time.time()-t0t:.0f}s')
X = np.concatenate([r[1] for r in REC]); Y = np.concatenate([r[2] for r in REC])
T = np.concatenate([r[3] for r in REC])
log(f'样本 {len(X):,}  正例率 {100*Y.mean():.3f}%（正例 {int(Y.sum()):,}）')

cut = np.quantile(T, 0.70); tr = T < cut
m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                    l2_regularization=1.0, random_state=42)
m_.fit(X[tr], Y[tr])
p = m_.predict_proba(X[~tr])[:, 1]
yy = Y[~tr]
log(f'测试 {len(yy):,}  基准 {100*yy.mean():.3f}%   AUC={roc_auc_score(yy, p):.4f}')

log('')
log('=' * 100)
log('【A】逐点：85% 精确率可达性')
log('=' * 100)
ok = None
for thr in np.arange(0.05, 0.995, 0.005):
    s = p >= thr
    if s.sum() < 30:
        break
    if yy[s].mean() >= 0.85:
        ok = (thr, int(s.sum()), 100*yy[s].mean(), 100*s.sum()/len(yy))
if ok:
    log(f'  ★ 85%精确率@P>={ok[0]:.3f}  报出 {ok[1]:,}（覆盖 {ok[3]:.3f}%）  精确率 {ok[2]:.1f}%')
else:
    log('  ✗ 无阈值达 85% 精确率')
for thr in (0.3, 0.5, 0.7, 0.9):
    s = p >= thr; nn = int(s.sum())
    if nn == 0:
        log(f'    thr={thr}: 无报出'); continue
    log(f'    thr={thr:.1f}  报出 {nn:,}  精确率 {100*yy[s].mean():5.1f}%  召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【B】局级：覆盖率 + 报出时的进度')
log('=' * 100)
for thr in (0.3, 0.5, 0.7):
    nreg = 0; cov = 0; progs = []
    for (sym, Xc, yc, otc, iac, starts, c, n) in REC:
        pc = m_.predict_proba(Xc)[:, 1]
        hit = pc >= thr
        for (t0, i1) in starts:
            j = np.flatnonzero(iac >= t0)
            if len(j) == 0:
                continue
            j0 = j[0]
            if otc[j0] < cut:
                continue
            nreg += 1
            k = np.flatnonzero(hit[j0:])
            if len(k) == 0:
                continue
            ij = int(iac[j0 + k[0]])
            if ij > i1:
                continue
            cov += 1
            progs.append((ij - t0) / max(i1 - t0, 1))
    if nreg == 0:
        continue
    log(f'  thr={thr:.1f}  测试局 {nreg}  报出 {cov}（覆盖 {100*cov/nreg:.1f}%）' + (
        f'  报出时进度中位 {100*np.median(progs):.1f}%' if progs else ''))

log('')
log('=' * 100)
log('【C】特征重要性（置换重要性，测试段抽样 20000 点）')
log('=' * 100)
rng = np.random.RandomState(0)
sub = rng.choice(len(yy), size=min(20000, len(yy)), replace=False)
Xsub = X[~tr][sub]; ysub = yy[sub]
try:
    pi = permutation_importance(m_, Xsub, ysub, n_repeats=3, random_state=0, n_jobs=1,
                                scoring='roc_auc')
    order = np.argsort(pi.importances_mean)[::-1]
    log('  top-12 特征（AUC 下降越大越重要）：')
    for k in order[:12]:
        log(f'    {FEAT[k]:<16s} {pi.importances_mean[k]:+.5f}')
    log('  最不重要的 5 个：')
    for k in order[-5:]:
        log(f'    {FEAT[k]:<16s} {pi.importances_mean[k]:+.5f}')
except Exception as e:
    log(f'  重要性计算失败：{e}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
