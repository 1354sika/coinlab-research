# -*- coding: utf-8 -*-
"""gambler_phase.py — 周期位置识别：现在是在局的哪个阶段？
用户洞察（2026-10-16）：
  "无论是大目标还是小目标，目前都无法精准的预测。
   只能说知道如果抓到了之后，什么时候会跌？
   那我们还是要回到预测这个周期什么时候开始或是否已经在这个周期的开头没过多久"

关键转向：**不预测"未来涨跌"（做不到），改为判断"当前处于周期的哪个位置"**。
      "什么时候会开始" 是预测未来（难）
      "是不是已经在开头" 是判断现在（可观测）

做法：
  1) 标记所有局，算出每个局的**持续区间** [t0, peak]（起点→峰值）
  2) 对每个时刻 t（局内 t0..peak），定义"进度" prog = (t - t0) / max(peak - t0, 1)
  3) 标签：P0 开头 prog<0.20 / P1 前中 0.20~0.50 / P2 后中 0.50~0.80 / P3 末期 >=0.80
  4) 特征：**只用 t 及之前的**（无前视）
  5) 报告：四分类混淆矩阵 + 各阶段召回 + 二分"是否在开头"的 AUC / 85% 精确率
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_phase.txt')

GAIN = 0.20
WMAX = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_phase · 周期位置识别（现在是在局的哪个阶段？）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['seg_ret', 'evt_rate', 'n_trans', 'dur_state', 'run_len', 'accel']

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
        regs = []
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
            regs.append((t0, i1))
            last = i1
        if not regs:
            continue

        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F = {}
        for k in (6, 12, 24, 48, 96, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        vol = rmean(np.abs(np.nan_to_num(r1)), 24)
        thr_evt = np.maximum(3.0 * vol, 0.01)
        is_evt = np.abs(np.nan_to_num(r1)) >= thr_evt
        F['evt_rate'] = pd.Series(is_evt.astype('float32')).rolling(48, min_periods=8).mean().values
        sg = np.sign(np.nan_to_num(r1))
        chg = np.zeros(n, dtype='float32'); chg[1:] = (sg[1:] != sg[:-1]).astype('float32')
        F['n_trans'] = pd.Series(chg).rolling(48, min_periods=8).sum().values
        st = np.zeros(n, dtype='int8')
        v24 = rmean(q, 24); vr = np.where(v24 > 0, q / np.maximum(v24, 1e-12), 1.0)
        st[(np.nan_to_num(F['ret24']) > 0.03) & (vr > 1.5)] = 1
        st[(np.nan_to_num(F['ret24']) < -0.03) & (vr > 1.5)] = 2
        chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (st[1:] != st[:-1]).astype('float32')
        _ix = np.arange(n)
        F['dur_state'] = (_ix - np.maximum.accumulate(np.where(chg2 > 0, _ix, 0)) + 1).astype('float64')
        F['run_len'] = (_ix - np.maximum.accumulate(np.where(chg > 0, _ix, 0)) + 1).astype('float64')
        F['accel'] = F['ret6'] - F['ret24'] / 4.0

        idxs = []; ys = []
        for (t0, i1) in regs:
            if i1 <= t0:
                continue
            dur = i1 - t0
            for t in range(t0, i1 + 1, 2):
                prog = (t - t0) / max(dur, 1)
                y = 0 if prog < 0.20 else (1 if prog < 0.50 else (2 if prog < 0.80 else 3))
                idxs.append(t); ys.append(y)
        if not idxs:
            continue
        idxs = np.array(idxs); ys = np.array(ys)
        X = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in FEAT], axis=1).astype('float32')
        if len(X) < 10:
            continue
        REC.append((sym, X, ys, ot[idxs]))
    del ch; gc.collect()

log(f'币 {len(REC)}   耗时 {time.time()-t0t:.0f}s')
X = np.concatenate([r[1] for r in REC]); Y = np.concatenate([r[2] for r in REC])
T = np.concatenate([r[3] for r in REC])
log(f'局内样本 {len(X):,}   各阶段：P0 {int((Y==0).sum()):,}  P1 {int((Y==1).sum()):,}  '
    f'P2 {int((Y==2).sum()):,}  P3 {int((Y==3).sum()):,}')

cut = np.quantile(T, 0.70); tr = T < cut
m4 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=50,
                                    l2_regularization=1.0, random_state=42)
m4.fit(X[tr], Y[tr])
pred = m4.predict(X[~tr])
log('')
log('=' * 100)
log('【A】四分类：阶段识别（测试段）')
log('=' * 100)
cm = confusion_matrix(Y[~tr], pred, labels=[0, 1, 2, 3])
log('  混淆矩阵（行=真实，列=预测）：P0早 P1前中 P2后中 P3末')
for i in range(4):
    row = cm[i] / max(cm[i].sum(), 1)
    log(f'    真实 P{i}  n={cm[i].sum():>7,d}   ' + '  '.join(f'{100*v:5.1f}%' for v in row))
log(f'  总体准确率 {100*(pred == Y[~tr]).mean():.1f}%   （随机基线 25%）')

log('')
log('=' * 100)
log('【B】二分：是否在"开头"（prog<0.20）')
log('=' * 100)
y2 = (Y == 0).astype('int8')
m2 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=50,
                                    l2_regularization=1.0, random_state=42)
m2.fit(X[tr], y2[tr])
p = m2.predict_proba(X[~tr])[:, 1]
yy = y2[~tr]
log(f'  测试基准正例率 {100*yy.mean():.2f}%   AUC={roc_auc_score(yy, p):.4f}')
ok = None
for thr in np.arange(0.05, 0.99, 0.01):
    s = p >= thr
    if s.sum() < 50:
        break
    if yy[s].mean() >= 0.85:
        ok = (thr, int(s.sum()), 100*yy[s].mean(), 100*s.sum() / len(yy))
if ok:
    log(f'  ★ 85%精确率@P>={ok[0]:.2f}  报出 {ok[1]:,}（覆盖 {ok[3]:.2f}%）  精确率 {ok[2]:.1f}%')
else:
    log('  ✗ 无阈值达 85% 精确率')
for thr in (0.3, 0.5, 0.7, 0.9):
    s = p >= thr; nn = int(s.sum())
    if nn == 0:
        log(f'    thr={thr}: 无报出'); continue
    log(f'    thr={thr:.1f}  报出 {nn:,}  精确率 {100*yy[s].mean():5.1f}%  召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
