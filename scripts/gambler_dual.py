# -*- coding: utf-8 -*-
"""gambler_dual.py — 双目标：操盘局（大）vs 普通散户博弈（小）
用户要求（2026-10-16）：
  1) "我们的模型应该是能够准确判断，操盘局 / 普通散户博弈。不止能看大，也能看小"
  2) "之前的目标未必是错误的，可以作为一个计算参考结果，以供模型决策"

做法：
  1) 三个口径标记：
       大局 = g20（5根内低点 → 288根内最高涨>=20% → 之后自峰腰斩）
       小局 = g10（同上，门槛 10%，不要求腰斩）
       其他 = 普通散户博弈
  2) 特征分两组：
       BASE = 基础特征（含 regime2 证明最重要的 ll12/ll96）
       REF  = **参考特征**：把此前各实验的输出当输入
              ref_dn  —— multi2 八个下跌条件的命中数（0~8）
              ref_phase —— "进度"代理（当前价相对近 288 根的上下位置）
              ref_evt —— 24h 内异动次数
  3) 训两个二分类器：A 是否大局起点 / B 是否小局起点
  4) 对比加/不加参考特征的 AUC 与 85% 可达性
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_dual.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_dual · 双目标：操盘局（大）vs 普通散户博弈（小）')
log('=' * 100)

B = ['ret6','ret12','ret24','ret48','ret96','ret288',
     'volr6','volr12','volr24','volr48','volr96','volr288',
     'trr6','trr12','trr24','trr48','trr96','trr288',
     'hh6','hh12','hh24','hh96','ll6','ll12','ll24','ll96',
     'seg_ret','up_frac24','evt_rate','n_trans','dur_state','run_len','accel',
     'hour_sin','hour_cos']
R = ['ref_dn', 'ref_up', 'ref_pos', 'ref_evt24']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

REC = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000: continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64'); ot = g['open_time'].values

        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        last = -10 ** 9
        big, small = [], []
        for t0 in cand:
            if t0 - last < WMAX: continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            gain = H / base - 1.0
            if gain < 0.10: continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            k2 = min(n - 1, i1 + 96)
            cutd = (k2 > i1 and l[i1:k2 + 1].min() <= base + (H - base) * 0.5)
            if gain >= 0.20 and cutd:
                big.append((t0, i1))
            elif gain >= 0.10:
                small.append((t0, i1))
            last = i1
        if not big and not small:
            continue

        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F = {}
        for k in (6,12,24,48,96,288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
            if k in (6,12,24,96):
                F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2,k//4)).min().values / c - 1.0
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)
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
        chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (st[1:] != st[:-1]).astype('float32')
        _ix = np.arange(n)
        F['dur_state'] = (_ix - np.maximum.accumulate(np.where(chg2 > 0, _ix, 0)) + 1).astype('float64')
        F['run_len'] = (_ix - np.maximum.accumulate(np.where(chg > 0, _ix, 0)) + 1).astype('float64')
        F['accel'] = F['ret6'] - F['ret24'] / 4.0
        hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
        F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)

        # ---- 参考特征（把之前实验的结论当输入）----
        # ref_dn：八个下跌条件命中数（multi2 用过的同族条件）
        dn_evt = (np.nan_to_num(F['ret12']) <= -0.03).astype('float32')
        conds = [
            (np.nan_to_num(F['ret12']) <= -0.03).astype('float32'),
            (np.nan_to_num(F['ret12']) <= -0.06).astype('float32'),
            (vrr >= 3.0).astype('float32'),
            (np.nan_to_num(F['trr24']) >= 2.0).astype('float32'),
            (np.abs(np.nan_to_num(r1)) > 3.0 * np.maximum(vol, 1e-4)).astype('float32'),
            (pd.Series(dn_evt).rolling(288, min_periods=24).sum().values >= 2).astype('float32'),
            (np.abs(np.nan_to_num(F['ret24'])) > 0.15).astype('float32'),
            (np.nan_to_num(F['ll24']) > -0.02).astype('float32'),
        ]
        F['ref_dn'] = np.sum(conds, axis=0)
        # ref_up：上行同族
        upc = [
            (np.nan_to_num(F['ret12']) >= 0.03).astype('float32'),
            (np.nan_to_num(F['ret12']) >= 0.06).astype('float32'),
            (vrr >= 3.0).astype('float32'),
            (np.nan_to_num(F['trr24']) >= 2.0).astype('float32'),
            (np.abs(np.nan_to_num(r1)) > 3.0 * np.maximum(vol, 1e-4)).astype('float32'),
            (np.nan_to_num(F['ret24']) > 0.15).astype('float32'),
            (np.nan_to_num(F['hh24']) < 0.02).astype('float32'),
            (np.nan_to_num(F['hh96']) < 0.02).astype('float32'),
        ]
        F['ref_up'] = np.sum(upc, axis=0)
        # ref_pos：当前价在近 288 根的上下位置 [0,1]
        h288 = pd.Series(h).rolling(WMAX, min_periods=72).max().values
        F['ref_pos'] = (c - l288) / np.maximum(h288 - l288, 1e-12)
        # ref_evt24：24h 异动次数
        F['ref_evt24'] = pd.Series(is_evt.astype('float32')).rolling(WMAX, min_periods=24).sum().values

        posB = np.zeros(n, dtype=bool); posS = np.zeros(n, dtype=bool)
        for (t0, i1) in big:   posB[t0:min(n, t0+EARLY+1)] = True
        for (t0, i1) in small: posS[t0:min(n, t0+EARLY+1)] = True
        posS = posS & ~posB

        idx = np.arange(WMAX, n - 12, 6)
        Xb = np.stack([F[k][idx] for k in B], axis=1).astype('float32')
        Xr = np.stack([F[k][idx] for k in R], axis=1).astype('float32')
        m = np.isfinite(Xb).all(axis=1) & np.isfinite(Xr).all(axis=1)
        idx = idx[m]; Xb = Xb[m]; Xr = Xr[m]
        yB = posB[idx].astype('int8'); yS = posS[idx].astype('int8')
        keep = np.flatnonzero(yB | yS)
        kneg = np.flatnonzero(~(yB | yS))[::8]
        sel = np.concatenate([keep, kneg])
        if len(sel) < 20: continue
        REC.append((sym, Xb[sel], Xr[sel], yB[sel], yS[sel], ot[idx[sel]]))
    del ch; gc.collect()

log(f'币 {len(REC)}   耗时 {time.time()-t0t:.0f}s')

def run(use_ref, ysel, tag):
    Xb = np.concatenate([r[1] for r in REC]); Xr = np.concatenate([r[2] for r in REC])
    Y = np.concatenate([r[3] if ysel == 'B' else r[4] for r in REC])
    T = np.concatenate([r[5] for r in REC])
    X = np.hstack([Xb, Xr]) if use_ref else Xb
    cut = np.quantile(T, 0.70); tr = T < cut
    if Y[tr].sum() < 50 or Y[~tr].sum() < 20:
        log(f'  {tag}: 正例不足（训 {int(Y[tr].sum())} 测 {int(Y[~tr].sum())}）'); return
    m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[~tr])[:, 1]; yy = Y[~tr]
    auc = roc_auc_score(yy, p)
    ok = None
    for thr in np.arange(0.05, 0.995, 0.005):
        s = p >= thr
        if s.sum() < 30: break
        if yy[s].mean() >= 0.85:
            ok = (thr, int(s.sum()), 100*yy[s].mean(), 100*s.sum()/len(yy))
    line = f'  {tag:<22s} 样本 {len(X):>8,d}  正例 {100*Y.mean():5.3f}%  AUC={auc:.4f}'
    if ok:
        line += f'  ★85% @P>={ok[0]:.3f} 报出{ok[1]:,}(覆盖{ok[3]:.3f}%)'
    else:
        line += '  ✗无阈值达85%'
    log(line)
    for thr in (0.3, 0.5, 0.7):
        s = p >= thr; nn = int(s.sum())
        if nn == 0: continue
        log(f'      thr={thr:.1f} 报出 {nn:,} 精确率 {100*yy[s].mean():5.1f}% 召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【A】大局（g20：涨>=20% 且之后腰斩）')
log('=' * 100)
run(False, 'B', 'A1 基础特征')
run(True,  'B', 'A2 基础+参考特征')

log('')
log('=' * 100)
log('【B】小局（g10：涨>=10%，非大局）')
log('=' * 100)
run(False, 'S', 'B1 基础特征')
run(True,  'S', 'B2 基础+参考特征')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
