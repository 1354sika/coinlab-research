# -*- coding: utf-8 -*-
"""gambler_big.py — 专攻大局（翻倍的局）的早期识别
用户要求（2026-10-16）："大局（翻倍的局）的早期识别这个依然非常重要"

现状：大局（+20% 且之后腰斩）正例率仅 0.362%，最高精确率 29.6%（thr=0.3，27 个报出）。
本脚本换一组**"启动前形态"特征**——描述局起点之前的横盘/缩量/蓄势，而不是当天的涨跌：

  PRE 特征（全部只用 t 及之前）：
    quiet96/quiet288  过去 96/288 根的波动率（越低=越横盘）
    shrink_now        当前量比（<1 表示缩量）
    shrink_days       过去 288 根里有几根量比<0.7（长期清淡）
    dist_ll288        当前价距 288 根最低点的距离（贴着低点=蓄势）
    dist_hh288        距 288 根最高点的距离
    range96/288       96/288 根的振幅（高点-低点）/均价
    test_cnt          过去 288 根内"上冲后回落"的小试探次数
    quiet_run         低波动连续了多少根
对比：只用当日特征（ret/volr/trr 等） vs +PRE 特征
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
OUT = os.path.join(EV, 'gambler_big.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_big · 专攻大局：启动前的横盘/缩量/蓄势形态')
log('=' * 100)

DAY = ['ret6','ret12','ret24','ret48','ret96','ret288',
       'volr6','volr12','volr24','volr48','volr96',
       'trr12','trr24','trr48','trr96',
       'hh6','hh12','hh24','hh96','ll6','ll12','ll24','ll96',
       'evt_rate','n_trans','accel','hour_sin','hour_cos']
PRE = ['quiet96','quiet288','shrink_now','shrink_days','dist_ll288','dist_hh288',
       'range96','range288','test_cnt','quiet_run','dd_from_hh288']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

REC = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 3000: continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64'); ot = g['open_time'].values
        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        last = -10 ** 9
        bigs = []
        for t0 in cand:
            if t0 - last < WMAX: continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            gain = H / base - 1.0
            if gain < 0.20: continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            k2 = min(n - 1, i1 + 96)
            if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (H - base) * 0.5):
                continue
            bigs.append((t0, i1)); last = i1
        if not bigs: continue

        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F = {}
        for k in (6,12,24,48,96,288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        for k in (12,24,48,96):
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
        for k in (6,12,24,96):
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
            F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2,k//4)).min().values / c - 1.0
        vol = rmean(np.abs(np.nan_to_num(r1)), 24)
        is_evt = np.abs(np.nan_to_num(r1)) >= np.maximum(3.0*vol, 0.01)
        F['evt_rate'] = pd.Series(is_evt.astype('float32')).rolling(48, min_periods=8).mean().values
        sg = np.sign(np.nan_to_num(r1))
        chg = np.zeros(n, dtype='float32'); chg[1:] = (sg[1:] != sg[:-1]).astype('float32')
        F['n_trans'] = pd.Series(chg).rolling(48, min_periods=8).sum().values
        F['accel'] = F['ret6'] - F['ret24'] / 4.0
        hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
        F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)

        # ---- PRE：启动前形态（全部只用过去）----
        av = np.abs(np.nan_to_num(r1))
        F['quiet96'] = rmean(av, 96)
        F['quiet288'] = rmean(av, 288)
        v96 = rmean(q, 96); v288 = rmean(q, 288)
        F['shrink_now'] = np.where(v96 > 0, q / np.maximum(v96, 1e-12), 1.0)
        lowq = (np.where(v288 > 0, q / np.maximum(v288, 1e-12), 1.0) < 0.7).astype('float32')
        F['shrink_days'] = pd.Series(lowq).rolling(288, min_periods=48).sum().values
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        h288 = pd.Series(h).rolling(WMAX, min_periods=72).max().values
        F['dist_ll288'] = c / np.maximum(l288, 1e-12) - 1.0
        F['dist_hh288'] = c / np.maximum(h288, 1e-12) - 1.0
        F['dd_from_hh288'] = c / np.maximum(h288, 1e-12) - 1.0
        hi96 = pd.Series(h).rolling(96, min_periods=24).max().values
        lo96 = pd.Series(l).rolling(96, min_periods=24).min().values
        F['range96'] = (hi96 - lo96) / np.maximum(c, 1e-12)
        hi288 = pd.Series(h).rolling(288, min_periods=72).max().values
        lo288 = pd.Series(l).rolling(288, min_periods=72).min().values
        F['range288'] = (hi288 - lo288) / np.maximum(c, 1e-12)
        # 小试探：14 根内涨幅>2% 之后 14 根内又跌回
        up2 = (pd.Series(c).pct_change(14).values > 0.02)
        F['test_cnt'] = pd.Series(up2.astype('float32')).rolling(288, min_periods=48).sum().values
        # 低波动连续根数
        qz = (av < 0.01).astype('float32')
        chg3 = np.zeros(n, dtype='float32'); chg3[1:] = (qz[1:] != qz[:-1])
        _ix = np.arange(n)
        F['quiet_run'] = (_ix - np.maximum.accumulate(np.where(chg3 > 0, _ix, 0)) + 1).astype('float64')

        posB = np.zeros(n, dtype=bool)
        for (t0, i1) in bigs: posB[t0:min(n, t0+EARLY+1)] = True
        idxs = np.arange(288, n - 12, 6)
        Xd = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in DAY], axis=1).astype('float32')
        Xp = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in PRE], axis=1).astype('float32')
        y = posB[idxs].astype('int8')
        keep = np.flatnonzero(y == 1); kneg = np.flatnonzero(y == 0)[::8]
        sel = np.concatenate([keep, kneg])
        if len(sel) < 30: continue
        REC.append((sym, Xd[sel], Xp[sel], y[sel], ot[idxs][sel]))
    del ch; del F; gc.collect()

log(f'币 {len(REC)}   耗时 {time.time()-t0t:.0f}s')

def run(use_pre, tag):
    Xd = np.concatenate([r[1] for r in REC]); Xp = np.concatenate([r[2] for r in REC])
    Y = np.concatenate([r[3] for r in REC]); T = np.concatenate([r[4] for r in REC])
    X = np.hstack([Xd, Xp]) if use_pre else Xd
    cut = np.quantile(T, 0.70); tr = T < cut
    if Y[tr].sum() < 40 or Y[~tr].sum() < 20:
        log(f'  {tag}: 正例不足（训 {int(Y[tr].sum())} 测 {int(Y[~tr].sum())}）'); return
    m_ = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.07, min_samples_leaf=30,
                                        l2_regularization=0.5, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[~tr])[:, 1]; yy = Y[~tr]
    auc = roc_auc_score(yy, p)
    log(f'  {tag:<22s} 样本 {len(X):>8,d}  正例 {100*Y.mean():.3f}%  AUC={auc:.4f}')
    for thr in (0.1, 0.2, 0.3, 0.5):
        s = p >= thr; nn = int(s.sum())
        if nn == 0:
            log(f'      thr={thr:.1f} 无报出'); continue
        log(f'      thr={thr:.1f} 报出 {nn:,} 精确率 {100*yy[s].mean():5.1f}% 召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【大局：当日特征 vs +启动前形态】')
log('=' * 100)
run(False, 'A 只用当日特征')
run(True,  'B +启动前形态(PRE)')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
