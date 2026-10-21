# -*- coding: utf-8 -*-
"""gambler_scale.py — 隐时间结构：不加固定窗口，让模型自己找"状态持续时间"
用户要求（2026-10-16）："4h 本来就是相对的。真正的时间是不确定的"
                    "各种持续状态的时间也是重要的参数。什么时候涨跌，持续了多久？"
                    "突然变这样还是持续的变这样，各个变化时间的持续时间的构成"

做法（不预设任何窗口长度）：
  1) 用"状态机"给每根 5m 打标签（静默/吸筹/拉升/滞涨/崩塌）
  2) 算**持续时间特征**（与固定窗口无关）：
       dur_state   当前状态已持续多少根
       dur_since_evt 距上次异动（|ret1|>=3%）多少根
       n_trans     过去 288 根里状态切换次数（"突然" vs "持续"）
       evt_rate    过去 288 根里异动次数占比
       run_up/run_dn 当前同向连续根数
  3) 对比：基础特征 vs 基础+隐时间结构，看 AUC 与 85% 精确率达成情况
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
OUT = os.path.join(EV, 'gambler_scale.txt')

WMAX = 288
buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_scale · 隐时间结构（状态持续时间，不预设窗口）')
log('=' * 100)

BASE = []
for k in (6, 12, 24, 48, 96, 288):
    BASE += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
EXT = ['dur_state', 'dur_since_evt', 'n_trans', 'evt_rate', 'run_dir', 'run_len', 'state_id']

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

        F = {}
        for k in (6, 12, 24, 48, 96, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)

        # ---- 状态机（用自适应波动率阈值，不用固定根数）----
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        vol = rmean(np.abs(np.nan_to_num(r1)), 24)
        thr_evt = np.maximum(3.0 * vol, 0.01)          # 异动阈值 = max(3σ, 1%)
        is_evt = np.abs(np.nan_to_num(r1)) >= thr_evt
        # 状态：0 静默 1 上行 2 下行（用 24 根趋势与波动）
        v24 = rmean(q, 24)
        vr = np.where(v24 > 0, q / np.maximum(v24, 1e-12), 1.0)
        st = np.zeros(n, dtype='int8')
        st[(F['ret24'] > 0.03) & (vr > 1.5)] = 1
        st[(F['ret24'] < -0.03) & (vr > 1.5)] = 2
        st[(np.nan_to_num(F['ret24']) > 0.15) & (vr > 2.0)] = 3   # 拉升
        st[(np.nan_to_num(F['ret24']) < -0.15) & (vr > 2.0)] = 4   # 崩塌
        # 持续时间（全部向量化，勿用 Python 循环：逐根循环在 57 币 × 31 万根上要跑几小时）
        chg = np.zeros(n, dtype='float32')
        chg[1:] = (st[1:] != st[:-1]).astype('float32')
        _ix = np.arange(n)
        _pos = np.where(chg > 0, _ix, 0)
        dur_state = (_ix - np.maximum.accumulate(_pos) + 1).astype('float64')
        _evtpos = np.where(is_evt, _ix, 0)
        dur_evt = (_ix - np.maximum.accumulate(_evtpos)).astype('float64')
        sg = np.sign(np.nan_to_num(r1))
        chg2 = np.zeros(n, dtype='float32')
        chg2[1:] = (sg[1:] != sg[:-1]).astype('float32')
        _pos2 = np.where(chg2 > 0, _ix, 0)
        run_len = (_ix - np.maximum.accumulate(_pos2) + 1).astype('float64')
        run_dir = sg.astype('float64')
        n_trans = pd.Series(chg).rolling(WMAX, min_periods=1).sum().values
        evt_rate = pd.Series(is_evt.astype('float32')).rolling(WMAX, min_periods=1).mean().values
        F['dur_state'] = dur_state
        F['dur_since_evt'] = dur_evt
        F['n_trans'] = n_trans
        F['evt_rate'] = evt_rate
        F['run_dir'] = run_dir
        F['run_len'] = run_len
        F['state_id'] = st

        # 标签：未来 1h 涨 >= 5%
        fu = np.full(n, np.nan); fu[:-12] = c[12:] / c[:-12] - 1.0
        ylab = (fu >= 0.05).astype('int8')
        idx = np.arange(WMAX, n - 12, 6)
        Xb = np.stack([F[k][idx] for k in BASE], axis=1).astype('float32')
        Xe = np.stack([F[k][idx] for k in EXT], axis=1).astype('float32')
        m = ~np.isnan(Xb).any(axis=1) & np.isnan(Xe).sum(axis=1) < 2
        Xe = np.nan_to_num(Xe, nan=0.0, posinf=0.0, neginf=0.0)
        REC.append((sym, Xb[m], Xe[m], ylab[idx[m]], ot[idx[m]]))
    del ch; gc.collect()

log(f'币 {len(REC)}  耗时 {time.time()-t0t:.0f}s')

def run(use_ext, tag):
    Xb = np.concatenate([r[1] for r in REC]); Xe = np.concatenate([r[2] for r in REC])
    Y = np.concatenate([r[3] for r in REC]); T = np.concatenate([r[4] for r in REC])
    X = np.hstack([Xb, Xe]) if use_ext else Xb
    cut = np.quantile(T, 0.70); tr = T < cut
    m_ = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[~tr])[:, 1]
    y = Y[~tr]
    auc = roc_auc_score(y, p)
    log(f'  {tag:<22s} 样本 {len(X):,}  基准 {100*y.mean():.3f}%  AUC={auc:.4f}')
    # 85% 精确率最低阈值
    ok = None
    for thr in np.arange(0.05, 1.0, 0.005):
        sel = p >= thr
        if sel.sum() < 30:
            break
        prec = y[sel].mean()
        if prec >= 0.85:
            ok = (thr, int(sel.sum()), 100*prec, 100*sel.sum()/len(y))
    if ok:
        log(f'      ★ 85%精确率@P>={ok[0]:.3f}  报出 {ok[1]:,}（覆盖 {ok[3]:.3f}%）  精确率 {ok[2]:.1f}%')
    else:
        log('      ✗ 无阈值达到 85% 精确率')
    for thr in (0.3, 0.5, 0.7, 0.9):
        sel = p >= thr; nn = int(sel.sum())
        if nn == 0:
            log(f'      thr={thr}: 无报出'); continue
        log(f'      thr={thr:.1f}  报出 {nn:,}  精确率 {100*y[sel].mean():5.1f}%  召回 {100*y[sel].sum()/max(y.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【对照】基础特征 vs 基础+隐时间结构')
log('=' * 100)
run(False, 'A 基础特征')
run(True, 'B 基础+隐时间结构')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
