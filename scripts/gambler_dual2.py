# -*- coding: utf-8 -*-
"""gambler_dual2.py — 三档局（大/小/小小）+ 操盘 vs 非操盘
用户要求（2026-10-16）：
  "同时不要忘记继续挖掘小小局。我们的模型应该是能看大小局，能分操盘与非操盘（也就是散户博弈）"

做法：
  1) 三档局（都用"低点起 288 根内涨幅"）：
       大局   gain >= 20% 且 之后自峰腰斩
       小局   gain >= 10%（且非大局）
       小小局 gain >= 5%（且非小局）
  2) 操盘标签（用户要的"分操盘与非操盘"）：
       操盘   局 且 起点同刻全市场 >=3 个币也在局起点（群体型 = 有组织）
       非操盘 其余
  3) 训 4 个二分类器：大局 / 小局 / 小小局 / 操盘
  4) 报告 AUC + 各阈值精确率/召回 + 85% 可达性
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
OUT = os.path.join(EV, 'gambler_dual2.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_dual2 · 三档局（大/小/小小）+ 操盘 vs 非操盘')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96','ll6','ll12','ll24','ll96',
         'seg_ret','up_frac24','evt_rate','n_trans','dur_state','run_len','accel',
         'sync_now','hour_sin','hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

# 第一遍：算出每个币的局，并把"局起点时刻"汇总成全局集合（用于算同步度）
RAW = []      # (sym, n, c, h, l, q, tr, ot, bigs, smalls, tinys)
STARTS_ALL = {}
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
        bigs, smalls, tinys = [], [], []
        for t0 in cand:
            if t0 - last < WMAX: continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            gain = H / base - 1.0
            if gain < 0.05: continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            k2 = min(n - 1, i1 + 96)
            cutd = (k2 > i1 and l[i1:k2 + 1].min() <= base + (H - base) * 0.5)
            if gain >= 0.20 and cutd:
                bigs.append((t0, i1))
            elif gain >= 0.10:
                smalls.append((t0, i1))
            else:
                tinys.append((t0, i1))
            last = i1
            STARTS_ALL.setdefault(int(ot[t0] // 300000), 0)
            STARTS_ALL[int(ot[t0] // 300000)] += 1
        if not (bigs or smalls or tinys):
            continue
        RAW.append((sym, n, c, h, l, q, tr, ot, bigs, smalls, tinys))
    del ch; gc.collect()

log(f'币 {len(RAW)}   全局局起点槽 {len(STARTS_ALL):,}   耗时 {time.time()-t0t:.0f}s')
slot_keys = np.array(sorted(STARTS_ALL))
slot_cnt = np.array([STARTS_ALL[k] for k in slot_keys])

def sync_at(t_ms, w=12):
    lo = t_ms // 300000 - w; hi = t_ms // 300000 + w
    a = np.searchsorted(slot_keys, lo); b = np.searchsorted(slot_keys, hi + 1)
    return int(slot_cnt[a:b].sum())

REC = []
for (sym, n, c, h, l, q, tr, ot, bigs, smalls, tinys) in RAW:
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F = {}
    for k in (6,12,24,48,96,288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    for k in (6,12,24,96):
        F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
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
    # sync_now：当前时刻**之前 2h 内**全市场局起点数（只回溯、不看未来，避免前视泄漏）
    slots = (ot // 300000)
    acc = np.concatenate([[0], np.cumsum(slot_cnt)])
    lo = np.searchsorted(slot_keys, slots - 24, side='left')   # 过去 2h
    hi = np.searchsorted(slot_keys, slots + 1, side='right')   # 截至当前时刻（不含未来）
    F['sync_now'] = acc[hi] - acc[lo]

    posB = np.zeros(n, dtype=bool); posS = np.zeros(n, dtype=bool); posT = np.zeros(n, dtype=bool)
    for (t0, i1) in bigs:   posB[t0:min(n, t0+EARLY+1)] = True
    for (t0, i1) in smalls: posS[t0:min(n, t0+EARLY+1)] = True
    for (t0, i1) in tinys:  posT[t0:min(n, t0+EARLY+1)] = True
    posS &= ~posB; posT &= ~(posB | posS)

    idxs = np.arange(WMAX, n - 12, 6)
    X = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in FEAT], axis=1).astype('float32')
    yB = posB[idxs].astype('int8'); yS = posS[idxs].astype('int8'); yT = posT[idxs].astype('int8')
    yO = ((posB[idxs] | posS[idxs] | posT[idxs]) & (F['sync_now'][idxs] >= 2)).astype('int8')
    keep = np.flatnonzero(yB | yS | yT)
    kneg = np.flatnonzero(~(yB | yS | yT))[::8]
    sel = np.concatenate([keep, kneg])
    if len(sel) < 30: continue
    REC.append((sym, X[sel], yB[sel], yS[sel], yT[sel], yO[sel], ot[idxs][sel]))
    del F; gc.collect()

log(f'样本组 {len(REC)}   耗时 {time.time()-t0t:.0f}s')

def run(yi, tag):
    X = np.concatenate([r[1] for r in REC])
    Y = np.concatenate([r[yi] for r in REC])
    T = np.concatenate([r[6] for r in REC])
    cut = np.quantile(T, 0.70); tr = T < cut
    if Y[tr].sum() < 40 or Y[~tr].sum() < 20:
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
    line = f'  {tag:<16s} 样本 {len(X):>8,d}  正例 {100*Y.mean():6.3f}%  AUC={auc:.4f}'
    if ok: line += f'  ★85% @P>={ok[0]:.3f} 报出{ok[1]:,}(覆盖{ok[3]:.3f}%)'
    else:  line += '  ✗无阈值达85%'
    log(line)
    for thr in (0.3, 0.5, 0.7):
        s = p >= thr; nn = int(s.sum())
        if nn == 0: continue
        log(f'      thr={thr:.1f} 报出 {nn:,} 精确率 {100*yy[s].mean():5.1f}% 召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【三档局 + 操盘判别】')
log('=' * 100)
run(2, '大局(+20%腰斩)')
run(3, '小局(+10%)')
run(4, '小小局(+5%)')
run(5, '操盘(局&同步>=3)')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
