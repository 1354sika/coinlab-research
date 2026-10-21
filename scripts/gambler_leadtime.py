# -*- coding: utf-8 -*-
"""gambler_leadtime.py — 预警提前量：能提前多久判断"局要开始了"？
用户问题（2026-10-16）：
  "分辨的时间窗口是多少？大概多久能分辨一个币在一段时间内的波动是不是局的开始？
   如果只能从周期的开始和结束分辨，那几乎没有意义"

★ 本脚本直接回答这个问题：用**同一套特征**，只改变**标签相对局起点的时间偏移**，
  测出"提前量 → 精确率"曲线。这是判断"模型有没有实战价值"的核心实验。

  档   局起点 t0 落在 t 之后           含义                提前量
  L0   t0 ∈ [t+0,   t+12 ]            现在就是起点         0 ~ 1h
  L1   t0 ∈ [t+12,  t+24 ]            1 小时后开始         1 ~ 2h
  L2   t0 ∈ [t+24,  t+48 ]            2~4 小时后开始       2 ~ 4h
  L3   t0 ∈ [t+48,  t+96 ]            4~8 小时后开始       4 ~ 8h
  L4   t0 ∈ [t+96,  t+288]            8~24 小时后开始      8 ~ 24h
  L5   t0 ∈ [t+288, t+864]            1~3 天后开始         1 ~ 3 天

  特征全部只用 t 及之前（严格无泄漏；标签与特征零重叠——上一轮已修正三处泄漏）。
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
OUT = os.path.join(EV, 'gambler_leadtime.txt')

WMAX = 288
LEADS = [(0, 12, 'L0 现在就是起点(0~1h)'), (12, 24, 'L1 提前 1~2h'),
         (24, 48, 'L2 提前 2~4h'), (48, 96, 'L3 提前 4~8h'),
         (96, 288, 'L4 提前 8~24h'), (288, 864, 'L5 提前 1~3天')]

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_leadtime · 预警提前量 vs 精确率（能提前多久？）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96',
         'll_d1','ll_d3','ll_d6','ll_d12','ll_d24','ll_d48','ll_d96','ll_d288',
         'll_rank','near_ll_run','ll_vol',
         'seg_ret','evt_rate','n_trans','dur_state','run_len','accel',
         'sync_obs','hour_sin','hour_cos']

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
sync_obs_g = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

# 每档收集：X、y、时间、币
BANK = {L[2]: {'X': [], 'Y': [], 'T': [], 'S': []} for L in LEADS}
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
    pos = np.clip(np.searchsorted(ts, ot, side='left'), 0, T - 1)
    F['sync_obs'] = sync_obs_g[pos]
    hh = ((ot // 3600000) % 24) / 24.0 * 2 * np.pi
    F['hour_sin'] = np.sin(hh); F['hour_cos'] = np.cos(hh)

    for (d0, d1, tag) in LEADS:
        yl = np.zeros(n, dtype='int8')
        for (t0, i1) in starts:
            lo = max(0, t0 - d1); hi = min(n, t0 - d0)
            if hi > lo: yl[lo:hi] = 1
        idxs = np.arange(288, n - 12, 6)
        X = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in FEAT], axis=1).astype('float32')
        y = yl[idxs]
        keep = np.flatnonzero(y == 1); kneg = np.flatnonzero(y == 0)[::12]
        sel = np.concatenate([keep, kneg])
        if len(keep) < 10: continue
        B = BANK[tag]
        B['X'].append(X[sel]); B['Y'].append(y[sel]); B['T'].append(ot[idxs][sel])
        B['S'].append(np.array([sym] * len(sel), dtype=object))
    del F; gc.collect()

log('')
log('=' * 100)
log('【核心结果】各提前量的可预测性（时间留出：前 70% 训练 → 后 30% 测试）')
log('=' * 100)
log(f'{"提前档":<26s} | {"测试样本":>9s} {"基准%":>7s} {"AUC":>7s} | {"85%可达":>22s} | {"thr=0.5 报出/精确率":>22s}')
for (d0, d1, tag) in LEADS:
    B = BANK[tag]
    if not B['X']:
        log(f'{tag:<26s} | 无样本'); continue
    X = np.concatenate(B['X']); Y = np.concatenate(B['Y']); Tt = np.concatenate(B['T'])
    cut = np.quantile(Tt, 0.70); tr = Tt < cut; te = ~tr
    if Y[tr].sum() < 40 or Y[te].sum() < 20:
        npos_tr = int(Y[tr].sum()); npos_te = int(Y[te].sum())
        log(f'{tag:<26s} | 正例不足（训 {npos_tr} 测 {npos_te}）'); continue
    m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[te])[:, 1]; yy = Y[te]
    auc = roc_auc_score(yy, p)
    ok = None
    for thr in np.arange(0.05, 0.995, 0.005):
        s = p >= thr
        if s.sum() < 5: break
        if yy[s].mean() >= 0.85:
            ok = (thr, int(s.sum()), 100*yy[s].mean(), 100*s.sum()/max(yy.sum(), 1))
            break
    s5 = p >= 0.5; nn5 = int(s5.sum())
    p5 = 100*yy[s5].mean() if nn5 else float('nan')
    okstr = f'@P>={ok[0]:.3f} 报出{ok[1]:,} 覆盖{ok[3]:.1f}%' if ok else '✗ 不可达'
    log(f'{tag:<26s} | {len(yy):>9,d} {100*yy.mean():7.3f} {auc:7.4f} | {okstr:>22s} | {nn5:>6,d} / {p5:5.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · L0 = 当前位置就是起点（提前 0~1h）—— 这档有意义才有实战价值')
log('  · L1~L5 = 提前 1h 到 3 天；精度随提前量衰减的速度决定"能不能提前布局"')
log('  · 若只有 L0 能做而 L1 以后立刻崩，说明模型只是"确认现在"，没有真正的领先能力')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
