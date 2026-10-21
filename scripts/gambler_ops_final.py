# -*- coding: utf-8 -*-
"""gambler_ops_final.py — 操盘判别（严格无泄漏版）
用户要求（2026-10-16）：
  "先实现操盘判别，但是要确保，操盘判别的成果不是因为未来信息泄露"
用户答复（ask_user）："先不投钱，只做出可验证的规则，实盘以后再说"

★★ 本版审计出并修正的泄漏 ★★
  旧版 sync_now = "过去 2h 内有多少币**形成局起点**"。
  但"局起点"是**事后判定**的（要看未来 288 根涨幅）——t 时刻根本不可能知道。
  ⇒ 等于把事后信息喂进了特征。
  本版改为：sync_now = "过去 2h 内有多少币 1h 涨 >= 5%"（t 时刻完全可观测）。

操盘标签（标签可用未来，特征不许）：
  ops = 该币在未来 EARLY 根内形成局（涨幅 >= 10%）
        且 该时刻同刻全市场有 >= 3 个币 1h 涨 >= 5%（只用同刻数据）

三重验证：
  A 时间留出：前 70% 时间训练 → 后 30% 测试
  B 币留出：奇序号币训练 → 偶序号币测试（币完全不重叠）
  C 特征审计表：逐项打印"该特征依赖哪个时间区间"
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
OUT = os.path.join(EV, 'gambler_ops_final.txt')

WMAX = 288
EARLY = 12

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_ops_final · 操盘判别（严格无泄漏版）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96','ll6','ll12','ll24','ll96',
         'seg_ret','evt_rate','n_trans','dur_state','run_len','accel',
         'sync_obs','hour_sin','hour_cos']

# 特征来源审计表（给用户核验）
AUDIT = {
    'ret{k}':      '过去 k 根收盘比（含当前）',
    'volr{k}':     '当前成交额 / 过去 k 根均量',
    'trr{k}':      '当前笔数 / 过去 k 根均笔数',
    'hh{k}':       '当前价 / 过去 k 根最高 - 1（<=0）',
    'll{k}':       '当前价 / 过去 k 根最低 - 1（>=0）',
    'seg_ret':     '当前价 / 过去 288 根最低 - 1',
    'evt_rate':    '过去 48 根里 |r1|>3σ 的比例',
    'n_trans':     '过去 48 根里涨跌方向切换次数',
    'dur_state':   '当前状态已持续根数（状态由过去信息判定）',
    'run_len':     '当前同向连续根数',
    'accel':       'ret6 - ret24/4（都是过去）',
    'sync_obs':    '过去 2h 内全市场 1h 涨>=5% 的币数（同刻可观测）',
    'hour_sin/cos':'UTC 小时（时间戳本身）',
}

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
Q = pd.DataFrame({r[0]: pd.Series(r[5], index=r[1]) for r in RAW}).sort_index()
TR = pd.DataFrame({r[0]: pd.Series(r[6], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

# ★ 修正后的 sync_obs：只用同刻可观测的"1h 涨>=5% 币数"
R1_all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R1_all, nan=0.0) >= 0.05).sum(axis=1)
sync_obs_global = pd.Series(up_cnt).rolling(24, min_periods=1).max().values   # 过去 2h 内的最大数
log(f'观测同步度：均值 {sync_obs_global.mean():.1f}  中位 {np.median(sync_obs_global):.0f}  '
    f'p90 {np.percentile(sync_obs_global,90):.0f}')

REC = []
for ri, (sym, ot, c, h, l, q, tr) in enumerate(RAW):
    n = len(c)
    W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= W.min(axis=1) + 1e-12
    hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    cand = np.flatnonzero(is_low & np.isfinite(hmax))
    last = -10 ** 9
    starts = []
    for t0 in cand:
        if t0 - last < WMAX: continue
        H = hmax[t0]; base = max(l[t0], 1e-12)
        if H / base - 1.0 < 0.10: continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 96)
        if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (H - base) * 0.5):
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
        F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2,k//4)).min().values / c - 1.0
    l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
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
    # ★ 修正后的 sync_obs：把全局"过去 2h 1h涨>=5% 币数"贴到本币时间轴
    pos = np.searchsorted(ts, ot, side='left')
    pos = np.clip(pos, 0, T - 1)
    F['sync_obs'] = sync_obs_global[pos]

    posE = np.zeros(n, dtype=bool)
    for (t0, i1) in starts: posE[t0:min(n, t0+EARLY+1)] = True
    # ★★ 第三次泄漏修正：标签不得包含任何已作为特征的量。
    #    旧版 yO = posE & (sync_obs>=3)，而 sync_obs 同时是特征
    #    ⇒ 模型只要学到 sync_obs>=3 就完美预测标签的一半，AUC 虚高到 0.999。
    #    本版标签 = 纯未来信息（本币未来 EARLY 根内形成局），与特征零重叠。
    yO = posE.astype('int8')

    idxs = np.arange(288, n - 12, 6)
    X = np.stack([np.nan_to_num(F[k][idxs], nan=0.0) for k in FEAT], axis=1).astype('float32')
    y = yO[idxs]
    keep = np.flatnonzero(y == 1); kneg = np.flatnonzero(y == 0)[::8]
    sel = np.concatenate([keep, kneg])
    if len(sel) < 30: continue
    REC.append((sym, X[sel], y[sel], ot[idxs][sel]))
    del F; gc.collect()

log(f'样本组 {len(REC)}   耗时 {time.time()-t0t:.0f}s')
log('')
log('=' * 100)
log('【特征审计表】每个特征依赖的时间区间')
log('=' * 100)
for k, v in AUDIT.items():
    log(f'  {k:<14s} → {v}')

X = np.concatenate([r[1] for r in REC]); Y = np.concatenate([r[2] for r in REC])
Tt = np.concatenate([r[3] for r in REC])
S = np.concatenate([[r[0]] * len(r[1]) for r in REC])
log('')
log(f'总样本 {len(X):,}  正例率 {100*Y.mean():.3f}%（正例 {int(Y.sum()):,}）')

def fit_eval(tr, te, tag):
    if Y[tr].sum() < 40 or Y[te].sum() < 20:
        log(f'  {tag}: 正例不足（训 {int(Y[tr].sum())} 测 {int(Y[te].sum())}）'); return
    m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[te])[:, 1]; yy = Y[te]
    auc = roc_auc_score(yy, p)
    log(f'  {tag:<26s} 训 {int(tr.sum()):>7,d} 测 {int(te.sum()):>7,d}  基准 {100*yy.mean():5.2f}%  AUC={auc:.4f}')
    for thr in (0.5, 0.7, 0.8):
        s = p >= thr; nn = int(s.sum())
        if nn == 0:
            log(f'      thr={thr:.1f} 无报出'); continue
        log(f'      thr={thr:.1f} 报出 {nn:>6,d} 精确率 {100*yy[s].mean():5.1f}% 召回 {100*yy[s].sum()/max(yy.sum(),1):5.1f}%')

log('')
log('=' * 100)
log('【验证 A】时间留出（前 70% 训练 → 后 30% 测试）')
log('=' * 100)
cut = np.quantile(Tt, 0.70)
fit_eval(Tt < cut, Tt >= cut, 'A 时间留出')

log('')
log('=' * 100)
log('【验证 B】币留出（奇序币训练 → 偶序币测试，币完全不重叠）')
log('=' * 100)
uniq = sorted(set(S))
odd = set(uniq[0::2]); even = set(uniq[1::2])
trB = np.array([s in odd for s in S]); teB = np.array([s in even for s in S])
fit_eval(trB, teB, 'B 币留出')

log('')
log('=' * 100)
log('【验证 C】双重留出（时间 + 币都不重叠）')
log('=' * 100)
fit_eval(trB & (Tt < cut), teB & (Tt >= cut), 'C 时间+币双留出')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
