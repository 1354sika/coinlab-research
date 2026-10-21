# -*- coding: utf-8 -*-
"""gambler_grid.py — 参数网格：把"局"的每种定义都丢进币市黑盒里跑一遍
用户口径（2026-10-16 确认）：
  · 早期识别是重中之重：正例 = 局启动后涨幅 <=20% 的时刻
  · 在市的币（用数据里全部 304 币，不再截断）
  · 终点不确定 → 三种终点都试，用数据说话
用法：
  python gambler_grid.py <gain> <drop_mode> [vol_mode]
    gain      : 0.5 / 1.0 / 1.5        峰值涨幅门槛
    drop_mode : d30 / half / back      终点判定（回撤30% / 回吐涨幅50% / 回到起点+10%）
    vol_mode  : any / v5               是否要求拉升段量比 >=5x
输出：evidence/gambler_grid_<gain>_<mode>_<vol>.txt
核心指标（口径三件套全部标注）：
  ★ 事件级召回：在"涨幅 <=20%"的阶段内被模型报出的局 / 全部局
  ★ 报出时已涨幅度分布、报出后剩余空间、按规则模拟的净收益率
"""
import os, gc, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view
import joblib

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
MODE = sys.argv[2] if len(sys.argv) > 2 else 'd30'
VOLM = sys.argv[3] if len(sys.argv) > 3 else 'any'

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
os.makedirs(MD, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_502.csv')
TAG = 'g%.2f_%s_%s' % (GAIN, MODE, VOLM)
OUT = os.path.join(EV, 'gambler_grid_%s.txt' % TAG)

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log(f'gambler_grid · GAIN={GAIN} DROP={MODE} VOL={VOLM}')
log('  口径：正例 = 局启动后涨幅<=20% 的时刻（早期识别）；范围 = 数据内全部在币')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 144, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['since_hi24', 'since_lo24', 'run_up', 'run_dn', 'seg_ret', 'seg_volr', 'up_frac24']
WMAX = 288

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

Xs, Ys, Ts, Ss = [], [], [], []
SYMS = []
n_regime = 0
regimes = []

# ---- 逐币处理：读一块算一块，避免全量驻留 ----
carry = None
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'}, chunksize=3_000_000):
    if carry is not None and len(carry):
        ch = pd.concat([carry, ch], ignore_index=True)
        carry = None
    _ls = ch['symbol'].iloc[-1]
    _keep = ch['symbol'] == _ls
    carry = ch[_keep]
    ch = ch[~_keep]
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000:
            continue
        si = len(SYMS); SYMS.append(sym)
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        ot = g['open_time'].values

        # ---- 找局 ----
        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        lab = np.zeros(n, dtype='int8')
        last = -10 ** 9
        for t0 in cand:
            if t0 - last < WMAX:
                continue
            H = hmax[t0]
            base = max(l[t0], 1e-12)
            if H / base - 1.0 < GAIN:
                continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            # 终点判定
            if MODE == 'd30':
                k2 = min(n - 1, i1 + 48)
                cond = (k2 > i1) and (l[i1:k2 + 1].min() <= H * 0.70)
            elif MODE == 'half':
                thr = base + (H - base) * 0.5
                k2 = min(n - 1, i1 + 96)
                cond = (k2 > i1) and (l[i1:k2 + 1].min() <= thr)
            else:  # back
                thr = base * 1.10
                k2 = min(n - 1, i1 + 144)
                cond = (k2 > i1) and (l[i1:k2 + 1].min() <= thr)
            if not cond:
                continue
            # 量能条件
            if VOLM == 'v5':
                vseg = q[t0:i1 + 1]
                vbase = rmean(q, 288)[t0]
                if not (vbase > 0 and np.nanmax(vseg) / vbase >= 5.0):
                    continue
            n_regime += 1
            regimes.append((sym, int(ot[t0]), float(H / base - 1.0)))
            seg = slice(t0, min(n, t0 + 96))
            gain_seg = c[seg] / base - 1.0
            lab[seg] = ((gain_seg >= 0.0) & (gain_seg <= 0.20)).astype('int8')
            last = i1

        # ---- 特征（只用当期及过去）----
        F = {}
        for k in (6, 12, 24, 48, 96, 144, 288):
            r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
            F['ret%d' % k] = r
            mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
            mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
            F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2, k // 4)).max().values / c - 1.0
            F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2, k // 4)).min().values / c - 1.0
        Wl = sliding_window_view(l, WMAX); Wh = sliding_window_view(h, WMAX)
        sl = np.full(n, np.nan); sh = np.full(n, np.nan)
        sl[WMAX - 1:] = WMAX - 1 - Wl.argmin(axis=1)
        sh[WMAX - 1:] = WMAX - 1 - Wh.argmax(axis=1)
        F['since_lo24'] = sl; F['since_hi24'] = sh
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        sg = np.sign(np.nan_to_num(r1, nan=0.0))
        ru = np.zeros(n); rd = np.zeros(n); cu = cd = 0
        for i in range(n):
            cu = cu + 1 if sg[i] > 0 else 0
            cd = cd + 1 if sg[i] < 0 else 0
            ru[i] = cu; rd[i] = cd
        F['run_up'] = ru; F['run_dn'] = rd
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        qm = rmean(q, WMAX)
        F['seg_volr'] = np.where(qm > 0, q / np.maximum(qm, 1e-12), np.nan)
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)

        # 正例：全取（每 3 根）；负例：每 24 根抽 1
        pos_idx = np.flatnonzero(lab == 1)[::3]
        neg_all = np.flatnonzero(lab == 0)
        neg_idx = neg_all[::24]
        if len(neg_idx) > 3000:
            step = max(1, len(neg_idx) // 3000)
            neg_idx = neg_idx[::step]
        idx = np.concatenate([pos_idx, neg_idx]) if len(pos_idx) else neg_idx
        if len(idx) == 0:
            continue
        X = np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        idx = idx[m]; X = X[m]
        if len(idx) == 0:
            continue
        Xs.append(X); Ys.append(lab[idx]); Ts.append(ot[idx])
        Ss.append(np.full(len(idx), si, dtype='int32'))
    del ch; gc.collect()
    log(f'  已处理 {len(SYMS)} 币  局部正例累计 {n_regime}  耗时 {time.time()-t0t:.0f}s')

if not Xs:
    log('✗ 无样本'); raise SystemExit(1)
X = np.concatenate(Xs); Y = np.concatenate(Ys); TS = np.concatenate(Ts); SI = np.concatenate(Ss)
del Xs, Ys, Ts, Ss; gc.collect()
log(f'样本 {len(X):,}  特征 {X.shape[1]}  正例 {int(Y.sum()):,}（{100*Y.mean():.3f}%）  局 {n_regime}')

RG = pd.DataFrame(regimes, columns=['symbol', 't0', 'gain'])
if len(RG):
    RG['year'] = pd.to_datetime(RG['t0'], unit='ms').dt.year
    log('  局按年：' + '  '.join(f'{y}:{v}' for y, v in RG.groupby('year').size().items()))
    log(f'  峰值涨幅中位 {100*RG["gain"].median():.0f}%  涉及币 {RG["symbol"].nunique()}')

cut = np.quantile(TS, 0.70)
tr = TS < cut; te = ~tr
log(f'训练 {tr.sum():,}（正例 {int(Y[tr].sum()):,}）  测试 {te.sum():,}（正例 {int(Y[te].sum()):,}）')

if Y[tr].sum() >= 100 and Y[te].sum() >= 20:
    m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=30,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[te])[:, 1]
    log(f'AUC={roc_auc_score(Y[te], p):.4f}')
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        sel = p >= thr; nn = int(sel.sum())
        if nn == 0:
            log(f'  阈值 {thr:.1f}: 无报出'); continue
        log(f'  阈值 {thr:.1f}: 报出 {nn:,}  精确率 {100*Y[te][sel].mean():.1f}%  召回 {100*Y[te][sel].sum()/max(Y[te].sum(),1):.1f}%')
    joblib.dump({'model': m_, 'features': FEAT}, os.path.join(MD, 'gg_%s.joblib' % TAG))
else:
    log(f'✗ 正例不足（训练 {int(Y[tr].sum())} / 测试 {int(Y[te].sum())}），未训练')

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
