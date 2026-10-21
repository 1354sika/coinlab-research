# -*- coding: utf-8 -*-
"""gambler_trade.py — 识别→下注→撤出：真的赚吗？（回答用户三问）
用户口径（2026-10-16）：
  · 判对就是有，判错没惩罚（不下注即可）→ 重点看**精确率**与**净收益**
  · 覆盖率目标 90%
  · 需要庞大样本，不能只在 170 个局里算
做法（g20 口径，局最多）：
  局 = 局部低点起 24h 内涨>=20% 且回吐涨幅 50%（4766 局 / 303 币）
  正例 = 局启动后涨幅 <=20% 的时刻
  训练：时间留出（前 70% / 后 30%）
  评估三件套：
    A 事件级精确率：报出的信号里，真的落在"某局 <=20% 阶段"的比例
    B 事件级覆盖率：全部局里，被至少报出一次的比例
    C 交易模拟：报出即买入，+30% 止盈 / -15% 止损 / 24h 超时；扣 0.04% 手续费 + 0.05% 滑点
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
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_trade.txt')
GAIN = 0.20
WMAX = 288
FEE = 0.0004      # maker 双边
SLIP = 0.0005     # 保守滑点（单边）
TP, SL, TIMEOUT = 0.30, -0.15, 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_trade · 识别→下注→撤出（局=+20%且回吐50%，303 币全量）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 144, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['since_hi24', 'since_lo24', 'run_up', 'run_dn', 'seg_ret', 'seg_volr', 'up_frac24']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

COINS = []
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
        lab = np.zeros(n, dtype='int8'); regs = []; last = -10 ** 9
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
            regs.append((int(t0), int(i1), float(base), float(H)))
            seg = slice(t0, min(n, t0 + 96))
            gsg = c[seg] / base - 1.0
            lab[seg] = ((gsg >= 0.0) & (gsg <= 0.20)).astype('int8')
            last = i1
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
        sl[WMAX - 1:] = WMAX - 1 - Wl.argmin(axis=1); sh[WMAX - 1:] = WMAX - 1 - Wh.argmax(axis=1)
        F['since_lo24'] = sl; F['since_hi24'] = sh
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        sg = np.sign(np.nan_to_num(r1, nan=0.0))
        ru = np.zeros(n); rd = np.zeros(n); cu = cd = 0
        for i in range(n):
            cu = cu + 1 if sg[i] > 0 else 0; cd = cd + 1 if sg[i] < 0 else 0
            ru[i] = cu; rd[i] = cd
        F['run_up'] = ru; F['run_dn'] = rd
        l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
        F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
        qm = rmean(q, WMAX); F['seg_volr'] = np.where(qm > 0, q / np.maximum(qm, 1e-12), np.nan)
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)
        pos = np.flatnonzero(lab == 1)[::3]
        neg_all = np.flatnonzero(lab == 0); neg = neg_all[::24]
        if len(neg) > 3000:
            neg = neg[::max(1, len(neg) // 3000)]
        idx = np.concatenate([pos, neg]) if len(pos) else neg
        if len(idx) == 0:
            continue
        X = np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        COINS.append(dict(sym=sym, X=X[m], y=lab[idx[m]], ot=ot[idx[m]], idxf=idx[m],
                          n=n, regs=regs, c=c, h=h, l=l))
    del ch; gc.collect()
    if len(COINS) % 50 == 0:
        log(f'  已处理 {len(COINS)} 币  耗时 {time.time()-t0t:.0f}s')

X = np.concatenate([c['X'] for c in COINS]); Y = np.concatenate([c['y'] for c in COINS])
TS = np.concatenate([c['ot'] for c in COINS])
cut = np.quantile(TS, 0.70)
tr = TS < cut
log(f'币 {len(COINS)}  局 {sum(len(c["regs"]) for c in COINS)}  样本 {len(X):,}  正例 {int(Y.sum()):,}')

m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=30,
                                    l2_regularization=1.0, random_state=42)
m_.fit(X[tr], Y[tr])
log(f'训练完成（训练 {int(tr.sum()):,}）')

log('')
log('=' * 100)
log('【A】报出信号 → 下注 → 撤出（测试段）')
log('=' * 100)
log(f'  规则：报出即买入；止盈 {100*TP:.0f}% / 止损 {100*SL:.0f}% / 超时 {TIMEOUT*5//60}h；'
    f'扣手续费 {100*FEE:.2f}% + 滑点 {100*SLIP:.2f}%（单边）')
log(f'  {"阈值":>6s} {"信号数":>8s} {"精确率":>8s} {"覆盖率":>8s} {"笔数":>8s} {"期望净%":>9s} {"中位%":>8s} {"胜率":>7s} {"止盈%":>7s} {"止损%":>7s} {"超时%":>7s}')
for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
    trades = []; cover = 0; tot_reg = 0; hit_sig = 0; n_sig = 0
    for c in COINS:
        p = m_.predict_proba(c['X'])[:, 1]
        fire = np.flatnonzero(p >= thr)
        # 事件级：信号是否落在某局 <=20% 阶段（用价格直接判定，不用时间回查）
        for j in fire:
            if c['ot'][j] < cut:
                continue
            n_sig += 1
            px = float(c['c'][c['idxf'][j]])
            for (t0, i1, base, H) in c['regs']:
                if t0 < cut:
                    continue
                if t0 <= c['ot'][j] and px / base - 1.0 <= 0.20:
                    hit_sig += 1; break
        # 覆盖率 + 交易模拟
        for (t0, i1, base, H) in c['regs']:
            if t0 < cut:
                continue
            tot_reg += 1
            msk = (c['ot'] >= t0) & (c['ot'] <= t0 + 30 * 300000) & (p >= thr)
            if not msk.any():
                continue
            cover += 1
            j = int(np.flatnonzero(msk)[0])
            ij = int(c['idxf'][j])
            px = float(c['c'][ij])
            ent = px * (1 + SLIP)
            seg = c['c'][ij:min(c['n'], ij + TIMEOUT)]
            if len(seg) == 0:
                continue
            ret = seg / ent - 1.0
            idx_tp = np.flatnonzero(ret >= TP)
            idx_sl = np.flatnonzero(ret <= SL)
            if len(idx_tp) and (not len(idx_sl) or idx_tp[0] <= idx_sl[0]):
                g = TP
            elif len(idx_sl):
                g = SL
            else:
                g = ret[-1]
            trades.append(g - FEE - 2 * SLIP)
    if n_sig == 0:
        log(f'  {thr:>6.2f}  无信号'); continue
    A = np.array(trades) if trades else np.array([np.nan])
    log(f'  {thr:>6.2f} {n_sig:>8,} {100*hit_sig/max(n_sig,1):>7.1f}% {100*cover/max(tot_reg,1):>7.1f}% '
        f'{len(A):>8,} {100*np.nanmean(A):>8.2f}% {100*np.nanmedian(A):>7.2f}% '
        f'{100*np.nanmean(A > 0):>6.1f}% {100*np.nanmean(A > TP*0.99):>6.1f}% '
        f'{100*np.nanmean(A < SL*0.99):>6.1f}% {100*np.nanmean(np.abs(A) < 0.05):>6.1f}%')

log('')
log('【B】判读')
log('  · 精确率 = 报出的信号里，真的落在"某局 <=20% 阶段"的比例（判对就是有）')
log('  · 覆盖率 = 全部局里被至少报出一次的比例（用户目标 90%）')
log('  · 期望净% = 每次下注的平均净收益（已扣手续费与滑点）')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
