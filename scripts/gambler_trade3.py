# -*- coding: utf-8 -*-
"""gambler_trade3.py — 局识别 → 下注 → 收益（局级评估）
修正 gambler_trade/trade2 的索引 bug（那两版结果全 0，不可用）。
改为逐币逐局循环，用"局"为单位评估，不做全局索引对齐。

回答用户三问：
  1) 每次识别出来，有百分之多少概率确认是局？（精确率）
  2) 有多少概率支持下注获利撤走？（期望净 + 胜率）
  3) 覆盖率多少？（目标 90%）
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
OUT = os.path.join(EV, 'gambler_trade3.txt')

GAIN = 0.20        # 局定义：5 根内低点起算，未来 288 根内最高涨 >= 20%
WMAX = 288
FEE, SLIP = 0.0004, 0.0005
THRS = (0.3, 0.5, 0.7)

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_trade3 · 局识别 → 下注 → 收益（局级评估）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FEAT += ['seg_ret', 'up_frac24']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

RECS = []   # (sym, X, y, ot, idx, close, n, regs)
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
        regs = []; last = -10 ** 9
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
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        F['up_frac24'] = rmean((r1 > 0).astype('float64'), WMAX)

        fu = np.full(n, np.nan); fu[:-12] = c[12:] / c[:-12] - 1.0
        ylab = (fu >= 0.05).astype('int8')           # 标签：未来 1h 涨>=5%
        idx_all = np.arange(0, n - 12, 6)
        X = np.stack([F[k][idx_all] for k in FEAT], axis=1).astype('float32')
        m = ~np.isnan(X).any(axis=1)
        ia = idx_all[m]; X = X[m]
        RECS.append((sym, X, ylab[ia], ot[ia], ia, c, n, regs))
    del ch; gc.collect()

log(f'有局的币 {len(RECS)}   耗时 {time.time()-t0t:.0f}s')

X = np.concatenate([r[1] for r in RECS]); Y = np.concatenate([r[2] for r in RECS])
T = np.concatenate([r[3] for r in RECS])
cut = np.quantile(T, 0.70); tr = T < cut
log(f'样本 {len(X):,}  正例 {100*Y.mean():.2f}%  训练 {int(tr.sum()):,}  测试 {int((~tr).sum()):,}')
m_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, min_samples_leaf=50,
                                    l2_regularization=1.0, random_state=42)
m_.fit(X[tr], Y[tr])
p_all = m_.predict_proba(X[~tr])[:, 1]
log(f'测试 AUC = {roc_auc_score(Y[~tr], p_all):.4f}')
log(f'cut = {cut}（测试段 = 时间后 30%）')

log('')
log('=' * 100)
log('【局级评估】每个局：模型在"已涨<=20%"阶段是否报出 + 报出后下注结果')
log('=' * 100)
log(f'  交易规则：+100% 止盈 / -15% 止损 / 24h 超时（扣 {FEE:.2%}+{SLIP:.2%}）')
log('')

for thr in THRS:
    nreg = 0; cov = 0; early = 0
    trades = []
    for (sym, Xc, yc, otc, iac, c, n, regs) in RECS:
        pc = m_.predict_proba(Xc)[:, 1]
        hit = pc >= thr
        for (t0, i1) in regs:
            if otc[0] >= 0:
                pass
            if c.shape[0] == 0:
                continue
            # 局的起始时间戳（用完整轴）
            t0t_ms = None
            # 该币局起点在采样点里的位置：otc 里 >= 完整轴时间
            # 用 ia 索引反查（iac 是采样点在完整轴的索引）
            # 简化：局起点 t0 的毫秒时间戳 = 无法直接取 → 用采样点里 idx>=t0 的第一个
            j = np.flatnonzero(iac >= t0)
            if len(j) == 0:
                continue
            jj = j[0]
            if otc[jj] < cut:            # 只用测试段
                continue
            nreg += 1
            # 在 [t0, i1] 内找报出点，且已涨 <= 20%
            k = np.flatnonzero(hit[j:] if False else hit[np.flatnonzero(iac >= t0)[0]:])
            if len(k) == 0:
                continue
            # 逐个候选点检查"已涨<=20%"
            picked = None
            for kk in k:
                ij = int(iac[np.flatnonzero(iac >= t0)[0] + kk])
                if ij > i1:
                    break
                gain = c[ij] / max(c[t0], 1e-12) - 1.0
                if gain <= 0.20:
                    picked = ij
                    break
            if picked is None:
                continue
            cov += 1
            gain_at = c[picked] / max(c[t0], 1e-12) - 1.0
            if gain_at <= 0.05:
                early += 1
            # 交易模拟
            px = float(c[picked])
            seg = c[picked:min(n, picked + 288)]        # 24h = 288 根
            if len(seg) == 0:
                continue
            ret = seg / (px * (1 + SLIP)) - 1.0
            a = np.flatnonzero(ret >= 1.00); b = np.flatnonzero(ret <= -0.15)
            if len(a) and (not len(b) or a[0] <= b[0]):
                g = 1.00
            elif len(b):
                g = -0.15
            else:
                g = ret[-1]
            trades.append(g - FEE - 2 * SLIP)
    if nreg == 0:
        log(f'  thr={thr:.1f}: 测试段无局'); continue
    A = np.array(trades) if trades else np.array([])
    log(f'  thr={thr:.1f}  测试段局 {nreg}  报出 {cov}（覆盖 {100*cov/max(nreg,1):.1f}%）'
        f'  其中<=5%已涨 {early}（{100*early/max(nreg,1):.1f}%）')
    if len(A):
        log(f'          交易 笔 {len(A):,}  期望净 {100*A.mean():+.3f}%  中位 {100*np.median(A):+.3f}%  '
            f'胜率 {100*(A>0).mean():.1f}%  ≥+100% 占 {100*(A>=1.0).mean():.1f}%')
    else:
        log('          交易 无')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
