# -*- coding: utf-8 -*-
"""gambler · 做空闭环 + 尾部赌法（一次读数据跑两个实验，回答"怎么才能赚几倍"）

用户口径（2026-10-16）：
  · 要的是**期望净收益率**；发现操盘者→前期跟盘→提前撤出→几倍收益
  · 讨论出的两条未验证路线：
     (1) 做空侧 —— "暴涨必暴跌、且崩得快（10~20 分钟）"，崩得快对做多灾难、对做空优势
     (2) 尾部赌法 —— 低命中 + 高盈亏比 + 严止损：买 20 次中 1 次真局，真局吃 200%，
         其余 −10% 止损 ⇒ 0.05×2.00 − 0.95×0.10 = +0.5%

数据：klines_5m_502.csv（304 币 × 3 年 5m）  全部特征 t 时刻可见，无事件定义、无泄漏
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD = 6, 24
DN, UP = -0.20, 0.20
FEE = 0.0004
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 做空闭环 + 尾部赌法（无前视）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
def _piv(c):
    m = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
    return m.where(m > 0).ffill(limit=3)
close, high, low = _piv('close'), _piv('high'), _piv('low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
qv = qv.where(qv >= 0)
del df; gc.collect()
T, N = close.shape
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
F = {}
for w in (3, 12, 48, 288):
    a = np.full((T, N), np.nan, np.float32); a[w:] = CV[w:] / CV[:-w] - 1.0
    F['ret%d' % w] = a
for w in (12, 48, 192):
    med = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
    med = np.where(med > 0, med, np.nan)
    F['volr%d' % w] = (QV / med).astype('float32')
for w in (48, 192):
    F['vol%d' % w] = pd.DataFrame(ret1).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
for w in (48, 288):
    hi = pd.DataFrame(HV).rolling(w, min_periods=w // 4).max().values
    lo = pd.DataFrame(LV).rolling(w, min_periods=w // 4).min().values
    F['dhi%d' % w] = (CV / hi - 1.0).astype('float32')
    F['dlo%d' % w] = (CV / lo - 1.0).astype('float32')
sgn = np.sign(np.nan_to_num(ret1)); idxr = np.arange(T)
d1 = np.zeros((T, N), np.float32)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (idxr - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
F['D1'] = d1
L = np.maximum(np.abs(d1).astype(np.int64), 1)
sr = np.clip(idxr[:, None] - (L - 1), 0, T - 1)
sp = np.take_along_axis(CV, sr, axis=0)
F['D2'] = (((CV / sp - 1.0) / L) * 100).astype('float32')
F['D3'] = ((CV / sp - 1.0) * 100).astype('float32')
sd = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
F['D4'] = (np.abs(ret1) > 3.0 * sd).astype('float32')
F['D5'] = (np.abs(ret1) <= 0.5 * sd).astype('float32')
F['rk_ret48'] = pd.DataFrame(F['ret48']).rank(axis=1, pct=True).values.astype('float32')
F['rk_volr12'] = pd.DataFrame(F['volr12']).rank(axis=1, pct=True).values.astype('float32')
mkt = np.nanmedian(ret1, axis=1); disp = np.nanstd(ret1, axis=1)
F['mkt48'] = np.repeat(pd.Series(mkt).rolling(48, min_periods=8).sum().values[:, None], N, axis=1).astype('float32')
F['D8'] = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
NAME = list(F.keys())
say('特征 %d 个  %.0fs' % (len(NAME), time.time() - t0))

fmin = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fmax = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
rows = np.arange(300, T - FWD - 1, STEP)
rr = np.repeat(rows, N); cc = np.tile(np.arange(N), len(rows))
X = np.stack([F[k][rr, cc] for k in NAME], axis=1)
yup = (fmax[rr, cc] / CV[rr, cc] - 1.0 >= UP).astype(np.int8)
ydn = (fmin[rr, cc] / CV[rr, cc] - 1.0 <= DN).astype(np.int8)
ok = np.isfinite(X).all(axis=1) & np.isfinite(fmax[rr, cc]) & np.isfinite(fmin[rr, cc])
X, rr, cc, yup, ydn = X[ok], rr[ok], cc[ok], yup[ok], ydn[ok]
del F; gc.collect()
say('样本 %s  涨 %.2f%%  崩 %.2f%%  %.0fs'
    % (format(len(X), ','), 100 * yup.mean(), 100 * ydn.mean(), time.time() - t0))

cut = int(T * 0.6); tr = rr < cut; te = ~tr
mu = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], yup[tr])
md = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], ydn[tr])
pu = mu.predict_proba(X[te])[:, 1]; pd_ = md.predict_proba(X[te])[:, 1]
rr_te, cc_te = rr[te], cc[te]
say('模型 AUC：涨 %.4f / 崩 %.4f  %.0fs'
    % (roc_auc_score(yup[te], pu), roc_auc_score(ydn[te], pd_), time.time() - t0))

Pup = np.full((T, N), np.nan, np.float32); Pdn = np.full((T, N), np.nan, np.float32)
Pup[rr_te, cc_te] = pu; Pdn[rr_te, cc_te] = pd_


def walk(side, thr, tp, sl, hold, trail=None):
    """side='long'/'short'；返回 (net_return, hold_bars, exit_reason_list)"""
    src = Pup if side == 'long' else Pdn
    out = []
    busy = {}
    for t in range(cut, T - hold - 1):
        js = np.where(np.isfinite(src[t]))[0]
        for j in js:
            if busy.get(j, -1) >= t:
                continue
            if not (src[t, j] >= thr):
                continue
            end = min(T, t + hold + 1)
            segH = HV[t + 1:end, j]; segL = LV[t + 1:end, j]
            if len(segH) == 0:
                continue
            b = CV[t, j]
            if side == 'long':
                tp_px, sl_px = b * (1 + tp), b * (1 + sl)
                h_tp = np.where(np.isfinite(segH) & (segH >= tp_px))[0]
                h_sl = np.where(np.isfinite(segL) & (segL <= sl_px))[0]
            else:
                tp_px, sl_px = b * (1 - tp), b * (1 + abs(sl))
                h_tp = np.where(np.isfinite(segL) & (segL <= tp_px))[0]
                h_sl = np.where(np.isfinite(segH) & (segH >= sl_px))[0]
            u_tp = int(h_tp[0]) if len(h_tp) else 10 ** 9
            u_sl = int(h_sl[0]) if len(h_sl) else 10 ** 9
            if u_sl <= u_tp and u_sl < 10 ** 9:
                u, px, reason = u_sl, sl_px, 'sl'
            elif u_tp < 10 ** 9:
                u, px, reason = u_tp, tp_px, 'tp'
            else:
                u = len(segH) - 1; px = CV[t + u + 1, j]; reason = 'timeout'
            if not np.isfinite(px) or px <= 0:
                continue
            r = float(px / b - 1.0)
            if side == 'short':
                r = -r
            out.append((r - FEE, u + 1, reason))
            busy[j] = t + u
    return out


say('\n' + '=' * 100)
say('【实验一】做空闭环：P(崩)≥阈值 → 做空（止盈 −tp% / 止损 +sl% / 超时 hold 根）')
say('=' * 100)
say('  %-7s %-8s %-7s %-7s %8s %11s %9s %8s %8s' %
    ('阈值', '止盈', '止损', '持有', '笔数', '期望净%', '中位%', '胜率', '≥+50%'))
for thr in (0.3, 0.5, 0.7):
    for tp, sl, hold in ((0.20, 0.10, 24), (0.50, 0.15, 24), (0.90, 0.20, 72)):
        o = walk('short', thr, tp, sl, hold)
        if len(o) < 20:
            say('  %-7.2f %-8s %-7s %-7d %8d  （样本不足）' % (thr, tp, sl, hold, len(o))); continue
        r = np.array([x[0] for x in o]); h = np.array([x[1] for x in o])
        say('  %-7.2f %-8.0f%% %-7.0f%% %-7d %8d %+10.3f%% %+8.2f%% %7.1f%% %7.1f%%'
            % (thr, 100 * tp, 100 * sl, hold, len(r), 100 * r.mean(), 100 * np.median(r),
               100 * (r > 0).mean(), 100 * (r >= 0.5).mean()))

say('\n' + '=' * 100)
say('【实验二】尾部赌法：P(涨)≥低阈值 → 做多（严止损 −10%%，不设止盈、只按时间/移动止损退出）')
say('=' * 100)
say('  %-7s %-9s %-8s %8s %11s %9s %8s %9s %9s' %
    ('阈值', '止损', '持有', '笔数', '期望净%', '中位%', '胜率', '≥+50%', '≥+100%'))
for thr in (0.05, 0.10, 0.20):
    for sl, hold in ((-0.10, 96), (-0.10, 288), (-0.20, 96)):
        o = walk('long', thr, 100.0, sl, hold)     # tp=100（等于不设止盈）
        if len(o) < 20:
            say('  %-7.2f %-9.0f%% %-8d %8d  （样本不足）' % (thr, 100 * sl, hold, len(o))); continue
        r = np.array([x[0] for x in o])
        say('  %-7.2f %-9.0f%% %-8d %8d %+10.3f%% %+8.2f%% %7.1f%% %8.1f%% %8.1f%%'
            % (thr, 100 * sl, hold, len(r), 100 * r.mean(), 100 * np.median(r),
               100 * (r > 0).mean(), 100 * (r >= 0.5).mean(), 100 * (r >= 1.0).mean()))

# 对照：随机做空 / 随机做多
rng = np.random.default_rng(0)
for side in ('short', 'long'):
    rets = []
    for _ in range(20000):
        j = int(rng.integers(0, N)); t = int(rng.integers(cut, T - 25))
        r = float(CV[t + 24, j] / CV[t, j] - 1.0)
        if not np.isfinite(r):
            continue
        rets.append(-r - FEE if side == 'short' else r - FEE)
    rets = np.array(rets)
    say('  对照 随机%s 持 24 根：期望净 %+.3f%%  胜率 %.1f%%'
        % ('做空' if side == 'short' else '做多', 100 * rets.mean(), 100 * (rets > 0).mean()))

open(os.path.join(EV, 'gambler_short_tail.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
