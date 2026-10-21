# -*- coding: utf-8 -*-
"""gambler 闭环 v2（**无前视版**）—— 修正 gambler_loop.py 的致命缺陷

为什么要重做（诚实交代）：
  gambler_loop.py 的特征里含 `bars_from_start` / `gain_from_start` / `high_gain`，
  这三个都依赖"事件起点 i"，而 i 是用 `未来 4h 最高价 ≥ 当前价×1.5` 事后定义的
  ⇒ **特征里混进了未来信息（前视泄漏）**。所以那版 +22%/笔 是虚高的，不可信。
  （注：标签置换/币留出检验**发现不了**这类特征泄漏，必须靠人审特征构造。）

本版规则（全部 t 时刻可见，无任何事件起点）：
  特征：30 个**纯滚动**多尺度特征（收益 3~288 根 / 量比 12~192 / 波动 / 距高低点 / D1-D8 / 市场）
  样本：全时段均匀抽样（每 3 根取 1 点），**不依赖任何事后定义**
  模型：P(涨·未来2h≥+20%)、P(崩·未来2h≤−20%)
  闭环：P(涨) ≥ 分位阈值 → 按收盘价买入；P(崩) ≥ 0.5 或持满 24 根 → 卖出
  成本：maker 双边 0.04%（另报 taker 0.20%）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD, MAXHOLD = 3, 24, 24
DN, UP = -0.20, 0.20
FEE_M, FEE_T = 0.0004, 0.0020
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 闭环 v2（无前视特征版）')
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
tm = close.index.values
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
F = {}
for w in (3, 6, 12, 24, 48, 96, 192, 288):
    a = np.full((T, N), np.nan, np.float32); a[w:] = CV[w:] / CV[:-w] - 1.0
    F['ret%d' % w] = a
for w in (12, 48, 192):
    med = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
    med = np.where(med > 0, med, np.nan)
    F['volr%d' % w] = (QV / med).astype('float32')
for w in (12, 48, 192):
    F['vol%d' % w] = pd.DataFrame(ret1).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
for w in (24, 96, 288):
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
d7 = np.zeros((T, N), np.float32)
for j in range(N):
    fz = np.r_[False, CV[1:, j] == CV[:-1, j]]
    lf = np.maximum.accumulate(np.where(~fz, idxr, -1))
    d7[:, j] = np.where(fz, idxr - lf, 0)
F['D7'] = d7
mkt = np.nanmedian(ret1, axis=1); disp = np.nanstd(ret1, axis=1)
F['mkt12'] = np.repeat(pd.Series(mkt).rolling(12, min_periods=4).sum().values[:, None], N, axis=1).astype('float32')
F['mkt48'] = np.repeat(pd.Series(mkt).rolling(48, min_periods=8).sum().values[:, None], N, axis=1).astype('float32')
F['disp48'] = np.repeat(pd.Series(disp).rolling(48, min_periods=8).mean().values[:, None], N, axis=1).astype('float32')
F['D8'] = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
NAME = list(F.keys())
say('特征 %d 个（全部滚动可见，无事件依赖）  %.0fs' % (len(NAME), time.time() - t0))

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

cut = int(T * 0.6)
tr = rr < cut; te = ~tr
mu = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], yup[tr])
md = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], ydn[tr])
pu = mu.predict_proba(X[te])[:, 1]; pd_ = md.predict_proba(X[te])[:, 1]
say('模型：AUC 涨 %.4f / 崩 %.4f   测试样本 %s'
    % (roc_auc_score(yup[te], pu), roc_auc_score(ydn[te], pd_), format(int(te.sum()), ',')))

# 概率 -> (T,N) 图（仅测试段）
Pup = np.full((T, N), np.nan, np.float32); Pdn = np.full((T, N), np.nan, np.float32)
Pup[rr[te], cc[te]] = pu; Pdn[rr[te], cc[te]] = pd_

# 入场阈值：取 P(涨) 的 99.9 分位（全市场极少数）
ENT = float(np.percentile(pu, 99.9))
say('\n入场阈值 = P(涨) 的 99.9 分位 = %.3f（测试段报出 %d 个时点）' % (ENT, int((pu >= ENT).sum())))

trades = []
busy = np.full(N, -1)
for t in range(cut, T - MAXHOLD - 1):
    js = np.where(np.isfinite(Pup[t]))[0]
    for j in js:
        if t <= busy[j]:
            continue
        if not (Pup[t, j] >= ENT and np.isfinite(Pdn[t, j]) and Pdn[t, j] < 0.5):
            continue
        exit_t = t + MAXHOLD; reason = 'timeout'
        for u in range(t + STEP, t + MAXHOLD + 1, STEP):
            if np.isfinite(Pdn[u, j]) and Pdn[u, j] >= 0.5:
                exit_t = u; reason = 'signal'; break
        r = float(CV[exit_t, j] / CV[t, j] - 1.0)
        trades.append((t, j, exit_t, exit_t - t, r, reason))
        busy[j] = exit_t
say('闭环交易 %d 笔  %.0fs' % (len(trades), time.time() - t0))

say('\n' + '=' * 100)
say('【无前视闭环结果】（测试段 = 后 40%% 时间，304 币）')
say('=' * 100)
if trades:
    a = np.array([(x[3], x[4]) for x in trades])
    hold, ret = a[:, 0], a[:, 1]
    days = (T - cut) * 5 / 1440.0
    say('  笔数 %d；平均持有 %.1f 根（%.0f 分钟）；**每天 %.2f 笔**'
        % (len(trades), hold.mean(), hold.mean() * 5, len(trades) / days))
    say('  胜率 %.1f%%；毛收益 均值 %+.3f%%  中位 %+.3f%%  P10 %+.2f%%  P90 %+.2f%%'
        % (100 * (ret > 0).mean(), 100 * ret.mean(), 100 * np.median(ret),
           100 * np.percentile(ret, 10), 100 * np.percentile(ret, 90)))
    say('  净 maker(0.04%%) 均值 %+.3f%%/笔  合计 %+.1f%%' % (100 * (ret.mean() - FEE_M), 100 * (ret - FEE_M).sum()))
    say('  净 taker(0.20%%) 均值 %+.3f%%/笔  合计 %+.1f%%' % (100 * (ret.mean() - FEE_T), 100 * (ret - FEE_T).sum()))
    say('  退出：信号 %d 笔 / 超时 %d 笔'
        % (sum(1 for x in trades if x[5] == 'signal'), sum(1 for x in trades if x[5] == 'timeout')))
    rng = np.random.default_rng(0)
    base = []
    for _ in range(50000):
        j = int(rng.integers(0, N)); t2 = int(rng.integers(cut, T - MAXHOLD - 1))
        v = CV[t2 + MAXHOLD, j] / CV[t2, j] - 1.0
        if np.isfinite(v):
            base.append(float(v))
    base = np.array(base)
    say('  对照：全市场随机时刻持有 24 根（n=%d）→ 均值 %+.3f%%  胜率 %.1f%%'
        % (len(base), 100 * base.mean(), 100 * (base > 0).mean()))
    pd.DataFrame(trades, columns=['entry_row', 'col', 'exit_row', 'hold', 'ret', 'reason']).to_csv(
        os.path.join(EV, 'gambler_trades_clean.csv'), index=False, encoding='utf-8')
else:
    say('  无交易（阈值下无信号）')

open(os.path.join(EV, 'gambler_loop_clean.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
