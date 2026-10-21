# -*- coding: utf-8 -*-
"""塌盘概率模型 v1：P(未来 1h 内回撤 ≥30% | 操盘手事件中的当前状态)

用户口径（2026-10-16）：识别操盘手 → 概率行为预测 → 高期望搞到就走、不逗留。
本模型直接回答那个"走不走"的问题：**给定当前状态，接下来 1 小时崩掉的概率是多少。**

数据：klines_5m_502.csv（304 币 × 3 年 5m）
事件：4h 窗口内未来最高价 ≥ 当前价 ×1.5（+50% 档，样本最多）
样本：事件窗口起点 → 峰值后 2h 之间的每个 5m 点
特征（全部 t 时刻可见，无前视）：
  bars_from_start  距事件起点根数
  gain_from_start  相对事件起点已涨幅
  high_from_start  相对事件起点最高价涨幅
  volr / D1 / D4 / ret60 / ret240 / vol240
标签：未来 12 根（1h）内 low 相对当前 close 的最大跌幅 ≤ −30%
切分：按事件时间 walk-forward（前 60% 事件训练，后 40% 测试）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
W5, THR, FWD = 48, 0.5, 12
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('塌盘概率模型 v1：P(未来 1h 内回撤 ≥30%% | 操盘手事件中的当前状态)')
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
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))
CV, HV, LV = close.values, high.values, low.values

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
qv60 = pd.DataFrame(qv).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)
volr = qv.values / qv60
ret60 = np.full((T, N), np.nan, np.float32); ret60[60:] = CV[60:] / CV[:-60] - 1.0
ret240 = np.full((T, N), np.nan, np.float32); ret240[240:] = CV[240:] / CV[:-240] - 1.0
vol240 = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
sgn = np.sign(np.nan_to_num(ret1)); d1 = np.zeros((T, N), np.float32)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (np.arange(T) - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
d4 = (np.abs(ret1) > 3.0 * vol240).astype(np.float32)
say('特征算完  %.0fs' % (time.time() - t0))

# 未来 12 根内最低 low
fwd_min = low.iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1]
fwd_min = fwd_min.shift(-1)          # 严格从下一根起（不含当前）
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gain = (fwd_high / close - 1.0).values
FMIN = fwd_min.values

# ---------------- 事件 + 样本 ----------------
P = gain >= THR
rows_f, cols_f = [], []
ev_meta = []
for j in range(N):
    col = P[:, j]
    if not col.any():
        continue
    starts = np.where(col & ~np.r_[False, col[:-1]])[0]
    last = -10 ** 9
    for i in starts:
        if i - last < W5:
            continue
        seg = HV[i:i + W5, j]
        pk = i + int(np.nanargmax(seg))
        last = int(i)
        ev_meta.append((j, int(i), int(pk)))
say('事件数 = %d  %.0fs' % (len(ev_meta), time.time() - t0))

for (j, i, pk) in ev_meta:
    end = min(T, pk + 24)              # 峰值后 2h
    for t in range(i + 1, end):
        if t - 1 < 0 or t + FWD > T:
            continue
        c = CV[t, j]
        if not np.isfinite(c) or c <= 0:
            continue
        fm = FMIN[t, j]
        if not np.isfinite(fm):
            continue
        rows_f.append(t); cols_f.append(j)
rows_f = np.array(rows_f); cols_f = np.array(cols_f)
say('样本点 = %s  %.0fs' % (format(len(rows_f), ','), time.time() - t0))

# 组装特征/标签
ev_id = -np.ones(len(rows_f), np.int64)
for k, (j, i, pk) in enumerate(ev_meta):
    m = (cols_f == j) & (rows_f >= i) & (rows_f < min(T, pk + 24))
    ev_id[m] = k
c0 = CV[rows_f, cols_f]
i_arr = np.array([ev_meta[k][1] for k in ev_id])
X = np.column_stack([
    (rows_f - i_arr).astype('float32'),
    c0 / CV[i_arr, cols_f] - 1.0,
    HV[rows_f, cols_f] / CV[i_arr, cols_f] - 1.0,
    volr[rows_f, cols_f], d1[rows_f, cols_f], d4[rows_f, cols_f],
    ret60[rows_f, cols_f], ret240[rows_f, cols_f], vol240[rows_f, cols_f],
]).astype('float32')
drop = FMIN[rows_f, cols_f] / c0 - 1.0
Y = (drop <= -0.30).astype(np.int8)
NAME = ['bars', 'gain', 'high_gain', 'volr', 'D1', 'D4', 'ret60', 'ret240', 'vol240']
say('特征 %d 个：%s' % (len(NAME), NAME))
say('样本 %s  正例(未来1h跌≥30%%) = %.2f%%' % (format(len(X), ','), 100 * Y.mean()))

# walk-forward：按事件时间切
ev_t = np.array([ev_meta[k][1] for k in range(len(ev_meta))])
order = np.argsort(ev_t)
split = int(len(ev_meta) * 0.6)
tr_ev = set(order[:split].tolist()); te_ev = set(order[split:].tolist())
tr = np.array([k in tr_ev for k in ev_id]); te = ~tr
say('训练样本 %s（%d 事件）  测试样本 %s（%d 事件）'
    % (format(int(tr.sum()), ','), len(tr_ev), format(int(te.sum()), ','), len(te_ev)))

clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=4,
                                     min_samples_leaf=100, l2_regularization=1.0,
                                     random_state=42, early_stopping=False)
clf.fit(X[tr], Y[tr])
p = clf.predict_proba(X[te])[:, 1]
say('AUC = %.4f' % roc_auc_score(Y[te], p))

say('\n' + '=' * 100)
say('【概率分档 -> 真实塌盘率】（测试段，按点统计）')
say('=' * 100)
say('  %-14s %10s %12s %12s' % ('P(塌盘)', '样本', '真实塌盘率', '毛收益代理'))
bins = [0, .05, .10, .20, .30, .40, .50, .70, 1.01]
for a, b in zip(bins[:-1], bins[1:]):
    m = (p >= a) & (p < b)
    if m.sum() < 30:
        continue
    say('  [%.2f,%.2f)   %10s %11.1f%% %+11.2f%%'
        % (a, b, format(int(m.sum()), ','), 100 * Y[te][m].mean(), 100 * drop[te][m].mean()))

say('\n【事件级：每个事件取测试段最大 P(塌盘) 的点】')
ev_rows = []
for k in te_ev:
    m = (ev_id[te] == k)
    if not m.any():
        continue
    pp = p[m]; yy = Y[te][m]
    ev_rows.append((float(pp.max()), int(yy[np.argmax(pp)])))
if ev_rows:
    amax = np.array([r[0] for r in ev_rows]); ayy = np.array([r[1] for r in ev_rows])
    say('  测试事件 %d 个；按最大 P 的档位：' % len(ev_rows))
    for a, b in zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1.01]):
        m = (amax >= a) & (amax < b)
        if m.sum() == 0:
            continue
        say('    P∈[%.1f,%.1f)  事件 %3d  其中塌盘 %3d（%.0f%%）'
            % (a, b, int(m.sum()), int(ayy[m].sum()), 100 * ayy[m].mean()))
    hit = amax >= 0.4
    say('  ★ P≥0.40 报警：覆盖 %d/%d 事件（%.0f%%），其中 %.0f%% 真塌盘'
        % (int(hit.sum()), len(amax), 100 * hit.mean(), 100 * ayy[hit].mean() if hit.any() else 0))
    say('  全体测试事件真塌盘率 = %.0f%%（基准）' % (100 * ayy.mean()))

# 特征重要性（置换近似：用相关系数符号给方向）
say('\n【特征与"塌盘"的相关方向（训练段）】')
for k, nm in enumerate(NAME):
    v = X[tr, k]
    if np.nanstd(v) == 0:
        continue
    say('  %-10s corr=%+.3f' % (nm, float(np.corrcoef(np.nan_to_num(v), Y[tr])[0, 1])))

open(os.path.join(EV, 'collapse_proba.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
