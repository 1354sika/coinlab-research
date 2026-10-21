# -*- coding: utf-8 -*-
"""gambler 双盲 / 稳健性检验 —— 这个模型到底准不准？

用户问题（2026-10-16）："这个模型到底准不准？有没有做双盲实验？"

四项检验（口径与 gambler v1 完全一致：未来 2h 跌≥20% / 涨≥20%；事件窗口样本 + 11 特征）
  A 币留出 cross-coin  ：一半币训练、另一半**从没见过的币**测试（币级零重叠）
  B 多折时间 walk-forward：按时间切 5 段，训练=历史、测试=其后一段（滚动 4 次）
  C 标签置换 shuffle    ：打乱标签重训，AUC 应掉回 ≈0.5（检验有无泄漏/过拟合）
  D 逐年稳定性          ：2024 / 2025 / 2026 各自训练+测试，看是否同年份都成立
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
W5, THR, PRE, POST, FWD = 48, 0.5, 48, 24, 24
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

def fit_eval(X, y, tr, te, tag):
    y = y.astype(np.int8)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        say('  %-28s  (单一类别，跳过)' % tag); return None
    clf = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_depth=4,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         random_state=42, early_stopping=False)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    best = None
    for t in np.arange(0.03, 0.995, 0.01):
        m = p >= t
        if m.sum() < 50:
            continue
        prec = float(y[te][m].mean())
        if prec >= 0.85:
            best = (float(t), int(m.sum()), prec, float(y[te][m].sum() / max(1, y[te].sum())))
    if best:
        say('  %-28s AUC=%.4f  基准=%.2f%%  ★85%%精确率@P≥%.2f 覆盖%.2f%% 精确率%.1f%% 召回%.1f%%'
            % (tag, auc, 100 * y[te].mean(), best[0], 100 * best[1] / int(te.sum()),
               100 * best[2], 100 * best[3]))
    else:
        say('  %-28s AUC=%.4f  基准=%.2f%%  ✗无阈值达85%%' % (tag, auc, 100 * y[te].mean()))
    return auc

t0 = time.time()
say('=' * 100)
say('gambler 双盲 / 稳健性检验')
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
syms = list(close.columns)
CV, HV, LV = close.values, high.values, low.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
qv60 = pd.DataFrame(qv).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)
volr = qv.values / qv60
ret60 = np.full((T, N), np.nan, np.float32); ret60[60:] = CV[60:] / CV[:-60] - 1.0
ret240 = np.full((T, N), np.nan, np.float32); ret240[240:] = CV[240:] / CV[:-240] - 1.0
vol240 = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
d4 = (np.abs(ret1) > 3.0 * vol240).astype(np.float32)
sgn = np.sign(np.nan_to_num(ret1))
d1 = np.zeros((T, N), np.float32); d7 = np.zeros((T, N), np.float32)
idxr = np.arange(T)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (idxr - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
    fz = np.r_[False, CV[1:, j] == CV[:-1, j]]
    lf = np.maximum.accumulate(np.where(~fz, idxr, -1))
    d7[:, j] = np.where(fz, idxr - lf, 0)
d8 = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')

fwd_min = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fwd_max = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gainW = (fwd_high / close - 1.0).values

P = gainW >= THR
ev = []
for j in range(N):
    col = P[:, j]
    if not col.any():
        continue
    starts = np.where(col & ~np.r_[False, col[:-1]])[0]
    last = -10 ** 9
    for i in starts:
        if i - last < W5:
            continue
        pk = i + int(np.nanargmax(HV[i:i + W5, j]))
        ev.append((j, int(i), int(pk))); last = int(i)
say('事件（局）= %d  %.0fs' % (len(ev), time.time() - t0))

rf, cf, evi = [], [], []
for k, (j, i, pk) in enumerate(ev):
    for t in range(max(0, i - PRE), min(T - 1, pk + POST)):
        if not (np.isfinite(CV[t, j]) and np.isfinite(fwd_min[t, j]) and np.isfinite(fwd_max[t, j])):
            continue
        rf.append(t); cf.append(j); evi.append(k)
rf = np.array(rf); cf = np.array(cf); evi = np.array(evi)
I = np.array([ev[k][1] for k in evi])
c0 = CV[rf, cf]
X = np.column_stack([
    (rf - I).astype('float32'), c0 / CV[I, cf] - 1.0, HV[rf, cf] / CV[I, cf] - 1.0,
    volr[rf, cf], d1[rf, cf], d4[rf, cf], d7[rf, cf],
    ret60[rf, cf], ret240[rf, cf], vol240[rf, cf], d8[rf, cf],
]).astype('float32')
ydn = (fwd_min[rf, cf] / c0 - 1.0 <= DN).astype(np.int8)
yup = (fwd_max[rf, cf] / c0 - 1.0 >= UP).astype(np.int8)
YEAR = pd.to_datetime(tm[rf], unit='ms', utc=True).year.values
say('样本 %s  崩 %.2f%%  涨 %.2f%%  %.0fs'
    % (format(len(X), ','), 100 * ydn.mean(), 100 * yup.mean(), time.time() - t0))

rng = np.random.default_rng(0)
for tag, y in (('崩(2h −20%)', ydn), ('涨(2h +20%)', yup)):
    say('\n' + '=' * 100)
    say('【%s】双盲检验' % tag)
    say('=' * 100)
    # A 币留出：一半币训练 / 另一半币测试
    coin_ids = np.unique(cf)
    rng.shuffle(coin_ids)
    half = set(coin_ids[:len(coin_ids) // 2].tolist())
    inc = np.array([c in half for c in cf])
    say('  A 币留出（%d 币训练 / %d 币测试，币不重叠）'
        % (len(half), len(coin_ids) - len(half)))
    fit_eval(X, y, inc, ~inc, '   训练币→测试新币')
    # C 标签置换（同切分，打乱标签）
    ysh = y.copy(); rng.shuffle(ysh)
    fit_eval(X, ysh, inc, ~inc, '   C 标签置换(应≈0.5)')
    # B 多折时间滚动
    say('  B 多折时间 walk-forward（5 段，训练=历史）')
    q = np.linspace(0, len(X), 6).astype(int)
    order = np.argsort(rf)
    Xs, ys, rs = X[order], y[order], rf[order]
    q = np.linspace(0, len(Xs), 6).astype(int)
    for k in range(1, 5):
        tr = np.zeros(len(Xs), bool); tr[:q[k]] = True
        te = np.zeros(len(Xs), bool); te[q[k]:q[k + 1]] = True
        yr = pd.to_datetime(tm[rs[q[k]]], unit='ms', utc=True).date()
        fit_eval(Xs, ys, tr, te, '   折%d  至 %s' % (k, yr))
    # D 逐年
    say('  D 逐年稳定性（训练=其余年份，测试=该年）')
    for yr in np.unique(YEAR):
        te = YEAR == yr
        tr = ~te
        if te.sum() < 500:
            continue
        fit_eval(X, y, tr, te, '   %d 年（%s 点）' % (yr, format(int(te.sum()), ',')))

open(os.path.join(EV, 'gambler_blind.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
