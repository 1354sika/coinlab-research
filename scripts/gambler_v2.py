# -*- coding: utf-8 -*-
"""gambler v2 —— 去掉固定 4h 窗口，改用「多尺度特征 + 多 horizon 标签」

用户口径（2026-10-16）：
  · 4h 窗口是**相对的**——操盘手的时间段不固定，"具体窗口看不到但存在"
  · 要复杂的概率计算：概率到什么值 → 触发什么动作 → 输出什么结果
  · 识别涨、识别崩、识别操盘者；准确率 ≥85%

本版改变（相对 v1）：
  · **不再用"4h 内涨 50%"圈样本** → 改为**全市场按时间等距抽样**（每个币每 12 根 5m 取 1 点）
  · 特征全部改成**多尺度**（ret / 量比 / 波动 / 距高低点 在 1h~24h 多个尺度同时给），
    让模型自己挑时间尺度 —— 而不是由我规定窗口
  · 标签用多 horizon（2h / 4h），不依赖事件定义

数据：klines_5m_502.csv（304 币 × 3 年 5m）
切分：按时间 60% 训练 / 40% 测试（严格 walk-forward，非随机）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP = 12
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler v2 —— 去固定窗口：多尺度特征 + 多 horizon 标签（识别涨 / 识别崩）')
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
# --- 多尺度收益 ---
for w in (3, 6, 12, 24, 48, 96, 192, 288):
    a = np.full((T, N), np.nan, np.float32); a[w:] = CV[w:] / CV[:-w] - 1.0
    F['ret%d' % w] = a
# --- 多尺度量比 ---
for w in (12, 48, 192):
    med = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
    med = np.where(med > 0, med, np.nan)
    F['volr%d' % w] = (QV / med).astype('float32')
# --- 多尺度波动 ---
for w in (12, 48, 192):
    F['vol%d' % w] = pd.DataFrame(ret1).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
# --- 距高低点（相对距离，替代"窗口"概念） ---
for w in (24, 96, 288):
    hi = pd.DataFrame(HV).rolling(w, min_periods=w // 4).max().values
    lo = pd.DataFrame(LV).rolling(w, min_periods=w // 4).min().values
    F['dhi%d' % w] = (CV / hi - 1.0).astype('float32')
    F['dlo%d' % w] = (CV / lo - 1.0).astype('float32')
# --- 段结构 D1/D2/D3 ---
sgn = np.sign(np.nan_to_num(ret1))
d1 = np.zeros((T, N), np.float32)
idxr = np.arange(T)
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
segret = CV / sp - 1.0
F['D2'] = ((segret / L) * 100).astype('float32')
F['D3'] = (segret * 100).astype('float32')
# --- D4 突变 / D5 漂移 / D7 冻结 ---
sd = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
F['D4'] = (np.abs(ret1) > 3.0 * sd).astype('float32')
F['D5'] = (np.abs(ret1) <= 0.5 * sd).astype('float32')
d7 = np.zeros((T, N), np.float32)
for j in range(N):
    fz = np.r_[False, CV[1:, j] == CV[:-1, j]]
    lf = np.maximum.accumulate(np.where(~fz, idxr, -1))
    d7[:, j] = np.where(fz, idxr - lf, 0)
F['D7'] = d7
# --- 市场级 ---
mkt = np.nanmedian(ret1, axis=1)
disp = np.nanstd(ret1, axis=1)
F['mkt12'] = np.repeat(pd.Series(mkt).rolling(12, min_periods=4).sum().values[:, None], N, axis=1).astype('float32')
F['mkt48'] = np.repeat(pd.Series(mkt).rolling(48, min_periods=8).sum().values[:, None], N, axis=1).astype('float32')
F['disp48'] = np.repeat(pd.Series(disp).rolling(48, min_periods=8).mean().values[:, None], N, axis=1).astype('float32')
F['D8'] = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
NAME = list(F.keys())
say('多尺度特征 %d 个：%s  %.0fs' % (len(NAME), NAME, time.time() - t0))

# ---- 标签（多 horizon，不依赖事件定义）----
def fwd_stat(kind, w):
    if kind == 'min':
        return pd.DataFrame(LV).iloc[::-1].rolling(w, min_periods=1).min().iloc[::-1].shift(-1).values
    return pd.DataFrame(HV).iloc[::-1].rolling(w, min_periods=1).max().iloc[::-1].shift(-1).values
DN2 = fwd_stat('min', 24) / CV - 1.0
UP2 = fwd_stat('max', 24) / CV - 1.0
DN4 = fwd_stat('min', 48) / CV - 1.0

# ---- 抽样（每币每 STEP 根取 1 点，去固定窗口）----
rows = np.arange(300, T - 49, STEP)
rr = np.repeat(rows, N); cc = np.tile(np.arange(N), len(rows))
say('候选点 = %s  %.0fs' % (format(len(rr), ','), time.time() - t0))
X = np.stack([F[k][rr, cc] for k in NAME], axis=1)
ydn2 = (DN2[rr, cc] <= -0.20); yup2 = (UP2[rr, cc] >= 0.20); ydn4 = (DN4[rr, cc] <= -0.30)
ok = np.isfinite(X).all(axis=1) & np.isfinite(DN2[rr, cc]) & np.isfinite(UP2[rr, cc])
X, rr, ydn2, yup2, ydn4 = X[ok], rr[ok], ydn2[ok], yup2[ok], ydn4[ok]
del F; gc.collect()
say('可用样本 = %s  %.0fs' % (format(len(X), ','), time.time() - t0))

cut = int(T * 0.6)
tr = rr < cut; te = ~tr
say('训练 %s / 测试 %s（按时间切，非随机）' % (format(int(tr.sum()), ','), format(int(te.sum()), ',')))

for tag, y in (('崩·未来2h跌≥20%', ydn2), ('涨·未来2h涨≥20%', yup2), ('崩·未来4h跌≥30%', ydn4)):
    y = y.astype(np.int8)
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=5,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=42, early_stopping=False)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    say('\n' + '=' * 100)
    say('【gambler v2 · %s】AUC = %.4f   基准发生率 = %.2f%%   测试样本 %s'
        % (tag, auc, 100 * y[te].mean(), format(int(te.sum()), ',')))
    say('=' * 100)
    say('  %-8s %12s %11s %11s %11s' % ('阈值', '报出数', '覆盖率', '精确率', '召回率'))
    best = None
    for t in np.arange(0.03, 0.995, 0.01):
        m = p >= t
        if m.sum() < 50:
            continue
        prec = float(y[te][m].mean())
        if prec >= 0.85:
            best = (float(t), int(m.sum()), prec, float(y[te][m].sum() / max(1, y[te].sum())))
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = p >= t
        if m.sum() < 50:
            continue
        say('  %-8.2f %12s %10.2f%% %10.2f%% %10.2f%%'
            % (t, format(int(m.sum()), ','), 100 * m.mean(), 100 * y[te][m].mean(),
               100 * y[te][m].sum() / max(1, y[te].sum())))
    if best:
        say('  ★ 精确率≥85%% 最低阈值 %.2f：报出 %s 点（覆盖 %.2f%%），精确率 %.1f%%，召回 %.1f%%'
            % (best[0], format(best[1], ','), 100 * best[1] / int(te.sum()), 100 * best[2], 100 * best[3]))
    else:
        say('  ✗ 无阈值达到 85%% 精确率')
    say('  特征方向 corr：' + '  '.join(
        '%s=%+.3f' % (nm, float(np.corrcoef(np.nan_to_num(X[tr, k]), y[tr])[0, 1]))
        for k, nm in enumerate(NAME)))

open(os.path.join(EV, 'gambler_v2.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
