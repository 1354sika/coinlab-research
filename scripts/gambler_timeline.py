# -*- coding: utf-8 -*-
"""gambler · 时间轴实验 —— 回答用户："是不是能知道何时收尾、但不知道何时下场？"

做法：对每个"局"（4h 内涨≥50%，仅用于**事后定位**，不进特征），
      把事件内的时点按"相对起点的根数 τ"分箱：
          τ<0   = 启动之前（"下场前"）
          τ≈0   = 启动点
          τ 增大 = 拉升中
          τ>峰值 = 崩中
      在**每个 τ 箱内**单独算 AUC（模型本身用全样本训练，不看 τ）：
          崩：未来 2h 跌 ≥20%
          涨：未来 2h 涨 ≥20%
      —— 这样就能看到"我们的判断力沿时间轴怎么变"。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD, W5, THR = 6, 24, 48, 0.5
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 时间轴实验：判断力沿"事件时间轴"的变化')
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
for w in (12, 48):
    med = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
    F['volr%d' % w] = (QV / np.where(med > 0, med, np.nan)).astype('float32')
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
sd = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
F['D4'] = (np.abs(ret1) > 3.0 * sd).astype('float32')
F['rk_ret48'] = pd.DataFrame(F['ret48']).rank(axis=1, pct=True).values.astype('float32')
mkt = np.nanmedian(ret1, axis=1)
F['mkt48'] = np.repeat(pd.Series(mkt).rolling(48, min_periods=8).sum().values[:, None], N, axis=1).astype('float32')
F['D8'] = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
NAME = list(F.keys())
say('特征 %d 个  %.0fs' % (len(NAME), time.time() - t0))

fmin = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fmax = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gainW = (fwd_high / close - 1.0).values

# 事件（仅用于事后定位 τ）
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
say('局 = %d 个  %.0fs' % (len(ev), time.time() - t0))

# 全样本模型（不看 τ）
rows = np.arange(300, T - FWD - 1, STEP)
rr = np.repeat(rows, N); cc = np.tile(np.arange(N), len(rows))
X = np.stack([F[k][rr, cc] for k in NAME], axis=1)
ydn = (fmin[rr, cc] / CV[rr, cc] - 1.0 <= DN).astype(np.int8)
yup = (fmax[rr, cc] / CV[rr, cc] - 1.0 >= UP).astype(np.int8)
ok = np.isfinite(X).all(axis=1)
X, rr, cc, ydn, yup = X[ok], rr[ok], cc[ok], ydn[ok], yup[ok]
cut = int(T * 0.6); tr = rr < cut
md = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, random_state=42, early_stopping=False).fit(X[tr], ydn[tr])
mu = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                    min_samples_leaf=200, random_state=42, early_stopping=False).fit(X[tr], yup[tr])
pdn_all = md.predict_proba(X)[:, 1]; pup_all = mu.predict_proba(X)[:, 1]
say('模型训练完（tau 不进特征）  %.0fs' % (time.time() - t0))

# 事件内时点 → τ = t - i
ev_of = -np.ones(len(rr), np.int64)
key = {}
for k, (j, i, pk) in enumerate(ev):
    key[(j, i)] = k
tau = np.full(len(rr), np.nan, np.float32)
is_ev = np.zeros(len(rr), bool)
for k, (j, i, pk) in enumerate(ev):
    m = (cc == j) & (rr >= i - 48) & (rr <= pk + 24)
    if m.any():
        tau[m] = rr[m] - i
        is_ev |= m
say('事件内时点 = %s  %.0fs' % (format(int(is_ev.sum()), ','), time.time() - t0))

BINS = [(-48, -24), (-24, -12), (-12, -6), (-6, 0), (0, 6), (6, 12), (12, 24), (24, 48), (48, 10000)]
say('\n' + '=' * 100)
say('【判断力沿时间轴】τ = 距"启动点"的根数（5m/根）；τ<0 = 启动之前')
say('=' * 100)
say('  %-14s %9s %10s %10s %10s %10s' % ('时间段', '样本', '崩发生率', 'AUC(崩)', '涨发生率', 'AUC(涨)'))
res = []
for a_, b_ in BINS:
    m = is_ev & (tau >= a_) & (tau < b_)
    if m.sum() < 200:
        continue
    lab = 'τ∈[%d,%d)' % (a_, b_)
    ad = roc_auc_score(ydn[m], pdn_all[m]) if len(np.unique(ydn[m])) > 1 else np.nan
    au = roc_auc_score(yup[m], pup_all[m]) if len(np.unique(yup[m])) > 1 else np.nan
    res.append((lab, int(m.sum()), ydn[m].mean(), ad, yup[m].mean(), au))
    say('  %-14s %9s %9.1f%% %10.4f %9.1f%% %10.4f'
        % (lab, format(int(m.sum()), ','), 100 * ydn[m].mean(), ad, 100 * yup[m].mean(), au))
say('\n  基准（测试段全样本）AUC：崩 %.4f / 涨 %.4f'
    % (roc_auc_score(ydn[~tr], pdn_all[~tr]), roc_auc_score(yup[~tr], pup_all[~tr])))

if res:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = range(len(res))
    ax.plot(xs, [r[3] for r in res], 'o-', label='AUC(predict collapse)')
    ax.plot(xs, [r[5] for r in res], 's-', label='AUC(predict rise)')
    ax.axhline(0.5, ls='--', color='k', lw=1, label='0.5 = no skill')
    ax.axvline(3.5, ls=':', color='k', lw=1)
    ax.text(3.55, 0.52, 'launch point (tau=0)', fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in res], rotation=25, fontsize=8)
    ax.set_ylabel('AUC'); ax.set_ylim(0.4, 1.0)
    ax.set_title('gambler: predictive power along the event timeline (304 coins / 3y)')
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    p = os.path.join(EV, 'gambler_timeline.png')
    plt.savefig(p, dpi=125, bbox_inches='tight')
    say('saved: %s' % p)

open(os.path.join(EV, 'gambler_timeline.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
