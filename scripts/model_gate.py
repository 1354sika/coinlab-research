# -*- coding: utf-8 -*-
"""模型做第二道确认：手工规则信号 + 概率模型过滤，能不能同时提高胜率与收益？

背景（前面实测）：
  · 手工规则（急跌+巨量+多币+过滤器）：胜率 71.19%、722 笔、净 maker +1.46%
  · 概率模型单独（p>0.6）：胜率 56.34%、128549 笔、净 maker +0.307%
  · 模型本身只有弱信号（AUC 0.5338），但覆盖广；手工规则精度高但样本少

本脚本检验：把两者叠加（只在手工信号发生时，再用模型概率筛）
  · 对比：手工单独 vs 手工+模型(p>0.5/0.55/0.6) vs 模型单独
  · 关键判据：交集胜率是否显著高于 71.19%、样本还剩多少、净收益是否更高
  · 严格 walk-forward：模型只用过去数据训练
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_5m_3y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, DEP, VOLR, CONCUR = 12, 0.05, 3.0, 2
FEE_M, FEE_T = 0.0002, 0.0010
D1, D7 = 288, 288 * 7

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float64', 'quote_volume': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P, QV = px.values, qv.values
T, N = P.shape
tms = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60
mkt1 = np.nanmean(ret1, axis=1)
disp = np.nanstd(ret1, axis=1)
F = {}
F['ret1'] = ret1
F['ret12'] = np.full((T, N), np.nan); F['ret12'][12:] = P[12:] / P[:-12] - 1.0
F['ret288'] = np.full((T, N), np.nan); F['ret288'][288:] = P[288:] / P[:-288] - 1.0
F['volr'] = volr
F['vol1h'] = pd.DataFrame(ret1).rolling(12).std().values
ma288 = pd.DataFrame(P).rolling(D1).mean().values
F['dist_ma'] = P / ma288 - 1.0
s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
F['seglen'] = seglen
F['segret'] = P / np.take_along_axis(P, start, axis=0) - 1.0
F['mkt1'] = mkt1[:, None] * np.ones((1, N))
F['mkt12'] = pd.Series(mkt1).rolling(12).sum().values[:, None] * np.ones((1, N))
F['disp'] = disp[:, None] * np.ones((1, N))
F['mktvol'] = pd.Series(mkt1).rolling(12).std().values[:, None] * np.ones((1, N))
F['rk_ret'] = pd.DataFrame(ret1).rank(axis=1, pct=True).values
F['rk_volr'] = pd.DataFrame(volr).rank(axis=1, pct=True).values
FEATS = list(F.keys())

FWD = np.full((T, N), np.nan); FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
Y = (FWD > 0).astype(np.int8)

# 手工规则信号
CV = volr > VOLR; CD = ret1 < 0
base = (streak := F['segret']) < -DEP
sig = base & CV
conc = sig.sum(axis=1)[:, None] * np.ones((1, N))
SIGM = sig & (conc >= CONCUR)
coin7 = P / pd.DataFrame(P).shift(D7).values - 1.0
SIGM = SIGM & (coin7 > -0.20)

X = np.stack([F[k].ravel() for k in FEATS], axis=1).astype(np.float32)
yv = Y.ravel(); fwdv = FWD.ravel()
tidx = np.repeat(np.arange(T), N)
okf = np.isfinite(X).all(axis=1) & np.isfinite(fwdv)
say('可用样本 %d（%.0fs）' % (int(okf.sum()), time.time() - t0))

# 用前 60% 训练模型（模拟"只能用历史"），对全期出概率
cut = int(T * 0.6)
tr = okf & (tidx < cut)
te = okf & (tidx >= cut)
say('训练 %d（前 60%%）  预测 %d（后 40%%）' % (int(tr.sum()), int(te.sum())))
clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=6,
                                     early_stopping=True, validation_fraction=0.1, random_state=0)
clf.fit(X[tr], yv[tr])
p_all = np.full(T * N, np.nan, np.float32)
p_all[te] = clf.predict_proba(X[te])[:, 1]
Pmat = p_all.reshape(T, N)
say('模型训练+预测完成  %.0fs' % (time.time() - t0))

# 手工信号（去重）发生在后 40% 的部分
def dedup_rows(S, hold=HOLD):
    out = np.zeros_like(S); last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            out[i] = sel; last[sel] = i
    return out

S = dedup_rows(SIGM)
ii, jj = np.where(S)
keep = (ii >= cut) & np.isfinite(Pmat[ii, jj]) & np.isfinite(FWD[ii, jj])
ii, jj = ii[keep], jj[keep]
y = FWD[ii, jj]; p = Pmat[ii, jj]
say('后 40%% 区段内手工信号 = %d 笔' % len(y))

say('\n' + '=' * 96)
say('【对比】手工信号 + 模型概率过滤（后 40% 测试区）')
say('  %-28s %7s %10s %10s %11s %11s' % ('口径', 'n', '胜率%', '毛收益%', '净maker%', '净taker%'))
def line(nm, m):
    if m.sum() < 5:
        say('  %-28s 样本不足 (%d)' % (nm, int(m.sum()))); return None
    yy = y[m]
    return dict(n=int(m.sum()), win=100 * (yy > 0).mean(), gross=100 * yy.mean(),
                netm=100 * (yy.mean() - 2 * FEE_M), nett=100 * (yy.mean() - 2 * FEE_T))
for nm, m in (('手工信号（全部）', np.ones(len(y), bool)),
              ('手工 + 模型 p>0.50', p > 0.50),
              ('手工 + 模型 p>0.55', p > 0.55),
              ('手工 + 模型 p>0.60', p > 0.60),
              ('手工 + 模型 p 最低 50% 剔除', p > np.nanmedian(p))):
    d = line(nm, m)
    if d:
        say('  %-28s %7d %9.1f%% %+9.4f%% %+10.4f%% %+10.4f%%' % (
            nm, d['n'], d['win'], d['gross'], d['netm'], d['nett']))

say('\n【按模型概率分档看手工信号（这才是关键）】')
qs = np.nanpercentile(p, [0, 20, 40, 60, 80, 100])
for a, b in zip(qs[:-1], qs[1:]):
    m = (p >= a) & (p < b if b < qs[-1] else p <= b)
    if m.sum() < 5:
        continue
    yy = y[m]
    say('  p∈[%.3f, %.3f)  n=%4d  胜率=%.1f%%  净maker=%+.4f%%' % (
        a, b, m.sum(), 100 * (yy > 0).mean(), 100 * (yy.mean() - 2 * FEE_M)))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].scatter(p, 100 * y, s=4, alpha=.3)
ax[0].axhline(0, color='k', lw=.8); ax[0].axvline(0.5, color='k', ls='--', lw=.8)
ax[0].set_xlabel('model p'); ax[0].set_ylabel('trade return %')
ax[0].set_title('manual signals colored by model probability')
names, wins = [], []
for nm, m in (('all', np.ones(len(y), bool)), ('p>0.5', p > 0.5), ('p>0.55', p > 0.55), ('p>0.6', p > 0.6)):
    if m.sum() >= 5:
        names.append('%s(n=%d)' % (nm, m.sum())); wins.append(100 * (y[m] > 0).mean())
ax[1].bar(range(len(names)), wins); ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=20, fontsize=8); ax[1].axhline(71.19, color='k', ls='--', lw=.8)
ax[1].set_title('win rate by gate (dashed = manual-alone)')
plt.tight_layout()
pth = os.path.join(EV, 'model_gate.png'); plt.savefig(pth, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % pth)
open(os.path.join(EV, 'model_gate.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
