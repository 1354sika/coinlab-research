# -*- coding: utf-8 -*-
"""概率预测模型：给定全市场状态 -> 预测每个币下一小时上涨的概率

用户方向：不再找"谁领先谁"的显性规律，改用概率模型覆盖盲区。
本脚本建第一个基线模型，严格 walk-forward（按时间切分，绝不随机划分）。

特征（全部为 t 时刻可见，无前视）：
  自身：ret1 / ret12 / ret288 / 量比 / 1h波动 / 距1日均线 / 段内长度 / 段内收益
  市场：市场ret1 / 市场ret12 / 横截面分化度 / 市场1h波动
  横截面：该币 ret1 的排名 / 量比排名
标签：未来 12 根（=60 分钟）收益 > 0

评估（不看准确率，只看得失）：
  · AUC
  · 按预测概率分档 -> 各档真实胜率 + 扣费后单笔收益
  · 只在 p>阈值 时下注 -> 对比基线（无条件下注 47.37% / 手工规则 71.19%）
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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

HOLD, FEE_M, FEE_T = 12, 0.0002, 0.0010
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


def rollsum(a, w):
    return pd.Series(a).rolling(w).sum().values


def rollstd(a, w):
    return pd.Series(a).rolling(w).std().values


F = {}
F['ret1'] = ret1
F['ret12'] = np.full((T, N), np.nan); F['ret12'][12:] = P[12:] / P[:-12] - 1.0
F['ret288'] = np.full((T, N), np.nan); F['ret288'][288:] = P[288:] / P[:-288] - 1.0
F['volr'] = volr
F['vol1h'] = pd.DataFrame(ret1).rolling(12).std().values
ma288 = pd.DataFrame(P).rolling(D1).mean().values
F['dist_ma'] = P / ma288 - 1.0
# 段内
s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
F['seglen'] = seglen
F['segret'] = P / np.take_along_axis(P, start, axis=0) - 1.0
# 市场
F['mkt1'] = mkt1[:, None] * np.ones((1, N))
F['mkt12'] = rollsum(mkt1, 12)[:, None] * np.ones((1, N))
F['disp'] = disp[:, None] * np.ones((1, N))
F['mktvol'] = rollstd(mkt1, 12)[:, None] * np.ones((1, N))
# 横截面排名
r_ret = pd.DataFrame(ret1).rank(axis=1, pct=True).values
r_volr = pd.DataFrame(volr).rank(axis=1, pct=True).values
F['rk_ret'] = r_ret
F['rk_volr'] = r_volr

FWD = np.full((T, N), np.nan); FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
Y = (FWD > 0).astype(np.int8)
FEATS = list(F.keys())
say('特征 %d 个：%s' % (len(FEATS), FEATS))

# 展平
X = np.stack([F[k].ravel() for k in FEATS], axis=1).astype(np.float32)
yv = Y.ravel(); fwdv = FWD.ravel()
tidx = np.repeat(np.arange(T), N)
ok = np.isfinite(X).all(axis=1) & np.isfinite(fwdv)
X, yv, fwdv, tidx = X[ok], yv[ok], fwdv[ok], tidx[ok]
say('可用样本 = %d（%.0fs）' % (len(X), time.time() - t0))

# walk-forward：按时间切 4 段，前段训练、后段测试
q = pd.to_datetime(tms, unit='ms', utc=True)
edges = pd.date_range(q.min(), q.max(), periods=6)
say('\n' + '=' * 96)
say('【walk-forward】训练=历史，测试=其后 1/5 段（绝不随机划分）')
say('  %-24s %7s %7s %8s %10s %10s %10s' % ('测试段', '训练n', '测试n', 'AUC', 'p>0.5胜率', 'p>0.6胜率', 'p>0.6净maker'))
rows_out = []
for k in range(1, 5):
    tr = tidx < np.searchsorted(tms, int(edges[k].timestamp() * 1000))
    te = (tidx >= np.searchsorted(tms, int(edges[k].timestamp() * 1000))) & \
         (tidx < np.searchsorted(tms, int(edges[k + 1].timestamp() * 1000)))
    if tr.sum() < 50000 or te.sum() < 5000:
        continue
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                         max_depth=6, early_stopping=True,
                                         validation_fraction=0.1, random_state=0)
    clf.fit(X[tr], yv[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(yv[te], p) if len(np.unique(yv[te])) > 1 else np.nan
    r1 = yv[te][p > 0.5].mean() if (p > 0.5).sum() > 50 else np.nan
    r2 = yv[te][p > 0.6].mean() if (p > 0.6).sum() > 50 else np.nan
    net2 = (fwdv[te][p > 0.6] - 2 * FEE_M).mean() if (p > 0.6).sum() > 50 else np.nan
    rows_out.append((str(edges[k].date()), str(edges[k + 1].date()), int(tr.sum()), int(te.sum()),
                     auc, r1, r2, net2, p, yv[te], fwdv[te], te))
    say('  %s~%s %7d %7d %8.4f %9.1f%% %9.1f%% %+9.4f%%' % (
        str(edges[k].date()), str(edges[k + 1].date()), tr.sum(), te.sum(),
        auc, 100 * r1, 100 * r2, 100 * net2))

say('\n【汇总：整段拼接（只统计有模型的测试段）】')
if rows_out:
    P_all = np.concatenate([r[8] for r in rows_out])
    Y_all = np.concatenate([r[9] for r in rows_out])
    F_all = np.concatenate([r[10] for r in rows_out])
    auc_all = roc_auc_score(Y_all, P_all)
    say('  合并 AUC = %.4f   样本 %d' % (auc_all, len(P_all)))
    say('  %8s %10s %11s %12s %12s' % ('概率档', '样本', '实际胜率%', '毛收益%', '净maker%'))
    for lo, hi in ((0.0, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 1.0)):
        m = (P_all >= lo) & (P_all < hi)
        if m.sum() < 100:
            continue
        say('  %.2f-%.2f %10d %10.1f%% %+11.4f%% %+11.4f%%' % (
            lo, hi, m.sum(), 100 * Y_all[m].mean(),
            100 * F_all[m].mean(), 100 * (F_all[m] - 2 * FEE_M).mean()))
    say('\n【与基线对比】')
    say('  无条件下注胜率        = %.2f%%（前面实测）' % (100 * Y_all.mean()))
    best = 0.6
    mm = P_all > best
    say('  模型 p>%.2f 下注     = 胜率 %.2f%%、单笔净 maker %+.4f%%、次数 %d' % (
        best, 100 * Y_all[mm].mean(), 100 * (F_all[mm] - 2 * FEE_M).mean(), int(mm.sum())))
    say('  手工规则（急跌+巨量+多币）= 胜率 71.19%%（722 笔）')
    say('  成本门槛（maker）      = 需要 59%% 以上胜率')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(P_all, bins=60); ax[0].axvline(0.5, color='k', lw=.8)
    ax[0].set_title('predicted probability distribution')
    bins = np.arange(0, 1.01, 0.05); idxs = np.digitize(P_all, bins)
    xs, ys_ = [], []
    for b in range(1, len(bins)):
        m = idxs == b
        if m.sum() > 200:
            xs.append(bins[b - 1]); ys_.append(100 * Y_all[m].mean())
    ax[1].plot(xs, ys_, 'o-'); ax[1].plot([0, 1], [0, 100], 'k--', lw=.8)
    ax[1].set_xlabel('predicted p'); ax[1].set_ylabel('actual win rate %')
    ax[1].set_title('calibration')
    ax[2].plot(np.cumsum(F_all[np.argsort(-P_all)] - 2 * FEE_M) / max(len(F_all), 1) * 100)
    ax[2].axhline(0, color='k', lw=.8); ax[2].set_title('cumulative net % (sorted by p desc)')
    plt.tight_layout()
    pth = os.path.join(EV, 'model_proba.png'); plt.savefig(pth, dpi=125, bbox_inches='tight')
    say('\nsaved: %s' % pth)

open(os.path.join(EV, 'model_proba.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
