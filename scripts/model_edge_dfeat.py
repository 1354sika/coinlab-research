# -*- coding: utf-8 -*-
"""期望收益增强判据：基线(13特征) vs 基线+D1~D8+F6(22特征)

目的（交接文档 §6 优先级1）：
  把「时间结构算子 D1~D8」与「量价背离 F6」加入模型特征，
  看单笔期望净收益能否从基线的 +0.0544%/笔 再抬升。

对照设计（同数据、同切分、同标签，唯一变量=特征集）：
  A = 基线特征（等价 model_edge.py 的 13 个）
  B = A + D1~D8 + F6（+9 个）
数据：klines_1m_1y.csv（60 币 × 1 年 × 1m）
切分：前 60% 训练、后 40% 测试（非随机，与 model_edge.py 一致）
标签：下一分钟收益 8 档；核心看 P(涨>0.5%) / P(跌>0.5%) 分档的净 maker 期望
"""
import os, sys, gc, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ops_duration import (d1_streak_len, d2_seg_speed, d3_seg_amp,
                          d4_jump_flag, d5_drift_flag, d6_seg_composition,
                          d7_frozen_len, d8_market_frozen_ratio)
from ops_fingerprint import f6_divergence

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
EV = os.path.join(ROOT, 'evidence')
FEE_M, FEE_T = 0.0002, 0.0010
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('期望收益增强对照：基线(13特征) vs 基线+D1~D8+F6(22特征)   数据=klines_1m_1y.csv')
say('=' * 100)

# ---------------- 读 ----------------
df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=5)
qv = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
T, N = close.shape
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))
del df; gc.collect()

ret1 = close.pct_change().astype('float32')
mkt1 = ret1.median(axis=1)
disp = ret1.std(axis=1)
def rep(s):
    return pd.DataFrame(np.repeat(np.asarray(s, dtype='float32')[:, None], N, axis=1),
                        index=close.index, columns=close.columns)

# ---------------- A：基线特征（= model_edge.py 13 个） ----------------
volr = qv / qv.rolling(60, min_periods=20).median()
ext_vol = (volr > 5).astype('float32')
n_pump = rep(ext_vol.sum(axis=1))
qv_rank = qv.rank(axis=1, pct=True)
base = {
    'ret1':    ret1,
    'ret5':    close.pct_change(5),
    'ret15':   close.pct_change(15),
    'ret60':   close.pct_change(60),
    'volr':    volr,
    'dist_ma': close / close.rolling(60, min_periods=30).mean() - 1,
    'mkt1':    rep(mkt1),
    'disp':    rep(disp),
    'mktvol':  rep(mkt1.rolling(60, min_periods=20).std()),
    'ext_vol': ext_vol,
    'n_pump':  n_pump,
    'qv_rank': qv_rank,
}
base['rk_ret'] = base['ret15'].rank(axis=1, pct=True)

# ---------------- B：新增 D1~D8 + F6 ----------------
say('算 D1~D8 ...  %.0fs' % (time.time() - t0))
new = {}
streak = d1_streak_len(ret1)
new['D1_streak'] = streak
new['D2_speed'] = d2_seg_speed(close, streak)
new['D3_amp'] = d3_seg_amp(close, streak)
new['D4_jump'] = d4_jump_flag(ret1, 240, 3.0)
new['D5_drift'] = d5_drift_flag(ret1, 240, 0.5)
sc, _asl = d6_seg_composition(ret1, 240)
new['D6_segcnt'] = sc.reindex(ret1.index)   # d6 少一行(chg)，对齐补 NaN
new['D7_frozen'] = d7_frozen_len(ret1)
_, d8r = d8_market_frozen_ratio(ret1, 60)
new['D8_frozen'] = rep(d8r)
del streak, sc, _asl; gc.collect()
_nh, _vr, div = f6_divergence(close, qv)
new['F6_div'] = div
del _nh, _vr; gc.collect()
say('新增特征 %d 个  %.0fs' % (len(new), time.time() - t0))

names_base = list(base.keys())
names_new = list(new.keys())
names = names_base + names_new
allF = {**base, **new}
say('A 特征 %d 个：%s' % (len(names_base), names_base))
say('B 特征 %d 个：%s' % (len(names), names_new))

cols = []
for k in names:
    v = allF[k]
    if not isinstance(v, pd.DataFrame):
        v = pd.DataFrame(np.asarray(v, dtype='float32'), index=close.index, columns=close.columns)
    if v.shape != (T, N):
        v = v.reindex(close.index)
    cols.append(v.values.ravel().astype('float32'))
X = np.stack(cols, axis=1)
del cols
del allF, base, new, volr, ext_vol, n_pump, qv_rank; gc.collect()

# ---------------- 标签 ----------------
EDGES = [-np.inf, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, np.inf]
nxt = close.shift(-1) / close - 1.0
R = nxt.values.ravel().astype('float32')
Y = np.digitize(R, EDGES[1:-1]).astype('int8')

ok = np.isfinite(X).all(axis=1) & np.isfinite(R)
X = np.ascontiguousarray(X[ok]); Y = Y[ok]; R = R[ok]
say('可用样本 %s   %.0fs' % (format(len(X), ','), time.time() - t0))

# ---------------- 切分（两套共用同一训练索引，保证唯一变量=特征） ----------------
a = int(len(X) * 0.6)
TRMAX = 6000000
rng = np.random.default_rng(7)
if a > TRMAX:
    tr = rng.choice(a, TRMAX, replace=False); tr.sort()
else:
    tr = np.arange(a)
say('训练边界 a=%s  实际训练 %s  测试 %s' % (format(a, ','), format(len(tr), ','), format(len(X) - a, ',')))

idx_base = list(range(len(names_base)))


def run(Xsub, tag):
    """训练 -> 测试段概率（8 档） -> 返回 pup05/pdn05/Yv/Rv"""
    m = HistGradientBoostingClassifier(max_iter=70, learning_rate=0.08, max_depth=6,
                                       max_bins=128, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=42,
                                       early_stopping=False)
    m.fit(Xsub[tr], Y[tr])
    b = len(Xsub)
    P = np.zeros((b - a, 8), dtype='float32')
    STEP = 2000000
    for s0 in range(a, b, STEP):
        s1 = min(s0 + STEP, b)
        pp = m.predict_proba(Xsub[s0:s1])
        tmp = np.zeros((s1 - s0, 8), dtype='float32')
        for ci, cls in enumerate(m.classes_):
            tmp[:, int(cls)] = pp[:, ci]
        P[s0 - a:s1 - a] = tmp
    pup = P[:, [6, 7]].sum(axis=1)
    pdn = P[:, [0, 1]].sum(axis=1)
    say('  [%s] 训练+预测完成  %.0fs' % (tag, time.time() - t0))
    return pup, pdn


tables = {}
for tag, cols in (('A_base', idx_base), ('B_dfeat', None)):
    Xsub = X if cols is None else np.ascontiguousarray(X[:, cols])
    pup, pdn = run(Xsub, tag)
    del Xsub; gc.collect()
    Yv, Rv = Y[a:], R[a:]
    tables[tag] = (pup, pdn, Yv, Rv)
    np.savez(os.path.join(EV, 'model_edge_dfeat_%s.npz' % tag), pup=pup, pdn=pdn, Yv=Yv, Rv=Rv)

# ---------------- 报告 ----------------
bins = [0, .02, .05, .10, .20, .30, .40, .50, .70, 1.01]
summ = {}
for tag in ('A_base', 'B_dfeat'):
    pup, pdn, Yv, Rv = tables[tag]
    say('\n' + '=' * 100)
    say('【%s】P(涨>0.5%%) 分档（做多期望）' % tag)
    say('  %-13s %10s %10s %11s %11s' % ('p(涨>0.5%)', '样本', '实际>0.5%', 'E毛%', '净maker%'))
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (pup >= lo) & (pup < hi)
        n = int(mm.sum())
        if n < 200:
            continue
        gm = Rv[mm].mean() * 100
        say('  [%.2f,%.2f)  %10s %9.2f%% %+10.4f%% %+10.4f%%'
            % (lo, hi, format(n, ','), (Yv[mm] >= 6).mean() * 100, gm, gm - 2 * FEE_M * 100))
    say('  【%s】P(跌>0.5%%) 分档（做空期望）' % tag)
    say('  %-13s %10s %10s %11s %11s' % ('p(跌>0.5%)', '样本', '实际<-0.5%', 'E毛做空%', '净maker%'))
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (pdn >= lo) & (pdn < hi)
        n = int(mm.sum())
        if n < 200:
            continue
        gm = -Rv[mm].mean() * 100
        say('  [%.2f,%.2f)  %10s %9.2f%% %+10.4f%% %+10.4f%%'
            % (lo, hi, format(n, ','), (Yv[mm] <= 1).mean() * 100, gm, gm - 2 * FEE_M * 100))
    # 阈值扫描：正期望截点
    nl = ns = 0
    best_l = best_s = None
    for t in np.arange(0.05, 0.96, 0.01):
        ml = pup >= t
        if ml.sum() >= 500:
            net = Rv[ml].mean() * 100 - 2 * FEE_M * 100
            if net > 0:
                nl += 1
                if best_l is None or net > best_l[1]:
                    best_l = (t, net, int(ml.sum()))
        ms = pdn >= t
        if ms.sum() >= 500:
            net = -Rv[ms].mean() * 100 - 2 * FEE_M * 100
            if net > 0:
                ns += 1
                if best_s is None or net > best_s[1]:
                    best_s = (t, net, int(ms.sum()))
    auc_l = roc_auc_score((Rv > 0.005).astype(int), pup)
    auc_s = roc_auc_score((Rv < -0.005).astype(int), pdn)
    summ[tag] = dict(nl=nl, ns=ns, best_l=best_l, best_s=best_s,
                     auc_l=auc_l, auc_s=auc_s)
    say('  正期望阈值个数：做多 %d / 做空 %d' % (nl, ns))
    say('  做多最好截点：%s' % (best_l,))
    say('  做空最好截点：%s' % (best_s,))
    say('  AUC 涨>0.5%% = %.4f   AUC 跌>0.5%% = %.4f' % (auc_l, auc_s))

say('\n' + '=' * 100)
say('【对照总表】唯一变量 = 特征集（同数据/同切分/同标签/同训练索引）')
say('=' * 100)
say('  %-10s %10s %10s %14s %14s %9s %9s' % (
    '特征集', '做多正截点', '做空正截点', '做多最优净%', '做空最优净%', 'AUC涨', 'AUC跌'))
for tag in ('A_base', 'B_dfeat'):
    s = summ[tag]
    bl = '%+.4f%%' % s['best_l'][1] if s['best_l'] else 'None'
    bs = '%+.4f%%' % s['best_s'][1] if s['best_s'] else 'None'
    say('  %-10s %10d %10d %14s %14s %9.4f %9.4f' % (
        tag, s['nl'], s['ns'], bl, bs, s['auc_l'], s['auc_s']))
dl = summ['B_dfeat']['best_l'][1] - summ['A_base']['best_l'][1] if (summ['B_dfeat']['best_l'] and summ['A_base']['best_l']) else float('nan')
ds = summ['B_dfeat']['best_s'][1] - summ['A_base']['best_s'][1] if (summ['B_dfeat']['best_s'] and summ['A_base']['best_s']) else float('nan')
say('  增强 vs 基线：做多最优净 %+.4f pp   做空最优净 %+.4f pp' % (dl, ds))
say('  增强 vs 基线：AUC涨 %+.4f   AUC跌 %+.4f'
    % (summ['B_dfeat']['auc_l'] - summ['A_base']['auc_l'],
       summ['B_dfeat']['auc_s'] - summ['A_base']['auc_s']))

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
for tag, col, mk in (('A_base', '0.45', 'o'), ('B_dfeat', 'C0', 's')):
    pup, pdn, Yv, Rv = tables[tag]
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (pdn >= lo) & (pdn < hi)
        if mm.sum() >= 200:
            xs.append((lo + hi) / 2); ys.append(-Rv[mm].mean() * 100)
    axes[0].plot(xs, ys, mk + '-', color=col, label=tag, lw=1.8)
axes[0].axhline(2 * FEE_M * 100, ls='--', color='k', lw=1, label='maker 往返成本')
axes[0].set_xlabel('P(跌>0.5%)'); axes[0].set_ylabel('E[毛]做空 %/笔')
axes[0].set_title('short: prob vs expected return'); axes[0].legend(fontsize=9); axes[0].grid(alpha=.3)
for tag, col, mk in (('A_base', '0.45', 'o'), ('B_dfeat', 'C0', 's')):
    pup, pdn, Yv, Rv = tables[tag]
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (pup >= lo) & (pup < hi)
        if mm.sum() >= 200:
            xs.append((lo + hi) / 2); ys.append(Rv[mm].mean() * 100)
    axes[1].plot(xs, ys, mk + '-', color=col, label=tag, lw=1.8)
axes[1].axhline(2 * FEE_M * 100, ls='--', color='k', lw=1)
axes[1].set_xlabel('P(涨>0.5%)'); axes[1].set_ylabel('E[毛]做多 %/笔')
axes[1].set_title('long: prob vs expected return'); axes[1].legend(fontsize=9); axes[1].grid(alpha=.3)
x = np.arange(4); w = 0.35
vals_a = [summ['A_base']['nl'], summ['A_base']['ns'], summ['B_dfeat']['nl'], summ['B_dfeat']['ns']]
axes[2].bar(x, vals_a, color=['0.6', '0.6', 'C0', 'C0'])
axes[2].set_xticks(x); axes[2].set_xticklabels(['A_long', 'A_short', 'B_long', 'B_short'], fontsize=9)
axes[2].set_ylabel('count of positive-edge thresholds'); axes[2].set_title('positive-edge截点个数')
plt.tight_layout()
pth = os.path.join(EV, 'model_edge_dfeat.png')
plt.savefig(pth, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % pth)

open(os.path.join(EV, 'model_edge_dfeat.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
