# -*- coding: utf-8 -*-
"""gambler_mid.py — 在"拉升过程中"分档：已经涨了 5~40% 时，能预测最终涨到哪一档吗？
关键发现（2026-10-16，前三轮实验）：
  在"局起点"时刻分档**全部失败**——真局(涨>=50%)与微局(涨10~20%)的起点特征几乎相同，
  因为"会涨多少"是**未来的事**，起点特征（贴低点/缩量/横盘）里没有这个信息。
  ⇒ 放宽真局定义到 494 个样本后，真局仍是 0 召回、0 精确率。

本脚本换一个位置问同一件事：**拉升已经启动之后**（t 时刻已涨 5~40%），
  用**过程特征**（涨速、量能放大节奏、回踩、市场同步、已持续时间）预测**最终涨幅档**。
  这才是用户真正要的"分得清是哪一种局"——因为决策发生在"已经看到它在动"之后。

标签（从 t 起未来 288 根内最大涨幅）：
    1 微(<20%)   2 弱(20~50%)   3 中(50~100%)   4 真(>=100%)
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_mid.txt')
WMAX = 288
C4 = ['微(<20%)', '弱(20~50%)', '中(50~100%)', '真(>=100%)']

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_mid · 在拉升过程中分档（已涨 5~40% 时，预测最终涨幅档）')
log('=' * 100)

FEAT = ['m_ret6','m_ret12','m_ret24','m_ret48','m_ret96',
        'm_volr6','m_volr12','m_volr24','m_volr48',
        'm_trr6','m_trr12','m_trr24',
        'm_over_lo','m_over_lo96','m_run_up','m_pullback','m_draw',
        'm_accel','m_vol_trend','m_tr_trend','m_evt',
        'm_sync','m_btc6','m_btc24','m_hour_sin','m_hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values
T, N = C.shape
R1all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R1all, nan=0.0) >= 0.05).sum(axis=1)
mkt_up_g = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

X, Y, TT = [], [], []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WMAX + 300: continue
    lo288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    fmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    fut = fmax / c - 1.0                       # 未来 288 根内最大涨幅
    # 采样：已经涨了 5~40%（拉升中），且未来可测
    sel = (r12 >= 0.05) & (over >= 0.05) & (over <= 0.40) & np.isfinite(fut)
    idx = np.flatnonzero(sel)
    if len(idx) < 300: continue
    keep = np.zeros(len(idx), dtype=bool); keep[::8] = True
    keep |= (fut[idx] >= 1.0)
    idx = idx[keep]
    if len(idx) < 100: continue

    y = np.full(len(idx), -1, dtype='int8')
    fv = fut[idx]
    y[fv < 0.20] = 1
    y[(fv >= 0.20) & (fv < 0.50)] = 2
    y[(fv >= 0.50) & (fv < 1.00)] = 3
    y[fv >= 1.00] = 4

    F = {}
    F['m_ret6']  = c / np.append([np.nan]*6,  c[:-6])  - 1.0
    F['m_ret12'] = c / np.append([np.nan]*12, c[:-12]) - 1.0
    F['m_ret24'] = c / np.append([np.nan]*24, c[:-24]) - 1.0
    F['m_ret48'] = c / np.append([np.nan]*48, c[:-48]) - 1.0
    F['m_ret96'] = c / np.append([np.nan]*96, c[:-96]) - 1.0
    for k in (6, 12, 24, 48):
        mv = rmean(q, k); F['m_volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
    for k in (6, 12, 24):
        mt = rmean(tr, k); F['m_trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    F['m_over_lo'] = over
    lo96 = pd.Series(l).rolling(96, min_periods=24).min().values
    F['m_over_lo96'] = c / np.maximum(lo96, 1e-12) - 1.0
    # 从 288 根低点到当前的持续根数
    is_low_seen = (l <= lo288 * 1.001)
    ix = np.arange(n)
    F['m_run_up'] = (ix - np.maximum.accumulate(np.where(is_low_seen, ix, 0)) + 1).astype('float64')
    # 回踩：当前价 / 自启动以来最高
    hi_since = pd.Series(h).rolling(48, min_periods=1).max().values
    F['m_pullback'] = c / np.maximum(hi_since, 1e-12) - 1.0
    # 最大回撤（自 288 根低点起）
    F['m_draw'] = hi_since / np.maximum(lo288, 1e-12) - 1.0
    F['m_accel'] = F['m_ret6'] - F['m_ret24'] / 4.0
    v6 = rmean(q, 6); v48 = rmean(q, 48)
    F['m_vol_trend'] = np.where(v48 > 0, v6 / np.maximum(v48, 1e-12), np.nan)
    t6 = rmean(tr, 6); t48 = rmean(tr, 48)
    F['m_tr_trend'] = np.where(t48 > 0, t6 / np.maximum(t48, 1e-12), np.nan)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    av = pd.Series(np.abs(r1)).rolling(48, min_periods=12).mean().values
    F['m_evt'] = pd.Series((np.abs(r1) > 3 * av).astype('float32')).rolling(48, min_periods=12).mean().values
    ii = pd.Index(ot)
    F['m_sync'] = pd.Series(mkt_up_g, index=ts).reindex(ii).values
    F['m_btc6'] = pd.Series(btc_r6, index=ts).reindex(ii).values
    F['m_btc24'] = pd.Series(btc_r24, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24
    F['m_hour_sin'] = np.sin(2 * np.pi * hh / 24.0)
    F['m_hour_cos'] = np.cos(2 * np.pi * hh / 24.0)

    X.append(np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32'))
    Y.append(y); TT.append(ot[idx])
    del F; gc.collect()

X = np.nan_to_num(np.concatenate(X), nan=0.0, posinf=0.0, neginf=0.0)
Y = np.concatenate(Y); TT = np.concatenate(TT)
log(f'样本（拉升中）{len(Y):,}')
for k in (1, 2, 3, 4):
    log(f'  {C4[k-1]:<14} {int((Y==k).sum()):>9,}  ({100.0*(Y==k).mean():.2f}%)')

cut = np.quantile(TT, 0.70)
tr = TT <= cut; te = ~tr
log(f'训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')
cnt = np.array([(Y[tr] == k).sum() for k in (1, 2, 3, 4)], dtype='float64')
w = np.array([1.0 / max(cnt[k-1], 1.0) for k in Y[tr]])
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=6,
                                     min_samples_leaf=20, l2_regularization=1.0, random_state=0)
clf.fit(X[tr], Y[tr], sample_weight=w)
cls = clf.classes_
pred = cls[np.argmax(clf.predict_proba(X[te]), axis=1)]
truth = Y[te]

log('')
log('=' * 100)
log('【混淆矩阵】行=真实 列=预测')
log('=' * 100)
labs = sorted(set(list(np.unique(truth)) + list(np.unique(pred))))
cm = confusion_matrix(truth, pred, labels=labs)
log('  ' + ' ' * 16 + ''.join(f'{C4[l-1][:12]:>14}' for l in labs))
for ri, l in enumerate(labs):
    log(f'  真{C4[l-1][:12]:<12}' + ''.join(f'{cm[ri][ci]:>14,}' for ci in range(len(labs))))
log('')
log('【各类指标】')
for l in labs:
    r = labs.index(l)
    tp = cm[r][r]; fp = sum(cm[x][r] for x in range(len(labs))) - tp; fn = sum(cm[r]) - tp
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else float('nan')
    rec = 100.0 * tp / (tp + fn) if (tp + fn) else float('nan')
    log(f'  {C4[l-1]:<14} 样本 {int(sum(cm[r])):>8,}  精确率 {prec:>6.1f}%  召回 {rec:>6.1f}%')
log(f'  总体准确率 {100.0*(pred == truth).mean():.1f}%   基准（全判微）{100.0*(truth==1).mean():.1f}%')
m4 = (pred == 4)
log(f'  ★ 模型报"真档(>=100%)" {int(m4.sum()):,} 个，其中真 {int((truth[m4]==4).sum()):,}'
    f'（精确率 {100.0*(truth[m4]==4).mean() if m4.sum() else float("nan"):.1f}%）')
m4t = (truth == 4)
if m4t.sum():
    log(f'  ★ 真实"真档" {int(m4t.sum()):,} 个，抓到 {int((pred[m4t]==4).sum()):,}'
        f'（召回 {100.0*(pred[m4t]==4).mean():.1f}%）')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
