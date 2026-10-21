# -*- coding: utf-8 -*-
"""gambler_longrun_5m · 连续 2 年运行验证（5m 3 年数据）

训练：2023-10 ~ 2024-09（12 个月）
测试：2024-10 ~ 2026-10（24 个月，连续不中断）
目的：回答「有没有做过连续几年的测试」
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
START = 50.0
POS = 0.10
MAXSYM = int(os.environ.get('SYMS', '60'))

VW = 240
CONF = 2880
HOLD = 1440
TPM, SLM = 0.30, 0.95

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_longrun_5m · 训练 1 年 + 连续测试 2 年')
log('=' * 100)

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_5m_3y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = pd.Series(l).rolling(121, min_periods=1).min().values
    lo_i = []
    i = 120
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 30
        else:
            i += 1
    pairs = []
    for i in lo_i:
        if i + CONF >= n: continue
        for j in range(i + 1, min(i + CONF + 1, n)):
            if c[j] >= l[i] * 1.03:
                pairs.append((i, j)); break
    return pairs


def build(pairs, o, h, l, c, t, v, tb):
    n = len(c)
    feats, entries = [], []
    hist_hits = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]; lo_p = l[li]
        tp_px = lo_p * (1.0 + TPM)
        sl_px = lo_p * SLM
        seg_h = h[j + 1:j + 1 + HOLD]
        seg_l = l[j + 1:j + 1 + HOLD]
        hit_tp = seg_h >= tp_px
        hit_sl = seg_l <= sl_px
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
        if i_tp >= 0 and i_sl >= 0:
            if i_tp < i_sl:
                ret = tp_px / e - 1; is_tp = 1
            else:
                ret = sl_px / e - 1; is_tp = 0
        elif i_tp >= 0:
            ret = tp_px / e - 1; is_tp = 1
        elif i_sl >= 0:
            ret = sl_px / e - 1; is_tp = 0
        else:
            ret = c[min(j + HOLD, n - 1)] / e - 1; is_tp = 0

        vw = v[j - VW:j + 1]
        v240 = float(np.std(np.log1p(np.maximum(vw, 0))))
        vm = float(np.mean(vw))
        volr = float(v[j] / vm) if vm > 0 else 1.0
        vb = np.sum(v[j - 14:j])
        tbr = float(np.sum(tb[j - 14:j]) / vb) if vb > 0 else 0.5
        r1 = e / c[j - 1] - 1 if j >= 1 else 0.0
        r15 = e / c[j - 15] - 1 if j >= 15 else 0.0
        r60 = e / c[j - 60] - 1 if j >= 60 else 0.0
        ts = pd.Timestamp(int(t[j]) // 1000, unit='s', tz='UTC')
        pos = e / lo_p if lo_p > 0 else 1.0
        lo_age = j - li
        h3 = float(np.mean(hist_hits[-3:])) if hist_hits else -1.0
        h10 = float(np.mean(hist_hits[-10:])) if hist_hits else -1.0
        feats.append([volr, tbr, v240, pos, lo_age, r1, r15, r60, ts.hour, ts.dayofweek, h3, h10])
        entries.append((int(t[j]), ret, is_tp))
        hist_hits.append(is_tp)
    return np.asarray(feats), entries


ROWS = []
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    X, E = build(pairs, o, h, l, c, t, v, tb)
    for (ts, ret, is_tp), feat in zip(E, X):
        ROWS.append((ts, ret, is_tp, feat))
log(f'入场点 {len(ROWS):,}  耗时 {time.time()-t0:.0f}s')

df = pd.DataFrame({'t': [r[0] for r in ROWS], 'ret': [r[1] for r in ROWS],
                   'tp': [r[2] for r in ROWS]})
X = np.array([r[3] for r in ROWS], dtype=float)
df['ym'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
months = sorted(df['ym'].unique())
log(f'时间跨度 {months[0]} ~ {months[-1]}（{len(months)} 月）')

# 训练 2023-10~2024-09，测试 2024-10~2026-10（连续 24 个月）
train_m = [m for m in months if m <= '2024-09']
test_m = [m for m in months if m >= '2024-10']
tr = np.where(df['ym'].isin(train_m).values)[0]
te = np.where(df['ym'].isin(test_m).values)[0]
log(f'训练 {len(tr):,}（{train_m[0]}~{train_m[-1]}，{len(train_m)} 月）')
log(f'测试 {len(te):,}（{test_m[0]}~{test_m[-1]}，{len(test_m)} 月连续）')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Pte = mdl.predict_proba(X[te])[:, 1]
log(f'AUC 样本外 {roc_auc_score(df["tp"].values[te], Pte):.4f}')

sel = te[Pte >= 0.5]
d = df.iloc[sel].copy()
d = d.sort_values('t')
log(f'P>=0.5 交易 {len(d)} 笔')

cap = START; peak = START; dd = 0.0; curve = []
for _, row in d.iterrows():
    cap *= (1 + POS * (row['ret'] - FEE))
    peak = max(peak, cap)
    dd = max(dd, (peak - cap) / peak)
    curve.append((row['t'], cap, row['ret']))
log('')
log('=' * 100)
log('【连续 2 年测试结果】')
log(f'  {len(d)} 笔  单笔均值 {d["ret"].mean()*100:.2f}%  胜率 {(d["ret"]>0).mean()*100:.1f}%  '
    f'最终 {cap:,.0f} 元  回撤 {dd*100:.1f}%')

log('')
log('【按月（连续 24 个月）】')
cvm = pd.DataFrame(curve, columns=['t', 'cap', 'ret'])
cvm['ym'] = pd.to_datetime(cvm['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
g = cvm.groupby('ym').agg(n=('ret', 'count'), mean=('ret', 'mean'), last=('cap', 'last'))
prev = START
for ym, row in g.iterrows():
    log(f'  {ym}  {int(row["n"]):>4} 笔  均值 {row["mean"]*100:>6.2f}%  月末 {row["last"]:>14,.0f} 元  {(row["last"]/prev-1)*100:>9.1f}%')
    prev = row['last']

log('')
log('=' * 100)
log('【判读】')
log('  · 这是「训练 1 年 + 连续 24 个月不中断」的验证')
log('  · 逐月看：牛市月笔数多、熊市月笔数少，但资金曲线是否持续向上')
log('  · 复利数字是数学放大，看单笔均值与胜率是否稳定')

out = os.path.join(OUT, 'longrun_5m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
