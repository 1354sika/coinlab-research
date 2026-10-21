# -*- coding: utf-8 -*-
"""gambler_annual · 严格按日历切分验证大周期（1 年数据）

数据：klines_1m_1y.csv（60 币，1 年）
切分：前 9 个月训练 / 后 3 个月测试（时间上完全隔开）
大周期口径同原版：入场=低点×1.03，止盈=低点×1.30，止损=低点×0.95，24h 超时，P>=0.5
成本 0.50%/笔，每笔 10% 仓位
"""
import os, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

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
log('gambler_annual · 严格按日历切分验证大周期')
log('=' * 100)
log('数据 klines_1m_1y.csv | 前 9 个月训练 / 后 3 个月测试 | 成本 0.50%/笔')
log('口径：入场=低点×1.03，止盈=低点×1.30（+26.2%），止损=低点×0.95（-7.8%），24h 超时')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=use, chunksize=2_000_000):
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
    """只用过去 121 根判定低点（无前视）

    原版用中心窗口 min(l[i-60:i+61]) 含未来 60 根 → 前视偏差。
    改为 rolling(121) 不含未来：索引 i 的值 = min(l[i-120:i+1])。
    """
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
        hr = ts.hour
        dw = ts.dayofweek
        pos = e / lo_p if lo_p > 0 else 1.0
        lo_age = j - li
        h3 = float(np.mean(hist_hits[-3:])) if hist_hits else -1.0
        h10 = float(np.mean(hist_hits[-10:])) if hist_hits else -1.0
        feats.append([volr, tbr, v240, pos, lo_age, r1, r15, r60, hr, dw, h3, h10])
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
log(f'时间跨度 {months[0]} ~ {months[-1]}  共 {len(months)} 个月')
if len(months) >= 4:
    ntr = max(1, int(len(months) * 0.75))
    train_m = months[:ntr]; test_m = months[ntr:]
else:
    train_m = months[:1]; test_m = months[1:]
log(f'训练月 {train_m[0]}~{train_m[-1]}（{len(train_m)} 月）  测试月 {test_m[0]}~{test_m[-1]}（{len(test_m)} 月）')

tr = np.where(df['ym'].isin(train_m).values)[0]
te = np.where(df['ym'].isin(test_m).values)[0]
log(f'训练 {len(tr):,} 笔（TP 率 {df["tp"].values[tr].mean()*100:.2f}%）  测试 {len(te):,} 笔（TP 率 {df["tp"].values[te].mean()*100:.2f}%）')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Ptr = mdl.predict_proba(X[tr])[:, 1]
Pte = mdl.predict_proba(X[te])[:, 1]
log(f'AUC 训练(样本内) {roc_auc_score(df["tp"].values[tr], Ptr):.4f}   测试(样本外) {roc_auc_score(df["tp"].values[te], Pte):.4f}')


def sim(E, P, thr):
    trades = []
    for k in range(len(E)):
        if P[k] >= thr:
            trades.append((E[k][0], E[k][1] - FEE))
    if not trades: return None
    trades.sort(key=lambda x: x[0])
    cap = START; peak = START; dd = 0.0; curve = []
    for ts, net in trades:
        cap *= (1 + POS * net)
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
        curve.append((ts, cap))
    return dict(n=len(trades), cap=cap, dd=dd, curve=curve,
                mean=np.mean([x[1] + FEE for x in trades]) * 100,
                win=np.mean([x[1] > 0 for x in trades]) * 100)


Etr = [ROWS[i] for i in tr]
Ete = [ROWS[i] for i in te]
a = sim(Etr, Ptr, 0.5)
b = sim(Ete, Pte, 0.5)
log('')
log('=' * 100)
log('【严格按日历切分 · 大周期 P>=0.5】')
for lab, s in [('训练段（前 %d 月，样本内）' % len(train_m), a), ('测试段（后 %d 月，样本外）' % len(test_m), b)]:
    if s is None:
        log(f'  {lab}: 无交易')
        continue
    log(f'  {lab}: {s["n"]:>5} 笔  均值 {s["mean"]:>6.2f}%  胜率 {s["win"]:>5.1f}%  {START:.0f} → {s["cap"]:>14,.0f} 元  回撤 {s["dd"]*100:>5.1f}%')

log('')
log('【测试段按月】')
if b:
    cv = pd.DataFrame(b['curve'], columns=['t', 'cap'])
    cv['m'] = pd.to_datetime(cv['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
    g = cv.groupby('m')['cap'].agg(['last', 'count'])
    prev = START
    for row in g.itertuples():
        m, last, cnt = row[0], row[1], row[2]
        log(f'  {m}  {cnt:>4} 笔  {last:>14,.0f} 元  {(last/prev-1)*100:>8.1f}%')
        prev = last

log('')
log('=' * 100)
log('【判读】')
log('  · 训练与测试在日历上完全隔开，无交错')
log('  · 测试段（样本外）若仍显著为正 ⇒ 大周期算法有真实样本外能力')
log('  · 测试段若亏或笔数崩 ⇒ 之前的 97.8% 胜率是时间邻近导致的虚高')

out = os.path.join(OUT, 'annual_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
