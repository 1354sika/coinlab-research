# -*- coding: utf-8 -*-
"""gambler_merge_1m_hist · 大周期算法跑整个历史

训练段训练模型 → 训练段(样本内) + 测试段(样本外) + 全历史合计
全部 1m 粒度；成本 0.50%/笔；50 元起步、每笔 10% 仓位
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

VW = 240            # 240 根 1m = 4h
CONF = 2880         # 低点确认 2880 根 = 48h
HOLD = 1440         # 大周期持仓 1440 根 = 24h
# 原版 gambler_merge_1m.py 口径：止盈=低点×1.30，止损=低点×0.95
# 相对入场价（低点×1.03）：止盈 +26.2%，止损 -7.8%
TPM, SLM = 0.30, 0.95

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_merge_1m_hist · 大周期跑整个历史')
log('=' * 100)
log(f'单位：v240=240根(4h)  低点确认={CONF}根(48h)  持仓={HOLD}根(24h)  成本={FEE*100:.2f}%/笔')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_502.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 3000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = np.full(n, np.inf)
    if n >= 121:
        w = swv(l, 121).min(axis=1)
        lo_w[60:n-60] = w
    lo_i = []
    i = 0
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 60
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
    """严格复刻 gambler_merge_1m.py 的口径
    入场=低点×1.03 那一刻；止盈=低点×1.30；止损=低点×0.95；同根双触算止损；24h 超时
    返回 feats, entries[(t, ret, is_tp)]
    """
    n = len(c)
    feats, entries = [], []
    hist_hits = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]
        lo_p = l[li]
        tp_px = lo_p * (1.0 + TPM)
        sl_px = lo_p * SLM
        ret = None; is_tp = 0
        for k in range(j + 1, min(j + HOLD + 1, n)):
            hit_tp = h[k] >= tp_px
            hit_sl = l[k] <= sl_px
            if hit_tp and hit_sl:
                ret = sl_px / e - 1; break
            if hit_tp:
                ret = tp_px / e - 1; is_tp = 1; break
            if hit_sl:
                ret = sl_px / e - 1; break
        if ret is None:
            ret = c[min(j + HOLD, n - 1)] / e - 1

        vw = v[j - VW:j + 1]
        v240 = float(np.std(np.log1p(np.maximum(vw, 0))))
        vm = float(np.mean(vw))
        volr = float(v[j] / vm) if vm > 0 else 1.0
        vb = np.sum(v[j - 14:j])
        tbr = float(np.sum(tb[j - 14:j]) / vb) if vb > 0 else 0.5
        r1 = e / c[j - 1] - 1 if j >= 1 else 0.0
        r15 = e / c[j - 15] - 1 if j >= 15 else 0.0
        r60 = e / c[j - 60] - 1 if j >= 60 else 0.0
        hr = pd.Timestamp(int(t[j]) // 1000, unit='s', tz='UTC').hour
        dw = pd.Timestamp(int(t[j]) // 1000, unit='s', tz='UTC').dayofweek
        pos = e / lo_p if lo_p > 0 else 1.0
        lo_age = j - li
        h3 = float(np.mean(hist_hits[-3:])) if hist_hits else -1.0
        h10 = float(np.mean(hist_hits[-10:])) if hist_hits else -1.0
        feats.append([volr, tbr, v240, pos, lo_age, r1, r15, r60, hr, dw, h3, h10])
        entries.append((int(t[j]), ret, is_tp))
        hist_hits.append(is_tp)
    return np.asarray(feats), entries


ALL = {}
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    X, E = build(pairs, o, h, l, c, t, v, tb)
    if len(E) == 0: continue
    ALL[sym] = (X, E)
log(f'币 {len(ALL)}  入场点 {sum(len(E) for _, E in ALL.values()):,}  耗时 {time.time()-t0:.0f}s')

TR, TE = {}, {}
for sym, (X, E) in ALL.items():
    m = len(E)
    if m < 4: continue
    hh = m // 2
    TR[sym] = (X[:hh], E[:hh])
    TE[sym] = (X[hh:], E[hh:])


def flatten(S):
    Xs, Es = [], []
    for sym, (X, E) in list(S.items()):
        if len(X) == 0: continue
        Xs.append(X); Es.extend(E)
    return (np.vstack(Xs) if Xs else np.zeros((0, 12)), Es)


Xtr, Etr = flatten(TR)
Xte, Ete = flatten(TE)
log(f'训练段 {len(Etr):,}   测试段 {len(Ete):,}')

from sklearn.ensemble import HistGradientBoostingClassifier
ytr = np.array([e[2] for e in Etr])
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(Xtr, ytr)
log(f'训练完成：训练段 TP 率 {ytr.mean()*100:.1f}%')


def sim(E, P, thr):
    trades = []
    for k, e in enumerate(E):
        if P[k] >= thr:
            trades.append((e[0], e[1] - FEE))
    if not trades: return 0, START, [], 0.0
    trades.sort(key=lambda x: x[0])
    cap = START; peak = START; dd = 0.0; curve = []
    for t, net in trades:
        cap *= (1 + POS * net)
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
        curve.append((t, cap))
    return len(trades), cap, curve, dd


from sklearn.metrics import roc_auc_score
Ptr = mdl.predict_proba(Xtr)[:, 1]
Pte = mdl.predict_proba(Xte)[:, 1]
ytr_l = np.array([e[2] for e in Etr])
yte_l = np.array([e[2] for e in Ete])
log(f'AUC：训练段(样本内) {roc_auc_score(ytr_l, Ptr):.4f}   测试段(样本外) {roc_auc_score(yte_l, Pte):.4f}')

n1, c1, cv1, dd1 = sim(Etr, Ptr, 0.5)
n2, c2, cv2, dd2 = sim(Ete, Pte, 0.5)
Eall = Etr + Ete
Pall = np.concatenate([Ptr, Pte])
n3, c3, cv3, dd3 = sim(Eall, Pall, 0.5)

log('')
log('=' * 100)
log('【大周期 · 全历史】')
log(f'  训练段（样本内）  {n1:>4} 笔  {START:.0f} → {c1:>15,.0f} 元  回撤 {dd1*100:>5.1f}%')
log(f'  测试段（样本外）  {n2:>4} 笔  {START:.0f} → {c2:>15,.0f} 元  回撤 {dd2*100:>5.1f}%')
log(f'  全历史合计       {n3:>4} 笔  {START:.0f} → {c3:>15,.0f} 元  回撤 {dd3*100:>5.1f}%')

log('')
log('【按月（全历史）】')
df = pd.DataFrame(cv3, columns=['t', 'cap'])
df['m'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
g = df.groupby('m')['cap'].agg(['last', 'count'])
log(f"{'月份':<10}{'笔数':>6}{'月末资金':>18}{'月收益%':>10}")
prev = START
for row in g.itertuples():
    m, last, cnt = row[0], row[1], row[2]
    log(f'{m:<10}{cnt:>6,}{last:>18,.0f}{(last/prev-1)*100:>9.1f}%')
    prev = last

log('')
log('=' * 100)
log('【判读】')
log('  · 训练段=样本内(模型见过)；测试段=样本外；全历史=两者合计')
log('  · 若测试段显著低于训练段 ⇒ 过拟合程度；若测试段亏 ⇒ 模型无效')
log('  · AUC 样本内 vs 样本外 的落差 = 过拟合的量化')

out = os.path.join(OUT, 'merge_1m_hist_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
