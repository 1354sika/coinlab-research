# -*- coding: utf-8 -*-
"""gambler_momentum_baseline · 纯动量规则 vs 模型

不用模型，只按"本币最近 3/10 次入场是否 TP"直接下注，
看是否与模型的 92% 胜率差不多 —— 若差不多，模型只是动量跟随着，不是预测器。

数据：klines_1m_1y.csv；按日历切分前 9 月训练 / 后 4 月测试
口径：入场=低点×1.03，止盈=低点×1.30，止损=低点×0.95，24h 超时；成本 0.50%/笔
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
FEATS = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hr','dw','h3','h10']

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_momentum_baseline · 纯动量规则 vs 模型')
log('=' * 100)

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
ntr = max(1, int(len(months) * 0.75))
train_m = months[:ntr]; test_m = months[ntr:]
tr = np.where(df['ym'].isin(train_m).values)[0]
te = np.where(df['ym'].isin(test_m).values)[0]
log(f'训练 {len(tr):,}  测试 {len(te):,}  训练月 {train_m[0]}~{train_m[-1]}  测试月 {test_m[0]}~{test_m[-1]}')

mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Pte = mdl.predict_proba(X[te])[:, 1]

h3 = X[te][:, 10]
h10 = X[te][:, 11]

RULES = {
    '模型 P>=0.5': Pte >= 0.5,
    'h3>=1.0（最近3次全TP）': h3 >= 1.0,
    'h3>=0.67（3次中2次TP）': h3 >= 0.67,
    'h3>=0.5': h3 >= 0.5,
    'h3>=1.0 & h10>=0.5': (h3 >= 1.0) & (h10 >= 0.5),
    'h10>=0.5': h10 >= 0.5,
    'h10>=0.7': h10 >= 0.7,
    '全部（不筛）': np.ones(len(te), dtype=bool),
}


def sim(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    rs = df['ret'].values[te][idx] - FEE
    ts = df['t'].values[te][idx]
    order = np.argsort(ts)
    rs = rs[order]
    cap = START; peak = START; dd = 0.0
    for r in rs:
        cap *= (1 + POS * r)
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
    return dict(n=len(rs), mean=rs.mean() * 100, med=np.median(rs) * 100,
                win=(rs > 0).mean() * 100, cap=cap, dd=dd * 100)


log('')
log('=' * 100)
log('【测试段（2026-07~10，样本外）· 纯动量规则 vs 模型】')
log(f"{'规则':<26}{'n':>7}{'均值%':>9}{'中位%':>9}{'胜率%':>8}{'50元→':>16}{'回撤%':>8}")
for name, mask in RULES.items():
    s = sim(mask)
    if s is None:
        log(f'{name:<26}{0:>7}')
        continue
    log(f"{name:<26}{s['n']:>7,}{s['mean']:>9.2f}{s['med']:>9.2f}{s['win']:>8.1f}{s['cap']:>16,.0f}{s['dd']:>8.1f}")

# 分档：h3 不同取值下的 TP 率（测试段全样本）
log('')
log('=' * 100)
log('【h3 分档（测试段全部入场点）· h3 本身对 TP 的预测力】')
vals = np.unique(h3)
log(f"{'h3':>8}{'n':>10}{'TP率%':>10}{'均值%':>10}")
for v in vals:
    m = h3 == v
    if m.sum() < 50: continue
    rs = df['ret'].values[te][m]
    log(f'{v:>8.2f}{int(m.sum()):>10,}{df["tp"].values[te][m].mean()*100:>10.1f}{rs.mean()*100:>10.2f}')

log('')
log('=' * 100)
log('【判读】')
log('  · 若「h3>=1.0」与「模型 P>=0.5」的胜率/均值接近 ⇒ 模型只是在追动量，不是预测器')
log('  · 若模型显著优于任何 h3 规则 ⇒ 模型有超出动量的增量信息')
log('  · 若「全部不筛」也接近 ⇒ 连动量都没用上，只是 RP')

out = os.path.join(OUT, 'momentum_baseline_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
