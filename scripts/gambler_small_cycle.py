# -*- coding: utf-8 -*-
"""gambler_small_cycle · 小周期探索（用动量 h3 筛选）

与大周期同一套入场逻辑（低点后涨 3%），但：
  止盈 +5%（相对入场）/ 止损 -2% / 持有 60 根(1h)
筛选对比：h3>=1.0（最近 3 次同口径 TP）/ 模型 / 不筛
数据：klines_1m_1y.csv；按日历切分前 75% 训练 / 后 25% 测试
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
HOLD = 60
TPM, SLM = 0.05, 0.98

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_small_cycle · 小周期（+5% / -2% / 1h）+ 动量筛选')
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


def build_small(pairs, o, h, l, c, t, v, tb):
    n = len(c)
    feats, entries = [], []
    hist_hits = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]
        tp_px = e * (1.0 + TPM)
        sl_px = e * SLM
        seg_h = h[j + 1:j + 1 + HOLD]
        seg_l = l[j + 1:j + 1 + HOLD]
        hit_tp = seg_h >= tp_px
        hit_sl = seg_l <= sl_px
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
        if i_tp >= 0 and i_sl >= 0:
            if i_tp < i_sl:
                ret = TPM; is_tp = 1
            else:
                ret = -(1 - SLM); is_tp = 0
        elif i_tp >= 0:
            ret = TPM; is_tp = 1
        elif i_sl >= 0:
            ret = -(1 - SLM); is_tp = 0
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
        pos = e / l[li] if l[li] > 0 else 1.0
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
    X, E = build_small(pairs, o, h, l, c, t, v, tb)
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
log(f'训练 {len(tr):,}（TP {df["tp"].values[tr].mean()*100:.1f}%）  测试 {len(te):,}（TP {df["tp"].values[te].mean()*100:.1f}%）')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Pte = mdl.predict_proba(X[te])[:, 1]
log(f'AUC 样本外 {roc_auc_score(df["tp"].values[te], Pte):.4f}')

h3 = X[te][:, 10]
RULES = {
    '不筛（全部）': np.ones(len(te), dtype=bool),
    'h3>=1.0': h3 >= 1.0,
    'h3>=0.67': h3 >= 0.67,
    '模型 P>=0.5': Pte >= 0.5,
}


def sim(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    rs = df['ret'].values[te][idx] - FEE
    ts = df['t'].values[te][idx]
    order = np.argsort(ts); rs = rs[order]
    cap = START; peak = START; dd = 0.0
    for r in rs:
        cap *= (1 + POS * r)
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
    return dict(n=len(rs), mean=rs.mean() * 100, med=np.median(rs) * 100,
                win=(rs > 0).mean() * 100, cap=cap, dd=dd * 100)


log('')
log('=' * 100)
log('【小周期 +5%/-2%/1h · 测试段】')
log(f"{'规则':<18}{'n':>8}{'均值%':>9}{'中位%':>9}{'胜率%':>8}{'50元→':>14}{'回撤%':>8}")
for name, mask in RULES.items():
    s = sim(mask)
    if s is None:
        log(f'{name:<18}{0:>8}')
        continue
    log(f"{name:<18}{s['n']:>8,}{s['mean']:>9.2f}{s['med']:>9.2f}{s['win']:>8.1f}{s['cap']:>14,.0f}{s['dd']:>8.1f}")

log('')
log('【h3 分档（测试段）】')
log(f"{'h3':>8}{'n':>10}{'TP率%':>10}{'均值%':>10}")
for v in np.unique(h3):
    m = h3 == v
    if m.sum() < 50: continue
    rs = df['ret'].values[te][m]
    log(f'{v:>8.2f}{int(m.sum()):>10,}{df["tp"].values[te][m].mean()*100:>10.1f}{rs.mean()*100:>10.2f}')

log('')
log('=' * 100)
log('【判读】')
log('  · 小周期的盈亏平衡胜率：赢 5%-0.5%=4.5%，输 2%+0.5%=2.5% → p* = 35.7%')
log('  · 若 h3>=1.0 的胜率 > 50% 且均值 > 0 ⇒ 动量在小周期也成立')
log('  · 若全不筛均值仍负、h3 筛选也救不回 ⇒ 小周期方向确实不成立')

out = os.path.join(OUT, 'small_cycle_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
