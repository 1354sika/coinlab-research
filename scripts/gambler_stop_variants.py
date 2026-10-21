# -*- coding: utf-8 -*-
"""gambler_stop_variants · 止损口径对比（全历史，1m）

A 原版：止损 = 低点 × 0.95（固定，相对入场 -7.8%）
B 你的想法：止损 = 入场后出现过的最高价 × 0.95（移动/trailing）
C 另一种理解：止损 = 入场前 1440 根（24h）最高价 × 0.95

止盈统一：低点 × 1.30（相对入场 +26.2%）
双触（同根内 TP 和 SL 都触）：按止损算（保守）
两种样本：全部入场点 / P>=0.5 筛后
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
TPM = 0.30

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_stop_variants · 止损口径对比（全历史，1m）')
log('=' * 100)
log(f'止盈统一：低点×{1+TPM:.2f}（相对入场 +{(1+TPM)/1.03-1:+.1%}）')
log('A 原版：止损=低点×0.95（相对入场 -7.8%）')
log('B 移动：止损=入场后最高价×0.95')
log('C 入场前：止损=入场前24h最高价×0.95')
log('双触（同根内 TP 和 SL 都触）→ 按止损算（保守）')

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
log(f'币 {len(RAWS)}  读 {time.time()-t0:.0f}s')


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


FEATS = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hr','dw','h3','h10']

def feats_of(li, j, o, h, l, c, t, v, tb, hist_hits):
    e = c[j]; lo_p = l[li]
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
    return [volr, tbr, v240, pos, lo_age, r1, r15, r60, hr, dw, h3, h10]


def sim_three(li, j, e, lo_p, h, l, c, n):
    """返回 (retA, retB, retC, is_tp)"""
    tp_px = lo_p * (1.0 + TPM)
    slA = lo_p * 0.95
    slC = float(np.max(h[max(0, j - 1439):j + 1])) * 0.95
    peak = e
    rA = rB = rC = None
    is_tp = 0
    hi = min(j + HOLD + 1, n)
    for k in range(j + 1, hi):
        hit_tp = h[k] >= tp_px
        # A
        if rA is None:
            hitA = l[k] <= slA
            if hit_tp and hitA: rA = slA / e - 1
            elif hit_tp: rA = tp_px / e - 1
            elif hitA: rA = slA / e - 1
        # C
        if rC is None:
            hitC = l[k] <= slC
            if hit_tp and hitC: rC = slC / e - 1
            elif hit_tp: rC = tp_px / e - 1
            elif hitC: rC = slC / e - 1
        # B：先更新峰值再判
        peak = max(peak, h[k])
        slB = peak * 0.95
        if rB is None:
            hitB = l[k] <= slB
            if hit_tp and hitB: rB = slB / e - 1
            elif hit_tp: rB = tp_px / e - 1
            elif hitB: rB = slB / e - 1
        if rA is not None and rB is not None and rC is not None:
            break
    kk = min(j + HOLD, n - 1)
    tail = c[kk] / e - 1
    if rA is None: rA = tail
    if rB is None: rB = tail
    if rC is None: rC = tail
    # 是否 TP（仅 A 口径，用于标签）
    is_tp = 1 if rA == tp_px / e - 1 else 0
    return rA, rB, rC, is_tp


ROWS = []   # (sym, t, retA, retB, retC, is_tp, X)
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    hist_hits = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]; lo_p = l[li]
        rA, rB, rC, is_tp = sim_three(li, j, e, lo_p, h, l, c, n)
        X = feats_of(li, j, o, h, l, c, t, v, tb, hist_hits)
        ROWS.append((sym, int(t[j]), rA, rB, rC, is_tp, X))
        hist_hits.append(is_tp)
log(f'入场点 {len(ROWS):,}  耗时 {time.time()-t0:.0f}s')

# 训练/测试按时间切半
df = pd.DataFrame({'sym': [r[0] for r in ROWS], 't': [r[1] for r in ROWS],
                   'A': [r[2] for r in ROWS], 'B': [r[3] for r in ROWS],
                   'C': [r[4] for r in ROWS], 'tp': [r[5] for r in ROWS]})
X = np.array([r[6] for r in ROWS], dtype=float)
df['i'] = np.arange(len(df))
df = df.sort_values('t').reset_index(drop=True)
X = X[df['i'].values]
half = len(df) // 2
tr = np.arange(half); te = np.arange(half, len(df))
log(f'训练 {len(tr):,}   测试 {len(te):,}   训练 TP 率 {df["tp"].values[tr].mean()*100:.1f}%')

from sklearn.ensemble import HistGradientBoostingClassifier
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
P = mdl.predict_proba(X)[:, 1]
log(f'AUC 训练 {roc_auc_score(df["tp"].values[tr], P[tr]):.4f}  测试 {roc_auc_score(df["tp"].values[te], P[te]):.4f}' if False else f'模型已训练')


def stats(rs):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return None
    cap = START; peak = START; dd = 0.0
    for r in rs:
        cap *= (1 + POS * (r - FEE))
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
    return dict(n=len(rs), mean=rs.mean()*100, med=np.median(rs)*100,
                win=(rs > 0).mean()*100, cap=cap, dd=dd*100, worst=rs.min()*100)


def show(title, idx):
    log('')
    log('=' * 100)
    log(f'【{title}】')
    log(f"{'口径':<26}{'n':>8}{'均值%':>9}{'中位%':>9}{'胜率%':>8}{'10%仓位终值':>18}{'回撤%':>8}{'最差%':>9}")
    for col, lab in [('A', 'A 低点×0.95（原版）'), ('B', 'B 入场后最高×0.95（移动）'), ('C', 'C 入场前24h最高×0.95')]:
        s = stats(df[col].values[idx] if idx is not None else df[col].values)
        if s is None: continue
        log(f"{lab:<26}{s['n']:>8,}{s['mean']:>9.2f}{s['med']:>9.2f}{s['win']:>8.1f}{s['cap']:>18,.0f}{s['dd']:>8.1f}{s['worst']:>9.1f}")


show('全历史 · 全部入场点（不筛模型）', None)
sel = np.where(P >= 0.5)[0]
show(f'全历史 · P>=0.5 筛后（{len(sel)} 笔）', sel)
sel_te = np.where((P >= 0.5))[0]
sel_te = sel_te[sel_te >= half]
show(f'测试段 · P>=0.5（{len(sel_te)} 笔）', sel_te)

log('')
log('=' * 100)
log('【判读】')
log('  · A vs B：B 是移动止损，若均值更高且回撤更低 ⇒ 你的想法更优')
log('  · C 是"入场前最高点"锚定，若远差于 A/B ⇒ 说明锚定过去高点不适合')
log('  · 双触一律按止损算，三种口径同一规则')

out = os.path.join(OUT, 'stop_variants_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
