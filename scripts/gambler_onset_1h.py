# -*- coding: utf-8 -*-
"""gambler_onset_1h.py — 换时间尺度再问一次：1 小时线上，起点能不能看出来

为什么换尺度（前几轮全在 5m 上做，全部失败）：
    用户要的周期是"18~24 小时涨够 30%"，但我们在 5m 上看它——
    噪声（单根振幅中位 0.37%）把"节奏"淹没了。
    1h 线把 12 根合成一根，信噪比约提升 √12 ≈ 3.5 倍。

【做法】5m → 1h 重采样（open/high/low/close/成交额/笔数/taker买量 全聚合）
【样本】还没起涨（相对 288 根 1h 低点涨幅 < 10%）
【标签】未来 6 / 12 / 24 小时（1h 线上 = 6/12/24 根）内是否起局
【特征】amph、volrh、trdh、tbhr（1h 尺度）+ 位置 over + 历史节奏 cnt_30d + since_last + 市场
【纪律】时间留出 + 币A/币B 留出
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 24          # 1h 线：一天 24 根
W = 8              # 8 根 = 8 小时
TRIG = 0.30
H_LIST = [6, 12, 24]

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 118)
log('gambler_onset_1h · 1 小时尺度上的起点预报')
log('=' * 118)

# 读 5m 后重采样为 1h
frames = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low',
                               'quote_volume', 'trades', 'taker_quote'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32', 'taker_quote': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time').copy()
        g['h'] = g['open_time'] // 3_600_000
        agg = g.groupby('h').agg(close=('close', 'last'), high=('high', 'max'),
                                 low=('low', 'min'), q=('quote_volume', 'sum'),
                                 tr=('trades', 'sum'), tq=('taker_quote', 'sum'))
        frames.append((sym, agg))
    del ch; gc.collect()
log(f'重采样完成：{len(frames)} 币   耗时 {time.time()-t0:.0f}s')

C = pd.DataFrame({s: pd.Series(a['close'].values, index=a.index) for s, a in frames}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12 = C.pct_change(12).values
with np.errstate(invalid='ignore'):
    up_cnt = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(6, min_periods=1).max().values
del C, R12; gc.collect()
log(f'1h 矩阵 {T} × {N}   训练<=第 {t_cut} 根')

FN = ['over', 'amph', 'volrh', 'trdh', 'tbhr', 'cnt_30d', 'since_last', 'mkt_up', 'r6', 'pos']
XS, TID, SID, YS = [], [], [], {h: [] for h in H_LIST}

for (sym, a) in frames:
    n = len(a)
    if n < WDAY * 30 + 50:
        continue
    c = a['close'].values.astype('float64'); h = a['high'].values.astype('float64')
    l = a['low'].values.astype('float64'); q = a['q'].values.astype('float64')
    tr = a['tr'].values.astype('float64'); tq = a['tq'].values.astype('float64')
    lo288 = pd.Series(l).rolling(WDAY, min_periods=12).min().values
    hi288 = pd.Series(h).rolling(WDAY, min_periods=12).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
        pos = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    trig = np.nan_to_num(over, nan=-1.0) >= TRIG
    start = np.zeros(n, dtype='float32'); prev = False
    for i in range(n):
        if trig[i] and not prev:
            start[i] = 1.0
        prev = trig[i]
    cs = pd.Series(start).cumsum().values
    cnt30 = np.zeros(n)
    for i in range(WDAY * 30, n):
        cnt30[i] = cs[i] - cs[i - WDAY * 30]
    sidx = np.flatnonzero(start > 0)
    since = np.full(n, 99999.0)
    if len(sidx):
        last = np.full(n, -1)
        for s in sidx:
            last[s:] = s
        gap = np.arange(n) - last
        gap[last < 0] = 99999
        since = np.log1p(gap)
    amph = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=4).mean().values
    amphb = pd.Series(amph).rolling(WDAY * 30, min_periods=WDAY).mean().values
    qb = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    tb = pd.Series(tr).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        amph_r = amph / np.maximum(amphb, 1e-12)
        volr = q / np.maximum(qb, 1e-12)
        trd = tr / np.maximum(tb, 1e-12)
        tbr = tq / np.maximum(q, 1e-12)
    volrh = pd.Series(volr).rolling(6, min_periods=3).mean().values
    trdh = pd.Series(trd).rolling(6, min_periods=3).mean().values
    tbhr = pd.Series(tbr).rolling(6, min_periods=3).mean().values
    r6 = np.full(n, np.nan); r6[6:] = c[6:] / np.maximum(c[:-6], 1e-12) - 1.0

    posv = np.searchsorted(ts, a.index.values)
    Ys = {}
    for Hh in H_LIST:
        fut = pd.Series(start).rolling(Hh, min_periods=1).max().shift(-(Hh - 1)).values
        Ys[Hh] = np.nan_to_num(fut, nan=0.0).astype('int8')

    ok = (np.nan_to_num(over, nan=1.0) < 0.10) & np.isfinite(amph_r) & np.isfinite(volrh) & \
         np.isfinite(trdh) & (np.arange(n) > WDAY * 31) & (np.arange(n) < n - 25)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    cols = [over, amph_r, volrh, trdh, tbhr, cnt30, since, mkt_up[posv], r6, pos]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])
    XS.append(M[idx].astype('float32')); TID.append(posv[idx].astype('int64'))
    SID.append(np.full(len(idx), sym))
    for Hh in H_LIST:
        YS[Hh].append(Ys[Hh][idx])
del frames; gc.collect()

X = np.vstack(XS); TID = np.concatenate(TID); SID = np.concatenate(SID)
Y = {k: np.concatenate(v) for k, v in YS.items()}
tr = TID < t_cut; te = ~tr
coins = np.array(sorted(np.unique(SID)))
rng = np.random.default_rng(11)
half = set(rng.choice(coins, size=len(coins)//2, replace=False).tolist())
coinA = np.isin(SID, list(half))
log(f'样本（未起涨 over<10%）{len(X):,}   币 {len(coins)}   训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')

log('\n' + '=' * 118)
log('【结果】1h 尺度上的起点预报')
log('=' * 118)
for Hh in H_LIST:
    y = Y[Hh]
    if y[tr].sum() < 50:
        log(f'  H={Hh}h 正例不足（训练 {int(y[tr].sum())}）'); continue
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_depth=4,
                                         min_samples_leaf=60, random_state=7)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    log(f'\n  H = {Hh}h   训练正例率 {100*y[tr].mean():.3f}%   测试正例率 {100*y[te].mean():.3f}%')
    log(f'    AUC(时间留出) = {roc_auc_score(y[te], p):.4f}')
    for gname, gm in (('币A', coinA), ('币B', ~coinA)):
        m = te & gm
        if y[m].sum() > 5:
            pp = clf.predict_proba(X[m])[:, 1]
            log(f'    AUC({gname}留出) = {roc_auc_score(y[m], pp):.4f}')
    base = y[te].mean()
    log(f'    {"阈值":>6}{"报出":>9}{"覆盖":>9}{"精确率":>9}{"召回":>8}{"提升":>8}')
    for thr in (0.1, 0.2, 0.3, 0.5):
        m = p >= thr
        if m.sum() == 0:
            log(f'    {thr:>6.1f}{"无":>9}'); continue
        prec = y[te][m].mean()
        log(f'    {thr:>6.1f}{int(m.sum()):>9,}{100*m.mean():>8.2f}%{100*prec:>8.2f}%'
            f'{100*m[y[te] == 1].sum()/max((y[te] == 1).sum(),1):>7.2f}%{prec/max(base,1e-9):>7.1f}x')

log('\n' + '=' * 118)
log('【单变量】1h 尺度各特征（H=12h 测试段 AUC）')
log('=' * 118)
y = Y[12]
for j, nm in enumerate(FN):
    x = X[te, j]
    if np.std(x) == 0:
        continue
    log(f'  {nm:<12}{roc_auc_score(y[te], x):>8.3f}')

out = os.path.join(OUT, 'onset_1h_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
