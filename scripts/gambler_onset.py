# -*- coding: utf-8 -*-
"""gambler_onset.py — 只为回答一个问题：周期「起点的起点」能不能认出来

用户判断（2026-10-17）：
    「不惜一切代价，找到认定周期刚开始或开始之前的规律。」

【本脚本与前几次的关键区别】
    前几次（价格特征 0.34 / 导数特征 0.524 / 振幅 0.926）都栽在同一个坑：
    **特征算在"已经动了"的时点**，标签是"未来还会不会涨"——本质是动量。
    这次强制：**样本只取「还没起涨」的时点（相对 288 根低点涨幅 < 10%）**，
    问的是"未来 6/12/24 小时内，它会不会进入一个 +30% 周期"。

【特征：这次加了此前从未用过的三类"前置结构"】
    A 历史节奏
        since_last   距上一次起局的小时数（log1p）      ← 此前实测 lift 220 倍的最强者
        cnt_30d      过去 30 天起局次数
    B 参与人数（不是价格）
        trades_ratio 成交笔数 / 30 日均值
        ats_ratio    单笔均额 / 30 日中位（大单占比）
        tbr24        主动买占比 24 根均值
    C 市场结构
        mkt_up/mkt_dn 全市场同刻涨/跌 >5% 的币数
        btc_r24      BTC 24h 涨幅
        disp         全市场横截面离散度（分化程度）
    D 已确认有效但样本外失效的（作为对照）
        amp_ratio    8h 振幅 / 30 日均值

【纪律】前 70% 时间训练 / 后 30% 测试；再做一次币留出；只报测试段。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; TRIG = 0.30
H_LIST = [6, 12, 24]

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 120)
log('gambler_onset · 只问一件事：还没起涨的时候，能不能预判它要起')
log('=' * 120)

RAW = []
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
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64'), g['taker_quote'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns); ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12 = C.pct_change(12).values
with np.errstate(invalid='ignore'):
    up_cnt = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
    dn_cnt = (np.nan_to_num(R12, nan=0.0) <= -0.05).sum(axis=1)
    disp = np.nanstd(R12, axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
mkt_dn = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r24 = np.zeros(T); btc_r24[288:] = btc[288:] / np.maximum(btc[:-288], 1e-12) - 1.0
del C, R12; gc.collect()
log(f'币 {N}   全局 {T}   训练<=第 {t_cut} 根   耗时 {time.time()-t0:.0f}s')

FN = ['since_last', 'cnt_30d', 'trades_ratio', 'ats_ratio', 'tbr24',
      'mkt_up', 'mkt_dn', 'btc_r24', 'disp', 'amp_ratio', 'over', 'volr24']
XS, TID, OVS, SIDS = [], [], [], []
YS = {h: [] for h in H_LIST}

for (sym, ot, c, h, l, q, tr, tq) in RAW:
    n = len(c)
    if n < WDAY * 4:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
    trig = np.nan_to_num(over, nan=-1.0) >= TRIG
    start = np.zeros(n, dtype='float32')
    prev = False
    for i in range(n):
        if trig[i] and not prev:
            start[i] = 1.0
        prev = trig[i]
    sidx = np.flatnonzero(start > 0)

    # A 历史节奏
    since_last = np.full(n, np.nan)
    cnt30 = np.zeros(n)
    if len(sidx) > 0:
        last = np.full(n, -1)
        for s in sidx:
            last[s:] = s
        gap = (np.arange(n) - last)
        gap[last < 0] = 99999
        since_last = np.log1p(gap * 5 / 60.0)            # 小时
        cs = pd.Series(start).cumsum().values
        for i in range(WDAY * 30, n):
            cnt30[i] = cs[i] - cs[i - WDAY * 30]

    # B 参与人数
    mt = pd.Series(tr).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        trades_ratio = tr / np.maximum(mt, 1e-12)
        ats = q / np.maximum(tr, 1e-12)
        ats_m = pd.Series(ats).rolling(WDAY * 30, min_periods=WDAY).median().values
        ats_ratio = ats / np.maximum(ats_m, 1e-12)
        tbr = tq / np.maximum(q, 1e-12)
    tbr24 = pd.Series(tbr).rolling(24, min_periods=6).mean().values
    # D 振幅（对照）
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    ampb = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_ratio = amp / np.maximum(ampb, 1e-12)
        volr = q / np.maximum(pd.Series(q).rolling(WDAY*30, min_periods=WDAY).mean().values, 1e-12)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values

    posv = np.searchsorted(ts, ot)
    # 未来 H 小时是否起局
    Ys = {}
    for Hh in H_LIST:
        k = int(Hh * 12)
        fut = pd.Series(start).rolling(k, min_periods=1).max().shift(-(k - 1)).values
        Ys[Hh] = np.nan_to_num(fut, nan=0.0).astype('int8')

    # ★ 样本：还没起涨（over < 10%）
    ok = (np.nan_to_num(over, nan=1.0) < 0.10) & np.isfinite(since_last) & \
         np.isfinite(trades_ratio) & np.isfinite(amp_ratio) & (np.arange(n) < n - 24*12 - 2) & \
         (np.arange(n) > WDAY * 31)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:
        idx = idx[::max(1, len(idx)//4000)]
    cols = [since_last, cnt30, trades_ratio, ats_ratio, tbr24,
            mkt_up[posv], mkt_dn[posv], btc_r24[posv], disp[posv], amp_ratio, over, volr24]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])
    XS.append(M[idx].astype('float32'))
    TID.append(posv[idx].astype('int64'))
    OVS.append(over[idx].astype('float32'))
    SIDS.append(np.full(len(idx), sym))
    for Hh in H_LIST:
        YS[Hh].append(Ys[Hh][idx])
del RAW; gc.collect()

X = np.vstack(XS); TID = np.concatenate(TID); OV = np.concatenate(OVS); SID = np.concatenate(SIDS)
Y = {k: np.concatenate(v) for k, v in YS.items()}
tr = TID < t_cut; te = ~tr
coins = np.array(sorted(np.unique(SID)))
rng = np.random.default_rng(11)
half = set(rng.choice(coins, size=len(coins)//2, replace=False).tolist())
coinA = np.isin(SID, list(half))
log(f'样本（未起涨 over<10%）{len(X):,}   币 {len(coins)}   训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')

log('\n' + '=' * 120)
log('【结果】还没起涨时的预判能力')
log('=' * 120)
for Hh in H_LIST:
    y = Y[Hh]
    if y[tr].sum() < 50:
        log(f'  H={Hh}h 正例不足（训练 {int(y[tr].sum())}）'); continue
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_depth=4,
                                         min_samples_leaf=80, random_state=7)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    pA = clf.predict_proba(X[te & coinA])[:, 1]
    yA = y[te & coinA]
    pB = clf.predict_proba(X[te & ~coinA])[:, 1]
    yB = y[te & ~coinA]
    log(f'\n  H = {Hh}h   训练正例率 {100*y[tr].mean():.3f}%   测试正例率 {100*y[te].mean():.3f}%')
    log(f'    AUC(时间留出) = {roc_auc_score(y[te], p):.4f}')
    if yA.sum() > 5 and yB.sum() > 5:
        log(f'    AUC(币A留出)  = {roc_auc_score(yA, pA):.4f}   AUC(币B留出) = {roc_auc_score(yB, pB):.4f}')
    log(f'    {"阈值":>6}{"报出":>9}{"覆盖":>9}{"精确率":>9}{"召回":>8}{"提升倍数":>10}')
    base = y[te].mean()
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        m = p >= thr
        if m.sum() == 0:
            log(f'    {thr:>6.1f}{"无":>9}'); continue
        prec = y[te][m].mean()
        log(f'    {thr:>6.1f}{int(m.sum()):>9,}{100*m.mean():>8.2f}%{100*prec:>8.2f}%'
            f'{100*m[y[te] == 1].sum()/max((y[te] == 1).sum(),1):>7.2f}%'
            f'{prec/max(base,1e-9):>9.1f}x')

# 单变量 AUC
log('\n' + '=' * 120)
log('【单变量】H=12h，各特征单独用（测试段 AUC）')
log('=' * 120)
y = Y[12]
for j, nm in enumerate(FN):
    x = X[te, j]
    if np.std(x) == 0:
        continue
    log(f'  {nm:<14}{roc_auc_score(y[te], x):>8.3f}')

out = os.path.join(OUT, 'onset_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
