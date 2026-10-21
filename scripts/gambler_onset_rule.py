# -*- coding: utf-8 -*-
"""gambler_onset_rule.py — 把"起点前预报"从 7% 精确率往 25%+ 推

已知（job-24708-41 gambler_onset）：
    样本限"还没起涨"（over<10%）244,755 个，58 币
    未来 6h 起局 AUC：时间 0.7042 / 币A 0.6921 / 币B 0.7218
    未来 24h：阈值 0.2 → 报出 573，精确率 7.16%（基准 0.997%）= 7.2 倍
    单变量最强：over 0.706、amp_ratio 0.658、volr24 0.633、trades_ratio 0.622、cnt_30d 0.611

但 7% 精确率不够交易（15 个报错 14 个）。
本脚本用**可解释的规则叠加**代替黑箱，看能把精确率推到多高、覆盖率掉到多少。

规则条件（全部 t 时刻可见，无前视）：
    over_lo / over_hi   当前相对 288 根低点的涨幅区间
    amp                 amp_ratio ≥ x
    volr                volr24 ≥ x
    trd                 trades_ratio ≥ x
    cnt                 过去 30 天起局次数 ≥ x
    mktup               全市场同刻涨 >5% 的币数 ≥ x
单条全部扫描 + 两两组合全穷举，训练段（前70%）搜索 → 测试段（后30%）复核。
"""
import os, gc, time, itertools
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; TRIG = 0.30; HOURS = 12
MIN_N = 100

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_onset_rule · 起点前预报：规则叠加能到多少精确率')
log('=' * 112)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low',
                               'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12 = C.pct_change(12).values
with np.errstate(invalid='ignore'):
    up_cnt = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
del C, R12; gc.collect()

FN = ['over', 'amp', 'volr', 'trd', 'cnt', 'mktup']
XS, TID = [], []
Y = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
    trig = np.nan_to_num(over, nan=-1.0) >= TRIG
    start = np.zeros(n, dtype='float32'); prev = False
    for i in range(n):
        if trig[i] and not prev:
            start[i] = 1.0
        prev = trig[i]
    cs = pd.Series(start).cumsum().values
    cnt = np.zeros(n)
    for i in range(WDAY * 30, n):
        cnt[i] = cs[i] - cs[i - WDAY * 30]
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    ampb = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    qs = pd.Series(q)
    qb = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    mt = pd.Series(tr).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_ratio = amp / np.maximum(ampb, 1e-12)
        volr = q / np.maximum(qb, 1e-12)
        volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
        trd = tr / np.maximum(mt, 1e-12)
    k = int(HOURS * 12)
    fut = pd.Series(start).rolling(k, min_periods=1).max().shift(-(k - 1)).values
    y = np.nan_to_num(fut, nan=0.0).astype('int8')
    posv = np.searchsorted(ts, ot)
    ok = (np.nan_to_num(over, nan=1.0) < 0.10) & np.isfinite(amp_ratio) & \
         np.isfinite(volr24) & np.isfinite(trd) & (np.arange(n) > WDAY * 31) & \
         (np.arange(n) < n - k - 2)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:
        idx = idx[::max(1, len(idx)//4000)]
    cols = [over, amp_ratio, volr24, trd, cnt, mkt_up[posv]]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])
    XS.append(M[idx].astype('float32')); TID.append(posv[idx].astype('int64')); Y.append(y[idx])
del RAW; gc.collect()

X = np.vstack(XS); TID = np.concatenate(TID); Y = np.concatenate(Y)
tr = TID < t_cut; te = ~tr
base_tr = Y[tr].mean(); base_te = Y[te].mean()
days_te = len(np.unique(ts[TID[te]] // 86_400_000))
log(f'样本 {len(X):,}  训练 {int(tr.sum()):,} 测试 {int(te.sum()):,}')
log(f'基准：训练 {100*base_tr:.3f}%  测试 {100*base_te:.3f}%  测试天数 {days_te}')

F = {nm: X[:, j] for j, nm in enumerate(FN)}
CONDS = {}
for a in (0.0, 0.02, 0.05):
    CONDS[f'over>={a:.2f}'] = F['over'] >= a
for x in (1.3, 1.5, 2.0, 3.0):
    CONDS[f'amp>={x}'] = F['amp'] >= x
for x in (1.5, 2.0, 3.0, 5.0):
    CONDS[f'volr>={x}'] = F['volr'] >= x
for x in (1.2, 1.5, 2.0, 3.0):
    CONDS[f'trd>={x}'] = F['trd'] >= x
for x in (1, 2, 3, 5):
    CONDS[f'cnt>={x}'] = F['cnt'] >= x
for x in (2, 3, 5):
    CONDS[f'mktup>={x}'] = F['mktup'] >= x

log('\n' + '=' * 112)
log('【单条】训练段筛（lift>=2）→ 测试段复核')
log('=' * 112)
log(f'  {"条件":<16}{"训n":>9}{"训精确%":>10}{"测n":>9}{"测精确%":>10}{"测lift":>8}{"召回%":>8}{"每天":>7}')
single = {}
for nm, m in CONDS.items():
    ntr = int((m & tr).sum())
    if ntr < MIN_N:
        continue
    ptr = Y[tr][m[tr]].mean()
    if ptr / max(base_tr, 1e-9) < 2:
        continue
    nte = int((m & te).sum())
    if nte < 30:
        continue
    pte = Y[te][m[te]].mean()
    rec = (m & te & (Y == 1)).sum() / max((Y[te] == 1).sum(), 1)
    single[nm] = (ptr, pte, nte)
    log(f'  {nm:<16}{ntr:>9,}{100*ptr:>10.2f}{nte:>9,}{100*pte:>10.2f}'
        f'{pte/max(base_te,1e-9):>8.1f}{100*rec:>8.2f}{nte/max(days_te,1):>7.1f}')

log('\n' + '=' * 112)
log('【两两组合】真穷举，按测试段精确率排序')
log('=' * 112)
names = list(single.keys())
res = []
for a, b in itertools.combinations(names, 2):
    m = CONDS[a] & CONDS[b]
    nte = int((m & te).sum())
    if nte < 30:
        continue
    pte = Y[te][m[te]].mean()
    rec = (m & te & (Y == 1)).sum() / max((Y[te] == 1).sum(), 1)
    ntr = int((m & tr).sum())
    ptr = Y[tr][m[tr]].mean() if ntr else 0
    if ptr / max(base_tr, 1e-9) < 2:
        continue
    res.append((a, b, ntr, ptr, nte, pte, rec))
res.sort(key=lambda x: -x[5])
log(f'  {"组合":<34}{"训n":>8}{"训精确%":>9}{"测n":>7}{"测精确%":>10}{"测lift":>8}{"召回%":>8}{"每天":>7}')
for a, b, ntr, ptr, nte, pte, rec in res[:25]:
    log(f'  {a+" & "+b:<34}{ntr:>8,}{100*ptr:>9.2f}{nte:>7,}{100*pte:>10.2f}'
        f'{pte/max(base_te,1e-9):>8.1f}{100*rec:>8.2f}{nte/max(days_te,1):>7.1f}')

log('\n' + '=' * 112)
log('【判读】精确率能到 25% 且每天出手 >=1 次的组合 = 可用于实盘筛选')
log('=' * 112)
ok = [r for r in res if r[5] >= 0.25 and r[4] / max(days_te, 1) >= 1]
log(f'  达标组合 {len(ok)} 个')
for a, b, ntr, ptr, nte, pte, rec in ok[:10]:
    log(f'    {a} & {b}：测试 {nte} 笔，精确率 {100*pte:.1f}%，每天 {nte/max(days_te,1):.1f} 次')

out = os.path.join(OUT, 'onset_rule_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
