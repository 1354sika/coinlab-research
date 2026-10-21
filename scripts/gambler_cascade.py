# -*- coding: utf-8 -*-
"""gambler_cascade.py — 把 onset 分数当「一环」用：级联筛选能不能把薄利变厚

用户判断（2026-10-17）：
    「就算只能做筛选器也是有正向效应的，起码可以作为有用的变量，
      作为我们还没发现的规律的一环。」

这句话指出了一个我们一直做错的地方：
    我们总想把「起点预测」直接当买入信号（结果全负），
    而它正确的用法可能是**级联的第一层**——把候选池从"全市场"缩到几个，
    再叠加第二层条件（更严格、更慢）。

【本脚本测两件事】
   ① 单独用：onset 分数单独筛选 + 固定退出，期望净是多少（复现已知的负值）
   ② 级联用：onset 分数 × 已验证的薄利规则 R1（市场齐涨≥2 & 24h涨≥5%）
      看期望能不能从 +0.91% 抬到更高

【纪律】onset 模型只用训练段（前70%）训练；测试段（后30%）打分；
        所有条件 t 时刻可见；扣费 0.09%。
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
W = 96; WDAY = 288; TRIG = 0.30; FEE = 0.0009
HOLD = 576          # 48h
TP = 0.30; SL = 0.15

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_cascade · onset 分数作为「一环」：级联筛选')
log('=' * 116)

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
log(f'币 {N}   全局 {T}   耗时 {time.time()-t0:.0f}s')

FN = ['over', 'amp', 'volr', 'trd', 'cnt', 'since']
XS, TID, YS, SID = [], [], [], []
R1F = []   # R1 条件（市场齐涨≥2 & 24h涨≥5%）
RES = []   # 每行的交易结果（+30/-15/48h）

for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4 + HOLD + 2:
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
    for i in range(WDAY*30, n):
        cnt[i] = cs[i] - cs[i-WDAY*30]
    sidx = np.flatnonzero(start > 0)
    since = np.full(n, 99999.0)
    if len(sidx):
        last = np.full(n, -1)
        for s in sidx:
            last[s:] = s
        gap = np.arange(n) - last
        gap[last < 0] = 99999
        since = np.log1p(gap * 5 / 60.0)
    amp = pd.Series((h-l)/np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    ampb = pd.Series(amp).rolling(WDAY*30, min_periods=WDAY).mean().values
    qb = pd.Series(q).rolling(WDAY*30, min_periods=WDAY).mean().values
    tb = pd.Series(tr).rolling(WDAY*30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_r = amp/np.maximum(ampb, 1e-12)
        volr24 = pd.Series(q/np.maximum(qb, 1e-12)).rolling(24, min_periods=6).mean().values
        trd = tr/np.maximum(tb, 1e-12)
    r288 = np.full(n, np.nan); r288[288:] = c[288:]/np.maximum(c[:-288], 1e-12) - 1.0
    posv = np.searchsorted(ts, ot)
    # 标签：未来 12h 起局
    fut = pd.Series(start).rolling(144, min_periods=1).max().shift(-143).values
    ystart = np.nan_to_num(fut, nan=0.0).astype('int8')

    # 交易结果（+30/-15/48h）
    wh = np.lib.stride_tricks.sliding_window_view(h, HOLD)
    wl = np.lib.stride_tricks.sliding_window_view(l, HOLD)
    wc = np.lib.stride_tricks.sliding_window_view(c, HOLD)
    m = wh.shape[0]
    c0 = c[:m]
    hit_tp = (wh >= (c0*(1+TP))[:, None])
    hit_sl = (wl <= (c0*(1-SL))[:, None])
    it = np.argmax(hit_tp, 1); il = np.argmax(hit_sl, 1)
    any_tp = hit_tp.any(1); any_sl = hit_sl.any(1)
    v = np.where(any_sl & (~any_tp | (il <= it)), -SL,
                 np.where(any_tp, TP, wc[:, -1]/c0 - 1.0)) - FEE
    del wh, wl, wc

    ok = (np.nan_to_num(over, nan=1.0) < 0.10) & np.isfinite(amp_r) & \
         np.isfinite(volr24) & (np.arange(n) > WDAY*31) & (np.arange(n) < m)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:
        idx = idx[::max(1, len(idx)//4000)]
    cols = [over, amp_r, volr24, trd, cnt, since]
    XS.append(np.column_stack([np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                               for x in cols])[idx].astype('float32'))
    TID.append(posv[idx].astype('int64')); YS.append(ystart[idx])
    SID.append(np.full(len(idx), sym)); RES.append(v[idx].astype('float32'))
    # 另外记录 R1 条件（只用于测试段）
    R1c = (mkt_up[posv] >= 2) & (np.nan_to_num(r288, nan=-1.0) >= 0.05)
    R1F.append(R1c[idx].astype('int8'))
del RAW; gc.collect()

X = np.vstack(XS); TID = np.concatenate(TID); Y = np.concatenate(YS)
Rv = np.concatenate(RES); SID = np.concatenate(SID); R1 = np.concatenate(R1F)
tr = TID < t_cut; te = ~tr
log(f'样本 {len(X):,}  训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')

clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_depth=4,
                                     min_samples_leaf=80, random_state=7)
clf.fit(X[tr], Y[tr])
p = clf.predict_proba(X)[:, 1]
log(f'onset 模型测试段 AUC = {roc_auc_score(Y[te], p[te]):.4f}')

log('\n' + '=' * 116)
log('【① 单独用 onset 分数】测试段，+30/-15/48h，扣 0.09%')
log('=' * 116)
base = Rv[te].mean()
log(f'  全样本（基准）笔 {int(te.sum()):,}  期望净 {100*base:+.2f}%')
log(f'  {"分位":>8}{"阈值":>8}{"笔数":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}')
for q in (50, 75, 90, 95, 99):
    thr = np.percentile(p[te], q)
    m = te & (p >= thr)
    if m.sum() < 30:
        continue
    log(f'  {q:>7}%{thr:>8.3f}{int(m.sum()):>8,}{100*Rv[m].mean():>+10.2f}'
        f'{100*np.median(Rv[m]):>9.2f}{100*(Rv[m] > 0).mean():>8.1f}')

log('\n' + '=' * 116)
log('【② 级联：R1 规则（市场齐涨≥2 & 24h涨≥5%） × onset 分数分位')
log('=' * 116)
r1m = te & (R1 == 1)
log(f'  R1 单独：笔 {int(r1m.sum()):,}  期望净 {100*Rv[r1m].mean():+.2f}%  '
    f'中位 {100*np.median(Rv[r1m]):+.2f}%  胜率 {100*(Rv[r1m] > 0).mean():.1f}%')
log(f'  {"再叠加 onset 分位":>16}{"笔数":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"提升":>8}')
for q in (50, 70, 80, 90):
    thr = np.percentile(p[r1m], q) if r1m.sum() > 0 else 0
    m = r1m & (p >= thr)
    if m.sum() < 30:
        continue
    e = Rv[m].mean()
    log(f'  {"≥P"+str(q):>16}{int(m.sum()):>8,}{100*e:>+10.2f}'
        f'{100*np.median(Rv[m]):>9.2f}{100*(Rv[m] > 0).mean():>8.1f}'
        f'{100*(e - Rv[r1m].mean()):>+7.2f}%')

out = os.path.join(OUT, 'cascade_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
