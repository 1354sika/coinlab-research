# -*- coding: utf-8 -*-
"""gambler_pure.py — 提纯：从"已识别的早期局候选"里挑出真大局
用户要求（2026-10-16）：
  "对所有识别到的局提纯，留下真的大局。如果真的没有，那么再同一时间，就只盯着小局"

候选（t 时刻可观测，严格无前视）：
    over288 = c / (288 根最低价) - 1    ∈ [3%, 40%]   已启动但未走远
    volr24  = 近 2h 平均量比 >= 1                     有量
结果（事后标注，**只用于评价提纯效果，不作为特征**）：
    真局 = 未来 48h 内最大涨幅 >= 50% 且 峰后 24h 回撤 >= 30%
    半真 = 未来 48h 内最大涨幅 20~50%
    假局 = 未来 48h 内最大涨幅 < 20%

输出：① 各单一特征分档的"真局率" ② 组合提纯规则的真局率与覆盖率
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_pure.txt')
WDAY = 288       # 24h
H48 = WDAY * 2   # 48h

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_pure · 提纯：从早期局候选里挑出真大局')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
R1all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R1all, nan=0.0) >= 0.05).sum(axis=1)
dn_cnt = (np.nan_to_num(R1all, nan=0.0) <= -0.05).sum(axis=1)
sync_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

rows = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    tx = np.full(n, np.nan); tx[24:] = tr[24:] / np.maximum(pd.Series(tr).rolling(24).mean().values[24:], 1e-12)
    # 结果（未来 48h）
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_maxup = fmax / c - 1.0
    # 峰后 24h 回撤
    dd = np.full(n, np.nan)
    # 仅对可选样本算
    ii = pd.Index(ot)
    sync_v = pd.Series(sync_up, index=ts).reindex(ii).values
    btc_v = pd.Series(btc_r24, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24

    sel = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1) & np.isfinite(f_maxup) & np.isfinite(r288)
    idx = np.flatnonzero(sel)
    if len(idx) < 200: continue
    idx = idx[::3]
    # 计算每个选中点的峰后回撤（未来 48h 内峰值 → 之后 24h）
    for t in idx:
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: dd_t = np.nan; continue
        ipk = int(np.argmax(seg))
        pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 <= t + 1 + ipk: dd_t = np.nan
        else: dd_t = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
        dd[t] = dd_t
    rows.append(pd.DataFrame({
        'sym': sym, 't': ot[idx], 'over': over[idx], 'r12': r12[idx], 'r288': r288[idx],
        'volr24': volr24[idx], 'tx': tx[idx], 'sync': sync_v[idx], 'btc': btc_v[idx],
        'hour': hh[idx], 'f_maxup': f_maxup[idx], 'dd': dd[idx]}))
    del hs, ls, qs; gc.collect()

df = pd.concat(rows, ignore_index=True)
del rows; gc.collect()
df = df[np.isfinite(df['f_maxup']) & np.isfinite(df['dd'])]
df['cls'] = '假局(<20%)'
df.loc[df['f_maxup'] >= 0.20, 'cls'] = '半真(20~50%)'
df.loc[(df['f_maxup'] >= 0.50) & (df['dd'] >= 0.30), 'cls'] = '真局(>=50%且崩>=30%)'
log(f'候选样本 {len(df):,}')
for k, v in df['cls'].value_counts().items():
    log(f'  {k:<24} {v:>9,}  ({100.0*v/len(df):.3f}%)')
base = (df['cls'] == '真局(>=50%且崩>=30%)').mean()
log(f'  基准真局率 = {100*base:.3f}%')

def tbl(nm, series, bins, labs):
    log('')
    log(f'【{nm}】')
    cut = pd.cut(series, bins=bins, labels=labs)
    gb = df.groupby(cut, observed=True)
    log(f'  {"分档":<16} {"样本":>10} {"真局率":>9} {"lift":>7} {"半真率":>9}')
    for k in labs:
        if k not in gb.groups: continue
        s = gb.get_group(k)
        r = (s['cls'] == '真局(>=50%且崩>=30%)').mean()
        h = (s['cls'] == '半真(20~50%)').mean()
        log(f'  {str(k):<16} {len(s):>10,} {100*r:>8.3f}% {r/base:>6.2f}x {100*h:>8.3f}%')

tbl('已涨幅 over288', df['over'], [0.03,0.05,0.08,0.12,0.20,0.30,0.41], ['3~5%','5~8%','8~12%','12~20%','20~30%','30~40%'])
tbl('近2h量比 volr24', df['volr24'], [1,1.5,2,3,5,10,1e9], ['1~1.5','1.5~2','2~3','3~5','5~10','>10'])
tbl('过去1h涨幅 r12', df['r12'], [-10,-0.02,0.02,0.05,0.10,0.20,10], ['<-2%','-2~2%','2~5%','5~10%','10~20%','>20%'])
tbl('过去24h涨幅 r288', df['r288'], [-10,-0.10,0.0,0.10,0.30,1.0,1e9], ['<-10%','-10~0%','0~10%','10~30%','30~100%','>100%'])
tbl('市场同步(涨>=5%币数)', df['sync'], [-1,0,1,3,10,1e9], ['0','1','2~3','4~10','>10'])
tbl('BTC 过去24h', df['btc'], [-10,-0.03,0.0,0.03,10], ['<-3%','-3~0%','0~3%','>3%'])
tbl('UTC 小时', df['hour'], [-1,6,12,18,24], ['0~6','6~12','12~18','18~24'])

log('')
log('=' * 100)
log('【组合提纯：真局率 与 覆盖率】')
log('=' * 100)
RULES = [
    ('R0 全部候选',                              df['over'] > 0),
    ('R1 volr24>=3',                             df['volr24'] >= 3),
    ('R2 volr24>=5',                             df['volr24'] >= 5),
    ('R3 volr24>=3 且 r12>=5%',                   (df['volr24'] >= 3) & (df['r12'] >= 0.05)),
    ('R4 volr24>=3 且 r12>=5% 且 同步>=2',          (df['volr24'] >= 3) & (df['r12'] >= 0.05) & (df['sync'] >= 2)),
    ('R5 volr24>=5 且 r12>=5% 且 同步>=2',          (df['volr24'] >= 5) & (df['r12'] >= 0.05) & (df['sync'] >= 2)),
    ('R6 volr24>=5 且 r12 5~15% 且 同步>=2',        (df['volr24'] >= 5) & (df['r12'] >= 0.05) & (df['r12'] < 0.15) & (df['sync'] >= 2)),
    ('R7 volr24>=3 且 r12 5~15% 且 over 8~25%',    (df['volr24'] >= 3) & (df['r12'] >= 0.05) & (df['r12'] < 0.15) & (df['over'] >= 0.08) & (df['over'] < 0.25)),
]
tot_true = (df['cls'] == '真局(>=50%且崩>=30%)').sum()
log(f'  {"规则":<40} {"样本":>9} {"真局率":>9} {"lift":>7} {"覆盖真局%":>11}')
for nm, m in RULES:
    s = df[m.values]
    if len(s) < 30:
        log(f'  {nm:<40} {len(s):>9,}  (样本不足)'); continue
    r = (s['cls'] == '真局(>=50%且崩>=30%)').mean()
    cov = 100.0 * (s['cls'] == '真局(>=50%且崩>=30%)').sum() / max(tot_true, 1)
    log(f'  {nm:<40} {len(s):>9,} {100*r:>8.3f}% {r/base:>6.2f}x {cov:>10.1f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 真局率显著高于基准(且 lift>2)的规则 = 有效提纯')
log('  · 若所有规则的 lift 都 <1.5 ⇒ 早期候选里无法区分真/假局，只能承认"识别不出大局"')
log('  · 全程无前视：over/r12/r288/volr24/tx/sync/btc/hour 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
