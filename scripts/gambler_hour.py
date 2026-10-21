# -*- coding: utf-8 -*-
"""gambler_hour.py — 逐小时检验：真的是"UTC 18~24 特殊"吗？
承 randsearch（job-47368-42）：60,000 次随机组合搜索挖出 hour18~24 相关组合，
但那次只把时段切成 4 个粗格（0~6/6~12/12~18/18~24）。本脚本把 24 个小时**逐个**拉出来看：

  ① 全候选样本上，各 UTC 小时的真局率（时段本身有没有差异）
  ② 在"已大涨子集(r288>=30% & volr6>=5)"上，各小时的真局率（时段是不是"放大器"）
  ③ 各小时的交易期望净收益（+50%TP/-15%SL/24hTO，扣 0.09%）

若 hour18~24 在 ①②③ 里都明显突出 ⇒ 是真信号；若只是个别小时偶然高 ⇒ 是噪声。
全程无前视：所有条件在 t 时刻可观测。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_hour.txt')
WDAY = 288
H48 = WDAY * 2
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_hour · 逐 UTC 小时检验（24 小时 × 三种口径）')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','quote_volume':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

def simulate(lo, c, h, l, tp=TP, sl=SL, to=TO):
    n = len(c)
    end = min(n - 1, lo + to)
    if end <= lo + 1: return None
    tp_p = c[lo] * (1 + tp); sl_p = c[lo] * (1 + sl)
    ht = np.flatnonzero(h[lo+1:end+1] >= tp_p)
    hs = np.flatnonzero(l[lo+1:end+1] <= sl_p)
    it = ht[0] if len(ht) else 10**9
    isl = hs[0] if len(hs) else 10**9
    if it == 10**9 and isl == 10**9: return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if it <= isl: return tp - FEE, 'tp'
    return sl - FEE, 'sl'

records = []      # 逐样本：hour, is_big, true, （交易结果单独收集）
TRADE = {h: [] for h in range(24)}
TRADE_BIG = {h: [] for h in range(24)}

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    hh = (ot // 3600000) % 24
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    base = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    ok = base & np.isfinite(r288) & np.isfinite(volr6) & np.isfinite(f_up)
    for t in np.flatnonzero(ok)[::3]:
        e = min(n - 1, t + H48); seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            d = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
            big = (r288[t] >= 0.30) and (volr6[t] >= 5) if np.isfinite(r288[t]) and np.isfinite(volr6[t]) else False
            records.append((int(hh[t]), int(big), 1 if (f_up[t] >= 0.50 and d >= 0.30) else 0))
    # 交易：按小时分别收集（候选池入场）
    for t in np.flatnonzero(ok):
        r = simulate(t, c, h, l)
        if r is None: continue
        TRADE[int(hh[t])].append(r[0])
    bigm = ok & (r288 >= 0.30) & (volr6 >= 5)
    last = -10 ** 9
    for t in np.flatnonzero(bigm):
        if t - last < 12: continue
        r = simulate(t, c, h, l)
        if r is None: continue
        TRADE_BIG[int(hh[t])].append(r[0]); last = t
    del hs, ls, qs; gc.collect()

df = pd.DataFrame(records, columns=['hour', 'big', 'true'])
del records; gc.collect()
base_all = df['true'].mean()
sub = df[df['big'] == 1]
base_big = sub['true'].mean() if len(sub) else float('nan')
log(f'全候选 {len(df):,}（基准真局率 {100*base_all:.3f}%）')
log(f'已大涨子集 {len(sub):,}（真局率 {100*base_big:.3f}%）')

log('')
log('=' * 100)
log('【① 全候选：逐小时真局率　② 已大涨子集：逐小时真局率　③ 逐小时交易期望净】')
log('=' * 100)
log(f'  {"UTC时":>5} {"①候选真局率":>12} {"①n":>9} {"②大涨真局率":>12} {"②n":>8} '
    f'{"③候选池交易期望净":>16} {"③n":>8} {"④大涨池交易期望净":>16} {"④n":>7}')
tab = df.groupby('hour')['true'].agg(['mean', 'size'])
subtab = sub.groupby('hour')['true'].agg(['mean', 'size'])
for hh in range(24):
    a = tab.loc[hh, 'mean'] if hh in tab.index else np.nan
    an = int(tab.loc[hh, 'size']) if hh in tab.index else 0
    b = subtab.loc[hh, 'mean'] if hh in subtab.index else np.nan
    bn = int(subtab.loc[hh, 'size']) if hh in subtab.index else 0
    ta = np.array(TRADE[hh]) if TRADE[hh] else np.array([np.nan])
    tb = np.array(TRADE_BIG[hh]) if TRADE_BIG[hh] else np.array([np.nan])
    mark = '  ←' if 18 <= hh <= 23 else ''
    log(f'  {hh:>5} {100*a:>11.3f}% {an:>9,} {100*b:>11.3f}% {bn:>8,} '
        f'{100*np.nanmean(ta):>15.3f}% {len(TRADE[hh]):>8,} '
        f'{100*np.nanmean(tb):>15.3f}% {len(TRADE_BIG[hh]):>7,}{mark}')

# hour18~24 汇总 vs 其他
h18 = df[(df['hour'] >= 18)]
ho = df[(df['hour'] < 18)]
s18 = sub[sub['hour'] >= 18]; so = sub[sub['hour'] < 18]
log('')
log('=' * 100)
log('【汇总：UTC 18~24 vs 其他时段】')
log('=' * 100)
log(f'  全候选：18~24 {100*h18["true"].mean():.3f}%（n={len(h18):,}）  vs  其他 {100*ho["true"].mean():.3f}%（n={len(ho):,}）')
log(f'  大涨子集：18~24 {100*s18["true"].mean():.3f}%（n={len(s18):,}）  vs  其他 {100*so["true"].mean():.3f}%（n={len(so):,}）')
t18 = np.concatenate([TRADE[a] for a in range(18, 24)]) if any(TRADE[a] for a in range(18, 24)) else np.array([np.nan])
to = np.concatenate([TRADE[a] for a in range(0, 18)]) if any(TRADE[a] for a in range(0, 18)) else np.array([np.nan])
log(f'  候选池交易：18~24 {100*np.nanmean(t18):.3f}%（n={len(t18):,}）  vs  其他 {100*np.nanmean(to):.3f}%（n={len(to):,}）')
b18 = np.concatenate([TRADE_BIG[a] for a in range(18, 24)]) if any(TRADE_BIG[a] for a in range(18, 24)) else np.array([np.nan])
bo = np.concatenate([TRADE_BIG[a] for a in range(0, 18)]) if any(TRADE_BIG[a] for a in range(0, 18)) else np.array([np.nan])
log(f'  大涨池交易：18~24 {100*np.nanmean(b18):.3f}%（n={len(b18):,}）  vs  其他 {100*np.nanmean(bo):.3f}%（n={len(bo):,}）')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若 24 小时的真局率曲线在 18~23 处形成明显高原 ⇒ 真时段效应')
log('  · 若只是个别小时高、相邻小时低 ⇒ 是随机波动，不可用')
log('  · 全程无前视')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
