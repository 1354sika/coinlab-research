# -*- coding: utf-8 -*-
"""gambler_inter.py — 特征交互：因果是否真的"环环相扣、互相叠加"？
用户主张（2026-10-16）："因果概率是环环相扣，互相影响并叠加的"

现状：此前全部是**单变量分档**（pure / pure2 / pure3），从未系统测**交互**。
本脚本对 4 个已验证有效维度做**两两交叉表**，检查是否存在"超加性"：

    lift(A且B) > lift(A) × lift(B)   ⇒ **正交互（确实叠加，1+1>2）**
    lift(A且B) ≈ lift(A) × lift(B)   ⇒ 独立（无交互）
    lift(A且B) < lift(A) × lift(B)   ⇒ 冗余（两条件在解释同一件事）

四个维度（t 时刻全部可观测，无前视）：
    volr24   近 2h 平均量比      档: [1, 3, 5, 10]
    r12      过去 1h 涨幅        档: [0, 2%, 5%, 10%]
    over288  距 288 根低点涨幅   档: [3%, 8%, 15%, 25%]
    r288     过去 24h 涨幅       档: [0, 10%, 30%]
目标：真局率（未来 48h 最大涨幅 >=50% 且 峰后 24h 回撤 >=30%）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_inter.txt')
WDAY = 288
H48 = WDAY * 2

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_inter · 特征交互（超加性检验）')
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

rows = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    sel = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1) & np.isfinite(f_up) & np.isfinite(r288) & np.isfinite(r12)
    idx = np.flatnonzero(sel)
    if len(idx) < 200: continue
    idx = idx[::3]
    for t in idx:
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    rows.append(pd.DataFrame({'over': over[idx], 'r12': r12[idx], 'r288': r288[idx],
                              'volr24': volr24[idx], 'f_up': f_up[idx], 'dd': dd[idx]}))
    del hs, ls; gc.collect()

df = pd.concat(rows, ignore_index=True)
del rows; gc.collect()
df = df[np.isfinite(df['f_up']) & np.isfinite(df['dd'])]
df['true'] = ((df['f_up'] >= 0.50) & (df['dd'] >= 0.30)).astype('int8')
base = df['true'].mean()
log(f'候选样本 {len(df):,}   基准真局率 {100*base:.3f}%')

def binned(s, cuts, names):
    """返回档名 Series（category 类型；倒序覆盖以保证边界正确）"""
    out = pd.Series(np.full(len(s), names[-1], dtype=object), index=s.index)
    for i in range(len(cuts) - 1, -1, -1):
        out[s < cuts[i]] = names[i]
    return out.astype('category')

DIMS = {
    'volr24':  binned(df['volr24'], [3, 5, 10], ['1~3', '3~5', '5~10', '>=10']),
    'r12':     binned(df['r12'],    [0.02, 0.05, 0.10], ['<2%', '2~5%', '5~10%', '>=10%']),
    'over288': binned(df['over'],   [0.08, 0.15, 0.25], ['3~8%', '8~15%', '15~25%', '>=25%']),
    'r288':    binned(df['r288'],   [0.00, 0.10, 0.30], ['<0%', '0~10%', '10~30%', '>=30%']),
}

# 单变量 lift 表
log('')
log('=' * 100)
log('【单变量参考：各档真局率】')
log('=' * 100)
single = {}
for k, v in DIMS.items():
    gb = df.groupby(v, observed=True)['true'].mean()
    single[k] = dict(gb)
    log(f'  {k:<10} ' + '  '.join(f'{n}:{100*r:.2f}%({100*r/(100*base):.2f}x)' for n, r in gb.items()))

# 两两交叉
import itertools
log('')
log('=' * 100)
log('【两两交叉表：lift(实测) vs lift(独立假设 = liftA × liftB)】')
log('=' * 100)
pairs = list(itertools.combinations(DIMS.keys(), 2))
for a, b in pairs:
    log('')
    log(f'--- {a} × {b} ---')
    ct = df.groupby([DIMS[a], DIMS[b]], observed=True)['true'].agg(['mean', 'size'])
    log(f'  {"":<14}' + ''.join(f'{bn:>18}' for bn in DIMS[b].cat.categories if bn in set(DIMS[b])))
    for an in DIMS[a].cat.categories:
        if an not in set(DIMS[a]): continue
        cells = []
        for bn in DIMS[b].cat.categories:
            if bn not in set(DIMS[b]): continue
            key = (an, bn)
            if key not in ct.index:
                cells.append(f'{"--":>18}'); continue
            r = ct.loc[key, 'mean']; n = int(ct.loc[key, 'size'])
            la = single[a].get(an, np.nan) / base if base else np.nan
            lb = single[b].get(bn, np.nan) / base if base else np.nan
            hx = la * lb
            got = r / base if base else np.nan
            tag = ''
            if np.isfinite(hx) and hx > 0:
                ratio = got / hx
                tag = '叠加' if ratio > 1.3 else ('冗余' if ratio < 0.75 else '')
            cells.append(f'{100*r:>6.2f}%/{got:>5.2f}x/n{n:<5}{tag}')
        log(f'  {an:<14}' + ''.join(f'{c:>18}' for c in cells))

log('')
log('=' * 100)
log('【三阶：最强组合叠加】')
log('=' * 100)
TRI = [
    ('T1 volr24>=5 & r12>=5% & over>=15%', (df['volr24'] >= 5) & (df['r12'] >= 0.05) & (df['over'] >= 0.15)),
    ('T2 volr24>=5 & r12>=5% & 同步... r288>=10%', (df['volr24'] >= 5) & (df['r12'] >= 0.05) & (df['r288'] >= 0.10)),
    ('T3 volr24>=10 & r12>=5% & over>=15%', (df['volr24'] >= 10) & (df['r12'] >= 0.05) & (df['over'] >= 0.15)),
    ('T4 volr24>=5 & r288>=30% & over>=15%', (df['volr24'] >= 5) & (df['r288'] >= 0.30) & (df['over'] >= 0.15)),
    ('T5 四条件全上 volr24>=5 & r12>=5% & r288>=10% & over>=15%',
     (df['volr24'] >= 5) & (df['r12'] >= 0.05) & (df['r288'] >= 0.10) & (df['over'] >= 0.15)),
]
tot_true = df['true'].sum()
log(f'  {"组合":<52} {"样本":>8} {"真局率":>9} {"lift":>7} {"覆盖真局%":>11}')
for nm, m in TRI:
    s = df[m.values]
    if len(s) < 20:
        log(f'  {nm:<52} {len(s):>8,}  (样本不足)'); continue
    r = s['true'].mean()
    log(f'  {nm:<52} {len(s):>8,} {100*r:>8.3f}% {r/base:>6.2f}x {100.0*s["true"].sum()/max(tot_true,1):>10.2f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 交叉格里标"叠加"的 ⇒ 两条件超加性，值得组合；标"冗余"的 ⇒ 两条件在解释同一件事')
log('  · 全程无前视：volr24/r12/over288/r288 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
