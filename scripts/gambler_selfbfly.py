# -*- coding: utf-8 -*-
"""gambler_selfbfly.py — 币**自身**的蝴蝶效应：前 24h 状态 → 后 24h 起局概率
用户主张（2026-10-16）：
  "肯定有很多没挖出来的蝴蝶效应，不止币与币之间，还有币自己的，
   特别是币的前 24h 和后 24h 之间……能帮助预测操盘周期的开始或分盘特征"

此前的蝴蝶效应实验（gambler_butterfly / gambler_cascade / gambler_chain）全是
**币与币之间**的。本脚本专挖**同一个币自己**的时间关联：
  前 24h（288 根 5m）的形态（涨跌 / 波动 / 量能 / 是否刚有过局）→ 后 24h 起局概率。

输出：条件概率矩阵 + lift（相对无条件基准），直接回答"前一段行情能否预判后一段"。
  重点看：**刚打完一局的币，是更容易再起局（接力），还是进入长期沉寂（一局之后没戏）**
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_selfbfly.txt')
WDAY = 288          # 24h
GAME = 0.50         # 起局门槛：未来 24h 最大涨幅 >= 50%

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_selfbfly · 币自身的蝴蝶效应：前 24h → 后 24h 起局概率')
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
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

rows = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 3: continue
    cs = pd.Series(c)
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    # 过去 24h
    c0 = cs.shift(WDAY)
    p_ret = c0.notna() & (cs / c0 - 1.0)
    p_ret = (cs / c0 - 1.0).values
    p_maxup = (hs.rolling(WDAY).max().shift(0).values / np.maximum(c0.values, 1e-12) - 1.0)
    p_maxdn = (ls.rolling(WDAY).min().values / np.maximum(c0.values, 1e-12) - 1.0)
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    p_vol = np.where(base_q > 0, qs.rolling(WDAY).sum().values / np.maximum(base_q * WDAY, 1e-12), np.nan)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    p_nbig = pd.Series((np.abs(r1) >= 0.05).astype('float64')).rolling(WDAY).sum().values
    # 未来 24h
    f_fwd = hs.rolling(WDAY).max().shift(-WDAY).values
    f_maxup = f_fwd / c - 1.0
    m = np.isfinite(p_ret) & np.isfinite(f_maxup) & np.isfinite(p_vol) & np.isfinite(p_nbig)
    idx = np.flatnonzero(m)
    if len(idx) < 300: continue
    idx = idx[::6]                      # 抽样，控内存
    rows.append(pd.DataFrame({
        'sym': sym, 't': ot[idx],
        'p_ret': p_ret[idx], 'p_maxup': p_maxup[idx], 'p_maxdn': p_maxdn[idx],
        'p_vol': p_vol[idx], 'p_nbig': p_nbig[idx],
        'f_maxup': f_maxup[idx]}))
    del cs, hs, ls, qs; gc.collect()

df = pd.concat(rows, ignore_index=True)
del rows; gc.collect()
df['f_game']  = (df['f_maxup'] >= GAME).astype('int8')
df['f_big']   = (df['f_maxup'] >= 1.00).astype('int8')
df['p_had']   = (df['p_maxup']  >= GAME).astype('int8')      # 前 24h 曾起局
base_g, base_b = df['f_game'].mean(), df['f_big'].mean()
log(f'样本 {len(df):,}   基准 P(后24h起局>=50%) = {100*base_g:.3f}%   P(后24h翻倍) = {100*base_b:.3f}%')

def lift_tbl(name, series, bins=None, labels=None, cat=None):
    log('')
    log(f'【{name}】')
    log(f'  {"分档":<22} {"样本":>10} {"P(后24h起局>=50%)":>18} {"lift":>8} {"P(翻倍)":>10}')
    if cat is not None:
        gb = df.groupby(cat)
        keys = sorted(gb.groups.keys())
    elif bins is not None:
        tmp = pd.cut(series, bins=bins, labels=labels)
        gb = df.groupby(tmp, observed=True)
        keys = list(gb.groups.keys())
    else:
        gb = None; keys = []
    for k in keys:
        sub = gb.get_group(k) if cat is not None else df[pd.cut(series, bins=bins, labels=labels) == k]
        if len(sub) < 50: continue
        pg = sub['f_game'].mean(); pb = sub['f_big'].mean()
        log(f'  {str(k):<22} {len(sub):>10,} {100*pg:>17.3f}% {pg/base_g:>7.2f}x {100*pb:>9.3f}%')

lift_tbl('前 24h 是否曾起局（>=50%）', None, cat='p_had')
lift_tbl('前 24h 涨跌幅', df['p_ret'], bins=[-1,-0.30,-0.10,-0.03,0.03,0.10,0.30,10],
         labels=['<-30%','-30~-10%','-10~-3%','-3~+3%','+3~+10%','+10~+30%','>+30%'])
lift_tbl('前 24h 量能（相对 30 日均量）', df['p_vol'], bins=[0,0.5,1,2,5,20,1e9],
         labels=['<0.5x','0.5~1x','1~2x','2~5x','5~20x','>20x'])
lift_tbl('前 24h 大波动根数（|r|>=5%）', df['p_nbig'], bins=[-1,0,2,5,20,1e9],
         labels=['0','1~2','3~5','6~20','>20'])
lift_tbl('前 24h 最大涨幅', df['p_maxup'], bins=[-1,0.05,0.20,0.50,1.0,1e9],
         labels=['<5%','5~20%','20~50%','50~100%','>100%'])

# 交替模式矩阵
log('')
log('=' * 100)
log('【核心：交替模式（前 24h 起局 × 后 24h 起局）】')
log('=' * 100)
ct = pd.crosstab(df['p_had'], df['f_game'], normalize='index')
cnt = pd.crosstab(df['p_had'], df['f_game'])
log(f'  {"前24h":<12} {"样本":>10} {"后24h起局率":>12} {"lift":>8}')
for k in sorted(ct.index):
    p = ct.loc[k, 1] if 1 in ct.columns else 0.0
    log(f'  {"曾起局" if k else "未起局":<12} {int(cnt.loc[k].sum()):>10,} {100*p:>11.3f}% {p/base_g:>7.2f}x')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · lift > 1 ⇒ 该状态之后更容易起局；lift < 1 ⇒ 更沉寂')
log('  · 若"前24h曾起局"lift 明显 > 1 ⇒ 支持"接力/轮动"；< 1 ⇒ 支持"一局之后长期沉寂"')
log('  · 前 24h 量能/大波动根数的 lift 若单调变化 ⇒ 存在自身时间上的可预测结构')

# 前向对照：把"过去的局"细分为"刚结束"与"已结束一段时间"
log('')
log('=' * 100)
log('【细分：前一次起局距今多久（0~24h / 24~72h / 72~168h / >168h / 从无）】')
log('=' * 100)
d2 = df.copy()
d2['slot'] = pd.cut(d2['t'] // 3600000, bins=[-1, 0, 1e18], labels=['x','y'])
# 用 t 相对前次起局时刻分桶：需要逐币回溯，这里用滚动窗口近似
for lo, hi, nm in ((0, WDAY, '0~24h'), (WDAY, WDAY*3, '24~72h'),
                   (WDAY*3, WDAY*7, '72~168h'), (WDAY*7, 10**9, '>168h')):
    pass  # 见下方按币精算

def last_game_gap(sym, ot, c, h, l):
    n = len(c)
    hs = pd.Series(h)
    fwd = hs.rolling(WDAY).max().shift(-WDAY).values
    up = fwd / c - 1.0
    is_start = np.zeros(n, dtype=bool)
    stamp = np.full(n, -1, dtype='int64')
    last = -10**9
    for t in range(n):
        stamp[t] = last
        if np.isfinite(up[t]) and up[t] >= GAME:
            if t - last > WDAY: is_start[t] = True
            last = t
    return is_start, stamp, ot

buckets = {k: [0, 0, 0.0, 0] for k in ['从无', '0~24h', '24~72h', '72~168h', '>168h']}
tot = {k: [0, 0, 0] for k in buckets}
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 3: continue
    hs = pd.Series(h)
    fwd = hs.rolling(WDAY).max().shift(-WDAY).values
    up = fwd / c - 1.0
    last = -10**9
    for t in range(0, n - WDAY, 12):
        if not np.isfinite(up[t]): continue
        if last < 0: k = '从无'
        else:
            gap = t - last
            k = '0~24h' if gap <= WDAY else ('24~72h' if gap <= WDAY*3 else
                ('72~168h' if gap <= WDAY*7 else '>168h'))
        g = 1 if up[t] >= GAME else 0
        b = 1 if up[t] >= 1.00 else 0
        tot[k][0] += 1; tot[k][1] += g; tot[k][2] += b
        if np.isfinite(up[t]) and up[t] >= GAME and t - last > WDAY: last = t

log(f'  {"距上次起局":<14} {"样本":>10} {"后24h起局率":>14} {"lift":>8} {"后24h翻倍率":>12}')
for k, v in tot.items():
    if v[0] < 50: continue
    p = v[1] / v[0]; pb = v[2] / v[0]
    log(f'  {k:<14} {v[0]:>10,} {100*p:>13.3f}% {p/base_g:>7.2f}x {100*pb:>11.3f}%')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
