# -*- coding: utf-8 -*-
"""gambler_early3.py — 无前视：能不能识别"刚开始、还在早期"的币？早期的是否真有更大空间？

用户问题（2026-10-16）："到底能不能发现，哪些币已经开始起局没多久，属于周期的早期？"

★ 拆成三问，本脚本只答第 2、3 问（第 1 问 ops85/leadtime 已答：确认已开始 AUC 0.94、提前 1h AUC 0.82）
   ① 能不能发现"正在起局"      → 已答（能）
   ② 能不能知道"这波到哪了"    → 本脚本：用**实时可见**的 over288（距 288 根低点涨幅）做进度代理
   ③ 早期的是否有更大空间      → 本脚本：按 over288 / 放量 / 启动信号分档，看未来 24h 最大涨幅分布

★ 全部特征只用 t 时刻及之前（over288、量比、涨幅都是实时可观测），标签才是未来。
   上一版 gambler_phase 的 AUC 0.8834 **有前视**（"局起点"事后定义），不能算数。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_early3.txt')
WDAY = 288
FEE = 0.0004 + 0.0005

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_early3 · 无前视的"早期识别" + "还有多大空间"')
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
    if n < WDAY * 3: continue
    hs = pd.Series(h)
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0                       # 实时：距低点涨幅
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0      # 实时：过去 1h 涨幅
    r24 = np.full(n, np.nan); r24[24:] = c[24:] / c[:-24] - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)   # 实时：单根量比
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values    # 实时：近 2h 平均量比
    fmax = hs.rolling(WDAY, min_periods=1).max().shift(-WDAY).values
    f_maxup = fmax / c - 1.0                                         # 未来 24h 最大涨幅（标签）
    f_min = pd.Series(l).rolling(WDAY, min_periods=1).min().shift(-WDAY).values
    f_maxdn = f_min / c - 1.0                                        # 未来 24h 最大跌幅
    # 只取"已经动了"的时刻：距低点已涨 >= 5%（说明这波已启动）
    sel = (over >= 0.05) & np.isfinite(f_maxup) & np.isfinite(volr24) & np.isfinite(r12)
    idx = np.flatnonzero(sel)
    if len(idx) < 300: continue
    idx = idx[::5]
    rows.append(pd.DataFrame({'sym': sym, 't': ot[idx], 'over': over[idx],
                              'r12': r12[idx], 'r24': r24[idx], 'volr24': volr24[idx],
                              'f_maxup': f_maxup[idx], 'f_maxdn': f_maxdn[idx]}))
    del hs; gc.collect()

df = pd.concat(rows, ignore_index=True)
del rows; gc.collect()
log(f'样本（已启动且 over>=5%）{len(df):,}')

log('')
log('=' * 100)
log('【问②：能不能知道这波到哪了 —— 按"距低点涨幅 over288"分档，看未来空间】')
log('=' * 100)
log(f'  {"over288 分档":<18} {"样本":>10} {"P(未来24h再涨>=30%)":>20} {"P(再涨>=50%)":>16} '
    f'{"未来24h最大涨(中位)":>20} {"未来24h最大跌(中位)":>20}')
bins = [0.05, 0.10, 0.20, 0.50, 1.00, 1e9]
labs = ['5~10%', '10~20%', '20~50%', '50~100%', '>100%']
cut = pd.cut(df['over'], bins=bins, labels=labs)
gb = df.groupby(cut, observed=True)
for k in labs:
    if k not in gb.groups: continue
    sub = gb.get_group(k)
    log(f'  {k:<18} {len(sub):>10,} {100.0*(sub["f_maxup"]>=0.30).mean():>19.2f}% '
        f'{100.0*(sub["f_maxup"]>=0.50).mean():>15.2f}% '
        f'{100*sub["f_maxup"].median():>19.2f}% {100*sub["f_maxdn"].median():>19.2f}%')

log('')
log('=' * 100)
log('【问③：早期 + 放量 + 温和启动 —— 这类组合未来空间最大吗】')
log('=' * 100)
log(f'  {"条件":<42} {"样本":>10} {"P(再涨>=30%)":>14} {"P(再涨>=50%)":>14} {"中位最大涨":>12} {"中位最大跌":>12}')
CONDS = [
    ('① 任意已启动(over>=5%)',                                    df['over'] >= 0.05),
    ('② 早期(over 5~20%)',                                        (df['over'] >= 0.05) & (df['over'] < 0.20)),
    ('③ 早期 + 放量(volr24>=2)',                                   (df['over'] >= 0.05) & (df['over'] < 0.20) & (df['volr24'] >= 2)),
    ('④ 早期 + 放量 + 过去1h涨>=5%',                               (df['over'] >= 0.05) & (df['over'] < 0.20) & (df['volr24'] >= 2) & (df['r12'] >= 0.05)),
    ('⑤ 早期 + 放量 + 过去1h涨 5~15%(温和)',                        (df['over'] >= 0.05) & (df['over'] < 0.20) & (df['volr24'] >= 2) & (df['r12'] >= 0.05) & (df['r12'] < 0.15)),
    ('⑥ 早期 + 缩量(volr24<1)',                                    (df['over'] >= 0.05) & (df['over'] < 0.20) & (df['volr24'] < 1)),
    ('⑦ 中段(over 20~50%)',                                       (df['over'] >= 0.20) & (df['over'] < 0.50)),
    ('⑧ 已大涨(over>50%)',                                         df['over'] >= 0.50),
]
for nm, m in CONDS:
    sub = df[m.values]
    if len(sub) < 100:
        log(f'  {nm:<42} {len(sub):>10,}  (样本不足)'); continue
    log(f'  {nm:<42} {len(sub):>10,} {100.0*(sub["f_maxup"]>=0.30).mean():>13.2f}% '
        f'{100.0*(sub["f_maxup"]>=0.50).mean():>13.2f}% '
        f'{100*sub["f_maxup"].median():>11.2f}% {100*sub["f_maxdn"].median():>11.2f}%')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若"早期(5~20%)"的 P(再涨>=50%) 明显高于"已大涨(>50%)" ⇒ 早期确实还有空间，进度可作实时判据')
log('  · 若两者相近 ⇒ over288 对未来空间无判别力，"早期识别"只解决"有没有在动"，不解决"还能涨多少"')
log('  · 未来 24h 最大跌中位若在早期档就很深 ⇒ 追早期的风险同样大')
log('  · 全程无前视：over/volr/r12 在 t 时刻全部可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
