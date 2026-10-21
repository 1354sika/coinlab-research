# -*- coding: utf-8 -*-
"""
问题：把全市场里"符合他那种交易"的样本全部找出来，作为训练集。
定义（按用户口径）：只看 K 线；入场后 8h（96 根 5m）内 max(high) 涨幅 >= 50%。
同时给出 >=20% / >=30% / >=50% / >=100% 四档，作为训练集正样本规模。
输出 coinlab/evidence/positives_census.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_502.csv'
OUT = 'D:/中信证券/coinlab/evidence/positives_census.txt'
H = 96; STEP = 96

D = pd.read_csv(CSV, usecols=['symbol','open_time','open','high','close'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','close'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

THS = [0.05, 0.10, 0.20, 0.50]
rows = []
for sym, g in D.groupby('symbol', sort=False):
    o = g.open.to_numpy(); h = g.high.to_numpy(); t = g.open_time.to_numpy()
    n = len(o)
    if n < H + 2:
        continue
    idx = np.arange(0, n - H, STEP)
    if len(idx) == 0:
        continue
    entry = o[idx]
    # 未来 H 根的 max high：用 rolling max on reversed
    from numpy.lib.stride_tricks import sliding_window_view
    hw = sliding_window_view(h, H)            # shape (n-H+1, H)
    fmax = hw.max(axis=1)                     # fmax[i] = max(h[i:i+H])
    fut = fmax[idx]
    ok = entry > 0
    r = np.where(ok, fut / np.maximum(entry, 1e-12) - 1, np.nan)
    rec = {'sym': sym, 'n': int(ok.sum())}
    for th in THS:
        rec['ge%g' % th] = int(np.nansum(r >= th))
    rows.append(rec)
    if len(rows) % 50 == 0:
        print('  ...%d 币 %.0fs' % (len(rows), time.time()-t0), flush=True)

P = pd.DataFrame(rows)
print('币数', len(P), '入场点', int(P.n.sum()), '%.1fs' % (time.time()-t0), flush=True)

L = []
L.append('"符合他那种交易"的正样本普查   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('定义：只看 K 线；每币每 96 根（8h）取一个入场点；入场后 96 根内 max(high) 涨幅 >= 阈值')
L.append('数据：klines_5m_502.csv（304 币）；入场点 %d' % int(P.n.sum()))
L.append('')
L.append('=== 一、正样本规模（这次能拿来训练的全部）')
L.append('  阈值     正样本数     占入场点      折合每年')
years = 3.0
for th in THS:
    c = 'ge%g' % th
    tot = int(P[c].sum())
    L.append('  >=%-5s %8d   %8.4f%%   %8.1f 个/年' % ('%d%%' % (th*100), tot, tot / max(int(P.n.sum()),1) * 100, tot / years))
L.append('')
L.append('=== 二、正样本最集中的币（Top 15，按 >=50%% 计数）')
top = P.sort_values('ge0.5', ascending=False).head(15)
for _, r in top.iterrows():
    L.append('  %-12s 入场 %6d  >=50%% %4d (%.2f%%)  >=20%% %4d' % (
        r['sym'], int(r['n']), int(r['ge0.5']), r['ge0.5']/max(r['n'],1)*100, int(r['ge0.2'])))
L.append('')
L.append('=== 三、读法')
L.append('  正样本数决定模型能不能训：')
L.append('    <500  个：只能做单变量阈值/直觉，训不出可靠分类器')
L.append('    500~5000：可以训 3~5 个特征的逻辑回归，但必须强正则 + 事件去重')
L.append('    >5000：可以试小模型（GBDT），但仍要防过拟合')
L.append('')
L.append('诚实边界：')
L.append('  1. 这是"到过 50%%"的浮盈口径，不是可实现收益（high 触价、零滑点、成交率未计）。')
L.append('  2. 每币每 8h 取一个入场点，同一波行情会被切成多个样本（样本重叠），真实独立事件数远少于上表。')
L.append('  3. 304 币含大量小市值，流动性差，50%% 涨幅的"到过"很可能卖不出去。')
L.append('  4. 未做去重，未按时间切分训练/验证。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
