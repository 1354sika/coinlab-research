# -*- coding: utf-8 -*-
"""gambler_weekly · 这套策略一星期赚多少（按自然周聚合）

规则与 trade_sim2 完全一致：
  入场：从低点涨 5%
  止盈：+50%    止损：-9.5%    超时：24h
  过滤：v240（5m 收益率 24h 滚动标准差）>= 训练段 p90

输出：
  1) 测试段（严格留出，v240 阈值只用训练段算）按 ISO 周聚合
  2) 全样本对照（同一规则，不切时段）
  每周给出：笔数、简单收益（10% 仓位累加）、复利收益（10% 仓位逐笔复利）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_weekly · 这套策略一星期赚多少')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
trades = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in need or len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        n = len(c)
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1
        v240 = pd.Series(r1).rolling(240, min_periods=50).std().values
        for _, row in by_sym[sym].iterrows():
            lo_i = int(row['lo_i']); lo_px = float(row['lo'])
            if lo_i >= n - 2: continue
            j = lo_i
            target = lo_px * 1.05
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1 or j - lo_i > 288: continue
            if not np.isfinite(v240[j]): continue
            tp_px = lo_px * 1.575
            sl_px = lo_px * 0.95
            entry = c[j]
            ret = None; reason = 'timeout'; bars = 0
            for k in range(j + 1, min(j + 289, n)):
                bars = k - j
                hit_tp = h[k] >= tp_px
                hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl:
                    ret = sl_px / entry - 1; reason = 'both→SL'; break
                if hit_tp:
                    ret = tp_px / entry - 1; reason = 'TP'; break
                if hit_sl:
                    ret = sl_px / entry - 1; reason = 'SL'; break
            if ret is None:
                kk = min(j + 288, n - 1)
                ret = c[kk] / entry - 1; bars = kk - j
            trades.append((ot[j], ret, bars, reason, v240[j], sym))
    del ch; gc.collect()

tr = pd.DataFrame(trades, columns=['t','ret','bars','reason','v240','sym'])
tr = tr.sort_values('t').reset_index(drop=True)
log(f'\n可模拟周期 {len(tr):,}')

cut = np.median(tr['t'].values)
trm = tr['t'].values <= cut
tem = tr['t'].values > cut
th = np.quantile(tr.loc[trm, 'v240'], 0.90)


def weekly(df, tag):
    if len(df) == 0:
        log(f'\n【{tag}】无样本'); return
    d = df.copy()
    d['dt'] = pd.to_datetime(d['t'], unit='ms', utc=True).dt.tz_localize(None)
    d['wk'] = d['dt'].dt.to_period('W').astype(str)
    rows = []
    for w, g in d.groupby('wk'):
        r = g['ret'].values
        rows.append((w, len(r), float(r.mean()),
                     float(np.prod(1.0 + 0.1 * r) - 1.0),
                     float(0.1 * r.sum()),
                     float(r.min()), float(r.max())))
    wk = pd.DataFrame(rows, columns=['wk','n','mean','comp','simp','worst','best'])
    span_w = (d['dt'].max() - d['dt'].min()).days / 7.0
    log(f'\n【{tag}】{len(df):,} 笔  跨度 {span_w:.0f} 周  有交易周 {len(wk)}')
    log(f'  周均笔数            {wk["n"].mean():>8.2f}')
    log(f'  周收益 简单(10%仓)   均值 {wk["simp"].mean():>+7.2%}  中位 {wk["simp"].median():>+7.2%}')
    log(f'  周收益 复利(10%仓)   均值 {wk["comp"].mean():>+7.2%}  中位 {wk["comp"].median():>+7.2%}')
    log(f'  最好周 简单 {wk["simp"].max():>+7.2%}   最差周 简单 {wk["simp"].min():>+7.2%}')
    log(f'  亏损周占比 {float((wk["simp"] < 0).mean()):>7.1%}')
    log(f'  单周笔数 中位 {wk["n"].median():>5.0f}  P90 {np.quantile(wk["n"],0.9):>5.0f}  最多 {wk["n"].max():>5.0f}')
    log(f'\n  最近 12 周明细：')
    log(f'    {"周":<12}{"笔":>5}{"简单":>10}{"复利":>10}')
    for _, r in wk.tail(12).iterrows():
        log(f'    {r["wk"]:<12}{int(r["n"]):>5}{r["simp"]:>+10.2%}{r["comp"]:>+10.2%}')
    return wk


sub_test = tr[tem & (tr['v240'].values >= th)]
weekly(sub_test, '测试段 v240>=p90（严格留出）')

sub_all = tr[tr['v240'].values >= th]
weekly(sub_all, '全样本 v240>=p90')

log('\n【判读】')
log('  · 简单(10%仓) = 每周每笔收益 × 10% 仓位后直接相加（不复利）')
log('  · 复利(10%仓) = 每笔 10% 仓位、收益滚入本金')
log('  · 亏损周占比 越低越好；周均笔数决定「一周能下几次注」')

out = os.path.join(OUT, 'weekly_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
