# -*- coding: utf-8 -*-
"""gambler_wide · 放宽周期筛选：网格对比

当前筛选（trade_sim2）：入场涨 5%、v240>=p90、止盈 +50%、止损 -9.5%、超时 24h
问题：55 周里只有 28 周有信号，一半时间空仓。

本脚本放宽三个维度，看「周期覆盖率」和「周收益」怎么变：
  入场门槛 th  ∈ {+3%, +5%, +8%}        （越低越容易触发）
  止盈     tpm ∈ {+20%, +30%, +50%}      （相对低点，越低越容易到）
  波动过滤 vp  ∈ {p50, p70, p90}        （越松越容易保留）

止损固定：相对低点 -5%（lo*0.95）；超时：24h
输出每个组合的：笔数、跨度周、有交易周占比、周均笔数、周收益中位、亏损周占比
全部按「测试段严格留出」（v240 阈值只用训练段算）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

THS = [0.03, 0.05, 0.08]
TPMS = [0.20, 0.30, 0.50]
VPS = [0.50, 0.70, 0.90]
SL_M = 0.95

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_wide · 放宽周期筛选网格')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}   网格 {len(THS)}x{len(TPMS)}x{len(VPS)}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
rows = []   # (th, tpm, t_entry, ret, reason, v240, sym)
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
            for th in THS:
                j = lo_i
                target = lo_px * (1.0 + th)
                while j < n - 1 and c[j] < target:
                    j += 1
                if j >= n - 1 or j - lo_i > 288: continue
                if not np.isfinite(v240[j]): continue
                entry = c[j]
                sl_px = lo_px * SL_M
                # 先检查止损是否已被跳空穿透（入场根之后）
                for tpm in TPMS:
                    tp_px = lo_px * (1.0 + tpm)
                    ret = None; reason = 'timeout'
                    for k in range(j + 1, min(j + 289, n)):
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
                        ret = c[kk] / entry - 1
                    rows.append((th, tpm, ot[j], ret, reason, v240[j], sym))
    del ch; gc.collect()

tr = pd.DataFrame(rows, columns=['th','tpm','t','ret','reason','v240','sym'])
log(f'\n入场点总数 {len(tr):,}')

cut = np.median(tr['t'].values)
trm = tr['t'].values <= cut
tem = tr['t'].values > cut

OUTROWS = []
for th in THS:
    for tpm in TPMS:
        m0 = (tr['th'].values == th) & (tr['tpm'].values == tpm)
        sub0 = tr[m0]
        if len(sub0) == 0: continue
        for vp in VPS:
            thr = np.quantile(sub0.loc[sub0.index.isin(sub0.index[trm[m0]]), 'v240'], vp) if trm[m0].sum() > 0 else np.quantile(sub0['v240'], vp)
            m1 = m0 & (tr['v240'].values >= thr) & tem
            sub = tr[m1]
            if len(sub) < 5: continue
            d = sub.copy()
            d['dt'] = pd.to_datetime(d['t'], unit='ms', utc=True).dt.tz_localize(None)
            d['wk'] = d['dt'].dt.to_period('W').astype(str)
            wkrows = []
            for w, g in d.groupby('wk'):
                r = g['ret'].values
                wkrows.append((w, len(r), float(0.1 * r.sum())))
            wk = pd.DataFrame(wkrows, columns=['wk','n','simp'])
            span = (d['dt'].max() - d['dt'].min()).days / 7.0
            OUTROWS.append(dict(th=th, tpm=tpm, vp=vp, n=len(sub),
                                weeks=span, active=len(wk),
                                act_ratio=len(wk)/max(span,1),
                                wk_n=wk['n'].median(),
                                wk_med=wk['simp'].median(),
                                wk_mean=wk['simp'].mean(),
                                neg=float((wk['simp'] < 0).mean()),
                                worst=wk['simp'].min()))

res = pd.DataFrame(OUTROWS).sort_values(['act_ratio','wk_med'], ascending=[False, False])
log('\n' + '=' * 110)
log('【网格结果】按「有交易周占比」降序，周收益中位为固定 10% 仓位')
log('=' * 110)
log(f'{"入场":>5}{"止盈":>6}{"波动滤":>6}{"笔数":>8}{"跨度周":>7}{"有交易周":>8}{"占比":>7}{"周笔数":>7}{"周中位":>9}{"周均值":>9}{"亏损周":>7}{"最差周":>9}')
for _, r in res.iterrows():
    log(f'{r["th"]:>6.0%}{r["tpm"]:>7.0%}{"p"+str(int(r["vp"]*100)):>7}{r["n"]:>8.0f}{r["weeks"]:>7.0f}'
        f'{r["active"]:>8.0f}{r["act_ratio"]:>7.0%}{r["wk_n"]:>7.0f}{r["wk_med"]:>+9.2%}{r["wk_mean"]:>+9.2%}'
        f'{r["neg"]:>7.1%}{r["worst"]:>+9.2%}')

log('\n【判读】')
log('  · 有交易周占比 = 有信号的周 / 数据跨度周；越接近 100% 说明越「每周都有机会」')
log('  · 周中位 = 有交易周的收益中位（固定 10% 仓位）')
log('  · 若放宽后笔数大增但周中位转负 ⇒ 放进来的是噪声，不是周期')

res.to_csv(os.path.join(OUT, 'wide_grid.csv'), index=False, encoding='utf-8-sig')
out = os.path.join(OUT, 'wide_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
