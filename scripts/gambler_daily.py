# -*- coding: utf-8 -*-
"""gambler_daily · 新参数按天聚合：能不能做到每天都有机会

参数（wide 网格最优格）：
  入场：从低点涨 3%（lo*1.03）
  止盈：lo*1.30（相对入场 +26.2%）
  止损：lo*0.95（相对入场 -7.8%）
  过滤：v240 >= 训练段 p50
  超时：24 小时

输出：总天数、有交易天数、占比、日笔数分布、日收益（固定 10% 仓位）、亏损日占比
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

TH = 0.03; TPM = 0.30; SLM = 0.95; VP = 0.50; HOLD = 288

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_daily · 新参数能否做到每天都有机会')
log('=' * 100)
log(f'参数：入场 lo×{1+TH:.2f} / 止盈 lo×{1+TPM:.2f} / 止损 lo×{SLM:.2f} / v240>=p{int(VP*100)} / 超时 {HOLD//12}h')
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
rows = []
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
            target = lo_px * (1.0 + TH)
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1 or j - lo_i > HOLD: continue
            if not np.isfinite(v240[j]): continue
            entry = c[j]
            tp_px = lo_px * (1.0 + TPM)
            sl_px = lo_px * SLM
            ret = None; reason = 'timeout'
            for k in range(j + 1, min(j + HOLD + 1, n)):
                hit_tp = h[k] >= tp_px
                hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl:
                    ret = sl_px / entry - 1; reason = 'both→SL'; break
                if hit_tp:
                    ret = tp_px / entry - 1; reason = 'TP'; break
                if hit_sl:
                    ret = sl_px / entry - 1; reason = 'SL'; break
            if ret is None:
                kk = min(j + HOLD, n - 1)
                ret = c[kk] / entry - 1
            rows.append((ot[j], ret, reason, v240[j], sym))
    del ch; gc.collect()

tr = pd.DataFrame(rows, columns=['t','ret','reason','v240','sym'])
tr = tr.sort_values('t').reset_index(drop=True)
log(f'\n入场点总数 {len(tr):,}')

cut = np.median(tr['t'].values)
trm = tr['t'].values <= cut
tem = tr['t'].values > cut
th_v = np.quantile(tr.loc[trm, 'v240'], VP)
sub = tr[tem & (tr['v240'].values >= th_v)].copy()
log(f'测试段入场 {len(sub):,} 笔（v240>=p{int(VP*100)} 阈值 {th_v:.5f}）')

if len(sub) == 0:
    log('无样本'); raise SystemExit

sub['dt'] = pd.to_datetime(sub['t'], unit='ms', utc=True).dt.tz_localize(None)
sub['day'] = sub['dt'].dt.date
all_days = pd.date_range(sub['dt'].min().date(), sub['dt'].max().date(), freq='D')
drows = []
for d, g in sub.groupby('day'):
    r = g['ret'].values
    drows.append((d, len(r), float(0.1 * r.sum()), float(r.mean())))
dy = pd.DataFrame(drows, columns=['day','n','simp','mean'])
log(f'\n跨度 {len(all_days)} 天   有交易 {len(dy)} 天（{len(dy)/len(all_days):.1%}）')
log(f'  日笔数   中位 {dy["n"].median():>5.0f}  P10 {np.quantile(dy["n"],0.1):>5.0f}  P90 {np.quantile(dy["n"],0.9):>5.0f}  最多 {dy["n"].max():>5.0f}')
log(f'  日收益   中位 {dy["simp"].median():>+8.2%}  均值 {dy["simp"].mean():>+8.2%}')
log(f'  亏损日占比 {float((dy["simp"]<0).mean()):>7.1%}   最差日 {dy["simp"].min():>+8.2%}   最好日 {dy["simp"].max():>+8.2%}')

log(f'\n  【出场构成】')
log(f'    止盈 {(sub.reason=="TP").mean():>6.1%}  止损 {(sub.reason=="SL").mean():>6.1%}  '
    f'双触 {(sub.reason=="both→SL").mean():>5.1%}  超时 {(sub.reason=="timeout").mean():>6.1%}')
r = sub['ret'].values
log(f'    每笔均值 {r.mean():>+7.2%}  中位 {np.median(r):>+7.2%}  最差 {r.min():>+7.1%}')

log(f'\n  最近 20 天明细：')
log(f'    {"日":<12}{"笔":>5}{"日收益(10%仓)":>14}{"均笔":>9}')
for _, x in dy.tail(20).iterrows():
    log(f'    {str(x["day"]):<12}{int(x["n"]):>5}{x["simp"]:>+14.2%}{x["mean"]:>+9.2%}')

log('\n【判读】')
log('  · 有交易天占比接近 100% ⇒ 每天都有机会')
log('  · 日笔数 P10 >= 1 ⇒ 九成的日子至少有一笔')
log('  · 若日笔数中位 < 1 或 P10 = 0 ⇒ 做不到每天，需要另一套模式')

out = os.path.join(OUT, 'daily_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
