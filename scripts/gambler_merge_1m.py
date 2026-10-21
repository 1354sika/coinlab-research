# -*- coding: utf-8 -*-
"""gambler_merge_1m · 全 1m 粒度：大周期 + 小周期

单位换算（1m 一根）：
  v240      = 240 根 = 4 小时滚动波动率
  低点确认   = 2880 根 = 48 小时内涨 3%
  大周期持仓 = 1440 根 = 24 小时
  小周期窗口 = 15 根 = 15 分钟跌 5%
  小周期持仓 = 60 根 = 60 分钟

大周期：局部低点 → 48h 内涨 3% 入场 → 前兆模型 P>=0.5 → +26.2% / -7.8% / 24h
小周期：15m 跌 5% + 量比>=4 → 做多 +2% / -1% / 60min
资金：各 50 元起步，每笔 10% 仓位，每笔扣 0.50%
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_1m_502.csv')

TH = 0.03; TPM = 0.30; SLM = 0.95
HOLD_BIG = 1440; CONF_WIN = 2880
S_R15 = -0.05; S_VOLR = 4.0; S_TP = 0.02; S_SL = 0.01; S_HOLD = 60
COST = 0.0050
START_EACH = 50.0; POS = 0.10

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
need = set(cyc.sym.unique())
log('=' * 100)
log('gambler_merge_1m · 全 1m 粒度：大周期 + 小周期')
log('=' * 100)
log(f'单位：v240=240根(4h)  低点确认=2880根(48h)  大周期持仓=1440根(24h)  小周期=15根(15m)跌5%/60根(60m)')
log(f'币 {len(need)}  成本 {COST:.2%}/笔')

big_rows, small_rows = [], []
t_read = time.time()
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','volume','quote_volume','taker_quote','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float64','taker_quote':'float64'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in need or len(g) < 20000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        qv = g['quote_volume'].values.astype('float64')
        tq = g['taker_quote'].values.astype('float64')
        n = len(c)
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1
        v240 = pd.Series(r1).rolling(240, min_periods=60).std().values
        volr = vol / pd.Series(vol).rolling(240, min_periods=60).mean().values
        tbr = np.where(qv > 0, tq / np.maximum(qv, 1e-12), np.nan)
        r15 = pd.Series(c).pct_change(15).values

        # 局部低点：121 根中心窗口最低
        roll_min = pd.Series(l).rolling(121, center=True, min_periods=60).min().values
        is_lo = (l == roll_min) & np.isfinite(roll_min)
        lo_idx = np.where(is_lo)[0]

        hist_hits = []
        for lo_i in lo_idx:
            if lo_i < 120 or lo_i >= n - CONF_WIN: continue
            lo_px = float(l[lo_i])
            if lo_px <= 0: continue
            # 从低点起找涨 3%
            j = lo_i + 1
            target = lo_px * (1.0 + TH)
            lim = min(lo_i + CONF_WIN, n - 1)
            while j < lim and c[j] < target:
                j += 1
            if j >= lim or c[j] < target: continue
            if not np.isfinite(v240[j]) or not np.isfinite(volr[j]) or not np.isfinite(tbr[j]):
                continue
            entry = c[j]
            tp_px = lo_px * (1.0 + TPM); sl_px = lo_px * SLM
            ret = None; is_tp = 0
            for k in range(j + 1, min(j + HOLD_BIG + 1, n)):
                hit_tp = h[k] >= tp_px; hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl: ret = sl_px / entry - 1; break
                if hit_tp: ret = tp_px / entry - 1; is_tp = 1; break
                if hit_sl: ret = sl_px / entry - 1; break
            if ret is None:
                kk = min(j + HOLD_BIG, n - 1); ret = c[kk] / entry - 1
            dt = pd.to_datetime(ot[j], unit='ms', utc=True)
            h3 = np.mean(hist_hits[-3:]) if hist_hits else -1.0
            h10 = np.mean(hist_hits[-10:]) if hist_hits else -1.0
            big_rows.append((ot[j], sym, float(volr[j]), float(tbr[j]), float(v240[j]),
                             float(entry / lo_px), float(j - lo_i),
                             float(r1[j]) if np.isfinite(r1[j]) else 0.0,
                             float(c[j] / c[max(j-15, 0)] - 1) if j >= 15 else 0.0,
                             float(c[j] / c[max(j-60, 0)] - 1) if j >= 60 else 0.0,
                             float(dt.hour), float(dt.dayofweek), float(h3), float(h10),
                             float(ret), float(is_tp)))
            hist_hits.append(is_tp)

        # 小周期：15m 跌 5% + 放量
        for j in range(241, n - 1):
            if not np.isfinite(r15[j]) or not np.isfinite(volr[j]): continue
            if r15[j] > S_R15 or volr[j] < S_VOLR: continue
            entry = c[j]
            tp_px = entry * (1.0 + S_TP); sl_px = entry * (1.0 - S_SL)
            ret = None
            for k in range(j + 1, min(j + S_HOLD + 1, n)):
                hit_tp = h[k] >= tp_px; hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl: ret = -S_SL; break
                if hit_tp: ret = S_TP; break
                if hit_sl: ret = -S_SL; break
            if ret is None:
                kk = min(j + S_HOLD, n - 1); ret = c[kk] / entry - 1
            small_rows.append((ot[j], ret, sym))
    del ch; gc.collect()
    log(f'  已读 {time.time()-t_read:.0f}s')

cols = ['t','sym','volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10','ret','y']
df = pd.DataFrame(big_rows, columns=cols).sort_values('t').reset_index(drop=True)
sm = pd.DataFrame(small_rows, columns=['t','ret','sym']).sort_values('t').reset_index(drop=True)
log(f'\n大周期入场点 {len(df):,}   小周期触发 {len(sm):,}')

X = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10']
cut = df['t'].values[int(len(df) * 0.5)]
tr = df[df.t <= cut]; te = df[df.t > cut].copy()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
mdl.fit(tr[X].fillna(-1), tr['y'])
te['p'] = mdl.predict_proba(te[X].fillna(-1))[:, 1]
log(f'前兆模型 AUC = {roc_auc_score(te["y"], te["p"]):.4f}   TP率 {te.y.mean():.1%}')

tm = te['t'].min()
sm = sm[sm.t > tm]
big_sel = te[te.p >= 0.5][['t','ret']].values.tolist()
sm_list = sm[['t','ret']].values.tolist()
log(f'测试段：大周期(P>=0.5) {len(big_sel):,} 笔   小周期 {len(sm_list):,} 笔')

def run_line(trades, name):
    trades = sorted(trades, key=lambda x: x[0])
    cap = START_EACH; peak = START_EACH; mdd = 0.0; wins = 0; rows = []
    for t, r in trades:
        net = r - COST
        pnl = cap * POS * net
        cap += pnl
        if net > 0: wins += 1
        peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak)
        rows.append((t, cap))
    log(f'\n【{name}】{len(trades):,} 笔  胜率 {wins/max(len(trades),1):.1%}  '
        f'{START_EACH:.0f} → {cap:,.2f} 元（{cap/START_EACH:.2f}x）  最大回撤 {mdd:.2%}')
    return pd.DataFrame(rows, columns=['t','cap']), cap, mdd

cv_b, cap_b, dd_b = run_line(big_sel, '大周期（P>=0.5）')
cv_s, cap_s, dd_s = run_line(sm_list, '小周期（15m跌5%放量）')

if len(cv_b) and len(cv_s):
    allt = sorted(set(cv_b.t.tolist()) | set(cv_s.t.tolist()))
    cb = cv_b.groupby('t')['cap'].last().reindex(allt).ffill().fillna(START_EACH).values
    cs = cv_s.groupby('t')['cap'].last().reindex(allt).ffill().fillna(START_EACH).values
    tot = cb + cs
    peak = np.maximum.accumulate(tot)
    mdd = float(((peak - tot) / peak).max())
    log(f'\n【合并】起步 {2*START_EACH:.0f} 元 → 最终 {tot[-1]:,.2f} 元（{tot[-1]/(2*START_EACH):.1f}x）')
    log(f'  最大回撤 {mdd:.2%}   （大周期 {dd_b:.2%} / 小周期 {dd_s:.2%}）')
    dt = pd.to_datetime(allt, unit='ms', utc=True).tz_localize(None)
    mon = pd.Series(dt).dt.to_period('M').astype(str)
    log('\n  按月（合并资金）：')
    prev = 2 * START_EACH
    for m, g in pd.DataFrame({'m':mon,'v':tot}).groupby('m'):
        end = g['v'].iloc[-1]
        log(f'    {m:<10}{end:>16,.2f}{(end/prev-1):>+10.2%}')
        prev = end

log('\n【判读】')
log('  · 全 1m 粒度，单位换算见文件头')
log('  · 每笔扣 0.50%；各线 50 元起步、每笔 10% 仓位')
log('  · 小周期 1m 版若仍负 ⇒ 急跌放量抄底在 1m 也不成立')

out = os.path.join(OUT, 'merge_1m_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
