# -*- coding: utf-8 -*-
"""gambler_merge · 两条线合并：大周期（趋势）+ 小周期（波动）

大周期（资产 A + B）：
  入场：从低点涨 3%，且前兆模型 P >= 0.5
  止盈：lo*1.30（+26.2% 相对入场）   止损：lo*0.95（-7.8%）   超时：24h

小周期（新线，5m 粒度近似）：
  触发：单根 5m 跌幅 >= 5% 且量比 >= 4（放量急跌）
  做多，止盈 +2%   止损 -1%   超时 60 根（5 小时）

资金：每条线各 50 元起步（合计 100），每笔仓位 = 该线当前资金 × 10%
成本：每笔 0.50%（taker 0.10%×2 + 滑点 0.15%×2）
输出：两条线各自结果 + 合并资金曲线 + 每月明细
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
CYC = os.path.join(OUT, 'cycles.csv')
SRC = os.path.join(D, 'klines_5m_3y.csv')

TH = 0.03; TPM = 0.30; SLM = 0.95; HOLD = 288
S_R1 = -0.05; S_VOLR = 4.0; S_TP = 0.02; S_SL = 0.01; S_HOLD = 60
COST = 0.0050
START_EACH = 50.0; POS = 0.10

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_merge · 大周期（趋势）+ 小周期（波动）合并')
log('=' * 100)
log(f'大周期：涨 3% + P>=0.5 / +26.2% / -7.8% / 24h / 仓位 {POS:.0%}')
log(f'小周期：5m 跌 {S_R1:.0%} + 量比>={S_VOLR:.0f} / +{S_TP:.0%} / -{S_SL:.0%} / {S_HOLD*5}min / 仓位 {POS:.0%}')
log(f'成本：每笔 {COST:.2%}   各线起步 {START_EACH:.0f} 元')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}

big = []    # (t, ret, sym)
small = []  # (t, ret, sym)
feat = []   # 大周期特征（训练前兆模型用）
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','volume','quote_volume','taker_quote','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float64','taker_quote':'float64'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in need or len(g) < 5000:
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
        v240 = pd.Series(r1).rolling(240, min_periods=50).std().values
        volr = vol / pd.Series(vol).rolling(240, min_periods=50).mean().values
        tbr = np.where(qv > 0, tq / np.maximum(qv, 1e-12), np.nan)
        hist_hits = []

        # ---------- 大周期 ----------
        for _, row in by_sym[sym].iterrows():
            lo_i = int(row['lo_i']); lo_px = float(row['lo'])
            if lo_i >= n - 2: continue
            j = lo_i
            target = lo_px * (1.0 + TH)
            while j < n - 1 and c[j] < target:
                j += 1
            if j >= n - 1 or j - lo_i > HOLD: continue
            if not np.isfinite(v240[j]) or not np.isfinite(volr[j]) or not np.isfinite(tbr[j]):
                continue
            entry = c[j]
            tp_px = lo_px * (1.0 + TPM); sl_px = lo_px * SLM
            ret = None; is_tp = 0
            for k in range(j + 1, min(j + HOLD + 1, n)):
                hit_tp = h[k] >= tp_px; hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl: ret = sl_px / entry - 1; break
                if hit_tp: ret = tp_px / entry - 1; is_tp = 1; break
                if hit_sl: ret = sl_px / entry - 1; break
            if ret is None:
                kk = min(j + HOLD, n - 1); ret = c[kk] / entry - 1
            dt = pd.to_datetime(ot[j], unit='ms', utc=True)
            h3 = np.mean(hist_hits[-3:]) if hist_hits else -1.0
            h10 = np.mean(hist_hits[-10:]) if hist_hits else -1.0
            feat.append((ot[j], sym, float(volr[j]), float(tbr[j]), float(v240[j]),
                         float(entry / lo_px), float(j - lo_i),
                         float(r1[j]) if np.isfinite(r1[j]) else 0.0,
                         float(c[j] / c[max(j-12, 0)] - 1) if j >= 12 else 0.0,
                         float(c[j] / c[max(j-60, 0)] - 1) if j >= 60 else 0.0,
                         float(dt.hour), float(dt.dayofweek), float(h3), float(h10),
                         float(ret), float(is_tp)))
            hist_hits.append(is_tp)

        # ---------- 小周期 ----------
        for j in range(241, n - 1):
            if not np.isfinite(r1[j]) or not np.isfinite(volr[j]): continue
            if r1[j] > S_R1 or volr[j] < S_VOLR: continue
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
            small.append((ot[j], ret, sym))
    del ch; gc.collect()

cols = ['t','sym','volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10','ret','y']
df = pd.DataFrame([r[:16] for r in feat], columns=cols).sort_values('t').reset_index(drop=True)
log(f'\n大周期入场点 {len(df):,}   小周期触发 {len(small):,}')

X = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10']
cut = df['t'].values[int(len(df) * 0.5)]
tr = df[df.t <= cut]; te = df[df.t > cut].copy()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
mdl.fit(tr[X].fillna(-1), tr['y'])
te['p'] = mdl.predict_proba(te[X].fillna(-1))[:, 1]
log(f'前兆模型 AUC = {roc_auc_score(te["y"], te["p"]):.4f}')

big_sel = te[te.p >= 0.5][['t','ret']].values.tolist()
sm = pd.DataFrame(small, columns=['t','ret','sym'])
tm = te['t'].min()
sm = sm[sm.t > tm][['t','ret']].values.tolist()
log(f'测试段：大周期 {len(big_sel):,} 笔   小周期 {len(sm):,} 笔')

def run_line(trades, name):
    trades = sorted(trades, key=lambda x: x[0])
    cap = START_EACH; peak = START_EACH; mdd = 0.0
    wins = 0; rows = []
    for t, r in trades:
        pnl = cap * POS * (r - COST)
        cap += pnl
        if (r - COST) > 0: wins += 1
        peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak)
        rows.append((t, cap))
    log(f'\n【{name}】{len(trades):,} 笔  胜率 {wins/max(len(trades),1):.1%}  '
        f'{START_EACH:.0f} → {cap:,.2f} 元（{cap/START_EACH:.1f}x）  最大回撤 {mdd:.2%}')
    return pd.DataFrame(rows, columns=['t','cap']), cap, mdd

cv_b, cap_b, dd_b = run_line(big_sel, '大周期（P>=0.5）')
cv_s, cap_s, dd_s = run_line(sm, '小周期（跌5%放量）')

# 合并：按时间轴加总两条线资金
if len(cv_b) and len(cv_s):
    allt = sorted(set(cv_b.t.tolist()) | set(cv_s.t.tolist()))
    cb = cv_b.set_index('t')['cap'].reindex(allt).ffill().fillna(START_EACH).values
    cs = cv_s.set_index('t')['cap'].reindex(allt).ffill().fillna(START_EACH).values
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
log('  · 两条线各 50 元、每笔 10% 仓位、每笔扣 0.50%')
log('  · 小周期是 5m 粒度近似（原 1m 版频率更高），触发数会偏少')
log('  · 合并回撤若低于任一条线 ⇒ 两条线性互补（这才是「混合」的意义）')

out = os.path.join(OUT, 'merge_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
