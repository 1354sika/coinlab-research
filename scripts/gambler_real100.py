# -*- coding: utf-8 -*-
"""gambler_real100 · 本金 100 的真实模拟：前兆模型筛 P>=0.5 后才下注

口径（全部无前视）：
  1. 训练段 = 时间前 50%，用单币启动前兆特征训练 GBDT，预测「这笔会不会到 +30%」
  2. 测试段 = 时间后 50%，对每个「从低点涨 3%」的启动点算概率 P
  3. 只做 P >= 0.5 的启动（筛掉 38% 的假启动）
  4. 每笔仓位 = 当前资金 × 10%（复利），逐笔按时间顺序结算
  5. 止盈 lo*1.30 / 止损 lo*0.95 / 超时 24h

输出：本金 100 起，最终资金、最大回撤、每月明细、交易笔数、胜率
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
START = 100.0; POS = 0.10
FEE_SLIP = 0.0050   # 每笔总成本：taker 0.10%×2 + 滑点 0.15%×2 = 0.50%
FEE_SLIP_LO = 0.0014  # 低档：maker 0.02%×2 + 滑点 0.05%×2

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_real100 · 本金 100 的真实模拟（前兆筛选 P>=0.5）')
log('=' * 100)
log(f'入场 lo×{1+TH:.2f} / 止盈 lo×{1+TPM:.2f} / 止损 lo×{SLM:.2f} / 超时 {HOLD//12}h  仓位 {POS:.0%}')
log(f'每笔成本：高档 {FEE_SLIP:.2%}（taker 0.10%×2 + 滑点 0.15%×2）/ 低档 {FEE_SLIP_LO:.2%}（maker 0.02%×2 + 滑点 0.05%×2）')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}
rows = []
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
            tp_px = lo_px * (1.0 + TPM)
            sl_px = lo_px * SLM
            ret = None; is_tp = 0
            for k in range(j + 1, min(j + HOLD + 1, n)):
                hit_tp = h[k] >= tp_px
                hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl:
                    ret = sl_px / entry - 1; is_tp = 0; break
                if hit_tp:
                    ret = tp_px / entry - 1; is_tp = 1; break
                if hit_sl:
                    ret = sl_px / entry - 1; is_tp = 0; break
            if ret is None:
                kk = min(j + HOLD, n - 1)
                ret = c[kk] / entry - 1; is_tp = 0
            dt = pd.to_datetime(ot[j], unit='ms', utc=True)
            h3 = np.mean(hist_hits[-3:]) if len(hist_hits) >= 1 else -1.0
            h10 = np.mean(hist_hits[-10:]) if len(hist_hits) >= 1 else -1.0
            rows.append((ot[j], sym, float(volr[j]), float(tbr[j]), float(v240[j]),
                         float(entry / lo_px), float(j - lo_i),
                         float(r1[j]) if np.isfinite(r1[j]) else 0.0,
                         float(c[j] / c[max(j-12, 0)] - 1) if j >= 12 else 0.0,
                         float(c[j] / c[max(j-60, 0)] - 1) if j >= 60 else 0.0,
                         float(dt.hour), float(dt.dayofweek), float(h3), float(h10),
                         float(ret), float(is_tp)))
            hist_hits.append(is_tp)
    del ch; gc.collect()

cols = ['t','sym','volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10','ret','y']
df = pd.DataFrame(rows, columns=cols).sort_values('t').reset_index(drop=True)
log(f'\n入场点 {len(df):,}  币 {df.sym.nunique()}  整体 TP 率 {df.y.mean():.1%}')

X = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10']
cut = df['t'].values[int(len(df) * 0.5)]
tr = df[df.t <= cut]; te = df[df.t > cut].copy()
log(f'训练 {len(tr):,}  测试 {len(te):,}')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
mdl.fit(tr[X].fillna(-1), tr['y'])
te['p'] = mdl.predict_proba(te[X].fillna(-1))[:, 1]
log(f'测试段 AUC = {roc_auc_score(te["y"], te["p"]):.4f}')

for th in [0.3, 0.5, 0.6, 0.7]:
    sub = te[te.p >= th]
    if len(sub) == 0: continue
    log(f'\n【P >= {th}】{len(sub):,} 笔  TP 率 {sub.y.mean():.1%}  均值 {sub.ret.mean():+.2%}')

# ---- 本金 100 逐笔复利 ----
sub = te[te.p >= 0.5].sort_values('t').reset_index(drop=True)
cap = START
peak = START
maxdd = 0.0
curve = []
for _, r in sub.iterrows():
    eq = cap * POS
    pnl = eq * (r['ret'] - FEE_SLIP)
    cap += pnl
    peak = max(peak, cap)
    dd = (peak - cap) / peak
    maxdd = max(maxdd, dd)
    curve.append((r['t'], cap, pnl))
cv = pd.DataFrame(curve, columns=['t','cap','pnl'])
cv['dt'] = pd.to_datetime(cv['t'], unit='ms', utc=True).dt.tz_localize(None)
cv['mon'] = cv['dt'].dt.to_period('M').astype(str)

log('\n' + '=' * 100)
log('【本金 100 逐笔复利 · P>=0.5】')
log('=' * 100)
log(f'  交易笔数 {len(sub):,}   胜率 {float((sub.ret>0).mean()):.1%}   单笔均值(扣前) {sub.ret.mean():+.2%}  单笔均值(扣后) {(sub.ret.mean()-FEE_SLIP):+.2%}')
log(f'  起始 {START:.0f} 元  →  最终 {cap:,.2f} 元   （{cap/START:.2f}x）')
log(f'  最大回撤 {maxdd:.2%}')
log(f'  最差单笔 {sub.ret.min():+.1%}   最好单笔 {sub.ret.max():+.1%}')

log('\n  按月明细：')
log(f'    {"月":<10}{"笔":>5}{"月末资金":>14}{"月收益":>10}')
prev = START
for mon, g in cv.groupby('mon'):
    end = g['cap'].iloc[-1]
    log(f'    {mon:<10}{len(g):>5}{end:>14,.2f}{(end/prev-1):>+10.2%}')
    prev = end

log('\n【对照：P>=0.7】')
sub7 = te[te.p >= 0.7].sort_values('t').reset_index(drop=True)
cap7 = START; peak7 = START; dd7 = 0.0
for _, r in sub7.iterrows():
    pnl = cap7 * POS * (r['ret'] - FEE_SLIP); cap7 += pnl
    peak7 = max(peak7, cap7); dd7 = max(dd7, (peak7-cap7)/peak7)
log(f'  笔数 {len(sub7):,}  最终 {cap7:,.2f} 元（{cap7/START:.2f}x）  最大回撤 {dd7:.2%}')

log('\n【对照：不做筛选，全部启动都做】')
suba = te.sort_values('t').reset_index(drop=True)
capa = START; peaka = START; dda = 0.0
for _, r in suba.iterrows():
    pnl = capa * POS * (r['ret'] - FEE_SLIP); capa += pnl
    peaka = max(peaka, capa); dda = max(dda, (peaka-capa)/peaka)
log(f'  笔数 {len(suba):,}  最终 {capa:,.2f} 元（{capa/START:.2f}x）  最大回撤 {dda:.2%}')

log('\n【判读】')
log('  · 这是「筛过再下注」的第一次闭环：训练段选模型 → 测试段只做高概率 → 逐笔复利')
log('  · 已扣每笔 taker 0.10%×2 + 滑点 0.15%×2 = 0.50%（高档）')
log('  · 每笔 10% 仓位；超时按收盘价出')

out = os.path.join(OUT, 'real100_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
