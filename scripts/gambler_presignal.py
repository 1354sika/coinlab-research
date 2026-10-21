# -*- coding: utf-8 -*-
"""gambler_presignal · 单币启动前兆：涨 3% 那一刻，能不能预判「这一笔会不会到 +30%」

背景：daily 证明「哪天有行情」预测不准（AUC 0.608）。
本脚本换粒度：不看「日子」，看「单个币的一次启动」——
在「从低点涨 3%」入场那一刻，用它自己的分钟特征预测：
  这一笔会到 +30% 止盈（TP），还是止损/超时（非 TP）？

特征（全部在入场根可知，无前视）：
  volr    量比：当前 5m volume / 过去 240 根均值
  tbr     taker 买占比：taker_quote / quote_volume
  v240    波动率：近 240 根 5m 收益率标准差
  pos     位置：entry / lo（离低点多远）
  lo_age  低点到入场经过多少根 5m
  r1/r15/r60  入场前 5m/1h/3h 涨跌幅
  hour/dow    小时、星期
  hist3   最近 3 次启动的 TP 命中率（按时间顺序，滚动，不含本次）
  hist10  最近 10 次启动的 TP 命中率

输出：GBDT AUC、分档实际命中率、特征重要性
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

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

cyc = pd.read_csv(CYC, encoding='utf-8-sig')
log('=' * 100)
log('gambler_presignal · 单币启动前兆：涨 3% 时预判会不会到 +30%')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

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
        hist_hits = []   # 该币历次启动结果（TP=1/非TP=0），按时间
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
            is_tp = 0
            for k in range(j + 1, min(j + HOLD + 1, n)):
                hit_tp = h[k] >= tp_px
                hit_sl = l[k] <= sl_px
                if hit_tp and hit_sl:
                    is_tp = 0; break
                if hit_tp:
                    is_tp = 1; break
                if hit_sl:
                    is_tp = 0; break
            dt = pd.to_datetime(ot[j], unit='ms', utc=True)
            h3 = np.mean(hist_hits[-3:]) if len(hist_hits) >= 1 else np.nan
            h10 = np.mean(hist_hits[-10:]) if len(hist_hits) >= 1 else np.nan
            rows.append((ot[j], sym,
                         float(volr[j]), float(tbr[j]), float(v240[j]),
                         float(entry / lo_px), float(j - lo_i),
                         float(r1[j]) if np.isfinite(r1[j]) else 0.0,
                         float(c[j] / c[max(j-12, 0)] - 1) if j >= 12 else 0.0,
                         float(c[j] / c[max(j-60, 0)] - 1) if j >= 60 else 0.0,
                         float(dt.hour), float(dt.dayofweek),
                         float(h3) if np.isfinite(h3) else -1.0,
                         float(h10) if np.isfinite(h10) else -1.0,
                         float(is_tp)))
            hist_hits.append(is_tp)
    del ch; gc.collect()

cols = ['t','sym','volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10','y']
df = pd.DataFrame(rows, columns=cols)
log(f'\n入场点 {len(df):,}  币 {df.sym.nunique()}  TP 比例 {df.y.mean():.1%}')

X = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hour','dow','hist3','hist10']
cut = np.median(df['t'].values)
tr = df[df.t <= cut]; te = df[df.t > cut]
log(f'训练 {len(tr):,}  测试 {len(te):,}')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
mdl.fit(tr[X].fillna(-1), tr['y'])
p = mdl.predict_proba(te[X].fillna(-1))[:, 1]
auc = roc_auc_score(te['y'], p) if te['y'].nunique() > 1 else float('nan')
log(f'\nGBDT 测试段 AUC = {auc:.4f}   基准 TP 率 {te.y.mean():.1%}')

log('\n【按预测概率分档（测试段）】')
for lo, hi in [(0, .3), (.3, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
    m = (p >= lo) & (p < hi)
    if m.sum() == 0: continue
    log(f'  P∈[{lo:.1f},{hi:.1f})  n={int(m.sum()):>6}  实际 TP {te.y.values[m].mean():>6.1%}'
        f'  相对基准 {te.y.values[m].mean()/te.y.mean():>5.2f}x')

log('\n【单特征 AUC（测试段）】')
for x in X:
    v = te[x].fillna(-1).values
    if np.std(v) == 0: continue
    a = roc_auc_score(te['y'], v)
    log(f'  {x:8s} AUC={max(a, 1-a):.4f}   方向={"正" if a>0.5 else "反"}')

log('\n【特征重要性 Top10】')
for f, v in sorted(zip(X, mdl.feature_importances_), key=lambda z: -z[1])[:10]:
    log(f'  {f:8s} {v:.3f}')

log('\n【判读】')
log('  · AUC > 0.65 ⇒ 单币启动前兆有用，可以「只做有把握的启动」')
log('  · AUC ≈ 0.5 ⇒ 涨 3% 那一刻没有区分信息，只能全做或全不做')
log('  · hist3/hist10 高重要性 ⇒ 币的「启动成功率」有持续性（老司机 vs 骗子）')

out = os.path.join(OUT, 'presignal_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
