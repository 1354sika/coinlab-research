# -*- coding: utf-8 -*-
"""gambler_days · 能不能提前看出「今天有行情」

背景：daily 结果显示 399 天只有 135 天有信号（33.8%），周期成簇。
问题：这 135 天能不能事先识别？如果能，就可以「只在行情日入场、空仓日休息」。

方法：
  1. 读 5m，按 UTC 日聚合全市场状态（收盘价 → 日收益、波动、上涨币占比）
  2. 标签：当天是否有入场信号（>=1 笔，用 daily 同一口径）
  3. 特征：只用「当天 0 点之前」的信息（滞后 1 天，避免前视）
  4. GBDT：训练=前 70% 天，测试=后 30% 天；输出 AUC + 分档实际命中率
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
log('gambler_days · 能不能提前看出「今天有行情」')
log('=' * 100)
log(f'周期 {len(cyc):,}  币 {cyc.sym.nunique()}')

need = set(cyc.sym.unique())
by_sym = {s: g for s, g in cyc.groupby('sym')}

sig_days = set()          # 有信号的日子
day_px = {}               # day -> list of (sym, close_at_day_start)
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
            sig_days.add(pd.to_datetime(ot[j], unit='ms', utc=True).date())
        # 记录每个 UTC 日 0 点附近的收盘价（用于市场状态）
        days = pd.to_datetime(ot, unit='ms', utc=True).date
        last = {}
        for k in range(n):
            last[days[k]] = c[k]
        for d, px in last.items():
            day_px.setdefault(d, []).append(px)
    del ch; gc.collect()

log(f'\n有信号的日子 {len(sig_days)}')
if len(day_px) == 0:
    log('无数据'); raise SystemExit

ds = sorted(day_px.keys())
# 构造每日市场状态（用当天最后价格 → 日收益；滞后 1 天当特征）
mkt = []
prev_px = None
for d in ds:
    px = np.array(day_px[d], dtype='float64')
    px = px[np.isfinite(px) & (px > 0)]
    if len(px) < 10:
        mkt.append(None); continue
    cur = np.median(px)
    if prev_px is None or len(prev_px) != len(px):
        mkt.append(None)
    else:
        r = (px / prev_px - 1.0)
        mkt.append(dict(med=float(np.median(r)), up=float((r > 0).mean()),
                        std=float(np.std(r)), n=len(px)))
    prev_px = px

# 特征：前 1/2/3/7 天的市场状态 + 当天是否有信号（标签）
rows = []
vals = [(d, m) for d, m in zip(ds, mkt) if m is not None]
for idx in range(7, len(vals)):
    d, m = vals[idx]
    feats = {}
    for lag in [1, 2, 3, 7]:
        mm = vals[idx - lag][1]
        feats[f'med_{lag}'] = mm['med']
        feats[f'up_{lag}'] = mm['up']
        feats[f'std_{lag}'] = mm['std']
    # 过去 7 天有信号的次数
    # 只用「昨天及更早」的信号历史（不含当天，避免标签泄漏）
    feats['sig7'] = float(sum(1 for k in range(idx - 7, idx) if vals[k][0] in sig_days))
    feats['sig3'] = float(sum(1 for k in range(idx - 3, idx) if vals[k][0] in sig_days))
    feats['dow'] = float(pd.Timestamp(d).dayofweek)
    rows.append((d, feats, 1.0 if d in sig_days else 0.0))

df = pd.DataFrame([dict(day=r[0], y=r[2], **r[1]) for r in rows])
X = [c for c in df.columns if c not in ('day','y')]
log(f'\n可用天数 {len(df)}   有信号天 {int(df.y.sum())} ({df.y.mean():.1%})')

cut = int(len(df) * 0.7)
tr, te = df.iloc[:cut], df.iloc[cut:]
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=7)
mdl.fit(tr[X], tr['y'])
p = mdl.predict_proba(te[X])[:, 1]
auc = roc_auc_score(te['y'], p) if te['y'].nunique() > 1 else float('nan')
log(f'\nGBDT 测试段 AUC = {auc:.4f}   基准有信号率 {te.y.mean():.1%}')
log(f'\n【按预测概率分档】')
for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
    m = (p >= lo) & (p < hi)
    if m.sum() == 0: continue
    log(f'  P∈[{lo:.1f},{hi:.1f})  n={int(m.sum()):>4}  实际有信号 {te.y.values[m].mean():>6.1%}')

log('\n【特征重要性 Top8】')
for f, v in sorted(zip(X, mdl.feature_importances_), key=lambda x: -x[1])[:8]:
    log(f'  {f:8s} {v:.3f}')

log('\n【判读】')
log('  · AUC > 0.65 ⇒ 「哪几天有行情」可以提前看出')
log('  · AUC ≈ 0.5 ⇒ 看不出，只能事后知道「那天有信号」')
log('  · sig7/sig3 已改为「不含当天」，若 AUC 仍高才是真的可提前看出')

out = os.path.join(OUT, 'days_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
