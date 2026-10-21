# -*- coding: utf-8 -*-
"""gambler_regime_live.py — 把「局的数学指纹」变成实时指标，看能否早期报出

由来（job-24708-30）：
    用户质疑「不是没有局，是你分辨局的方式错了」——**他说对了**。
    用六种"形状类"指标重测（局内 891 段 vs 局外 175 段），全部显著：
        skew 偏度    0.822 vs 0.013   **66×**    p=9e-37
        cusum 变点   0.0059 vs 0.0022 **2.67×**  p=1e-68
        amp 振幅     0.0076 vs 0.0037 **2.06×**  p=5e-60
        tail3 右尾   0.0096 vs 0.0066  1.46×    p=8e-17
        kurt 峰度    8.46 vs 6.73      1.26×    p=6e-06
        hurst 成交额 0.896 vs 0.857    1.05×    p=1e-07
    ⇒ 局真实存在；旧 VR 检验（二阶矩、全段平均）看不见，是因为尺子选错了。

【本脚本】把这些整段统计量改成**每根 K 线可算的滚动窗口指标**（W=96 根=8h），
用它们预测「未来 H 小时内会不会起局」，并与旧尺子（滚动 VR）对照。

标签：局的起点 = 相对 288 根低点首次涨 ≥30% 的那一根
切分：前 70% 时间训练 / 后 30% 测试（无前视）
输出：AUC、阈值表、报出时已涨幅度（越小越早期）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; TRIG = 0.30
H_LIST = [6, 12, 24]

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 118)
log('gambler_regime_live · 局的数学指纹 → 滚动实时指标 → 能否提前报出')
log('=' * 118)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T = len(ts)
t_cut = int(T * 0.70)
del C; gc.collect()
log(f'读入 {len(RAW)} 币   全局时间轴 {T} 根   训练<=第 {t_cut} 根   耗时 {time.time()-t0:.0f}s')

FN = ['rskew', 'rkurt', 'rtail3', 'ramp', 'rcusum', 'rvr', 'rvolr']
XS, TID, OV = [], [], []
YS = {h: [] for h in H_LIST}
FEAT_SAMPLES = 0

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 2:
        continue
    lg = np.log(np.maximum(c, 1e-12))
    r = np.full(n, np.nan); r[1:] = lg[1:] - lg[:-1]
    rs = pd.Series(r)
    mp = max(W // 2, 10)
    with np.errstate(invalid='ignore', divide='ignore'):
        rskew = rs.rolling(W, min_periods=mp).skew().values
        rkurt = rs.rolling(W, min_periods=mp).kurt().values
        mu = rs.rolling(W, min_periods=mp).mean().values
        sd = rs.rolling(W, min_periods=mp).std().values
        # 右尾：当期超过 (当期均值+3σ) 的比例（滚动均值近似，全向量化）
        exceed = (r > (mu + 3 * sd)).astype('float64')
        rtail3 = pd.Series(exceed).rolling(W, min_periods=mp).mean().values
    ramp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=mp).mean().values
    # CUSUM：窗口内 |累计偏离| 的最大值 / sqrt(W)
    rcusum = pd.Series(np.nan_to_num(r, nan=0.0) - np.nan_to_num(mu, nan=0.0)).cumsum().diff(W).abs() \
             .rolling(W, min_periods=mp).max().values / np.sqrt(W)
    # 旧尺子：滚动方差比（对照）
    rq = pd.Series(lg).diff(W)
    v1 = rs.rolling(W, min_periods=mp).var().values
    vq = rq.rolling(W, min_periods=mp).var().values
    with np.errstate(invalid='ignore', divide='ignore'):
        rvr = vq / (W * v1)
    qs = pd.Series(q)
    base = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = q / np.maximum(base, 1e-12)
    rvolr = pd.Series(volr).rolling(6, min_periods=3).mean().values

    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
    trig = np.nan_to_num(over, nan=-1.0) >= TRIG
    start = np.zeros(n, dtype='float32')
    prev = False
    for i in range(n):
        if trig[i] and not prev:
            start[i] = 1.0
        prev = trig[i]

    cols = [rskew, rkurt, rtail3, ramp, rcusum, rvr, rvolr]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])
    ok = (np.isfinite(rskew) & np.isfinite(rcusum) & np.isfinite(ramp)
          & np.isfinite(over) & (np.arange(n) < n - max(H_LIST) * 12 - 1))
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:
        idx = idx[::max(1, len(idx) // 4000)]
    XS.append(M[idx].astype('float32'))
    TID.append(np.searchsorted(ts, ot)[idx].astype('int64'))
    OV.append(over[idx].astype('float32'))
    for Hh in H_LIST:
        k = int(Hh * 12)
        fut = pd.Series(start).rolling(k, min_periods=1).max().shift(-(k - 1)).values
        YS[Hh].append(np.nan_to_num(fut[idx], nan=0.0).astype('int8'))
    FEAT_SAMPLES += len(idx)
del RAW; gc.collect()

X = np.vstack(XS); TID = np.concatenate(TID); OV = np.concatenate(OV)
Y = {k: np.concatenate(v) for k, v in YS.items()}
del XS
tr = TID < t_cut; te = ~tr
log(f'样本 {len(X):,}   训练 {int(tr.sum()):,}   测试 {int(te.sum()):,}   耗时 {time.time()-t0:.0f}s')

log('\n' + '=' * 118)
log('【结果】未来 H 小时内是否会起局（局的起点 = 从 288 根低点首次涨 ≥30%）')
log('=' * 118)
for Hh in H_LIST:
    y = Y[Hh]
    log(f'\n  H = {Hh} 小时   训练正例 {100*y[tr].mean():.3f}%   测试正例 {100*y[te].mean():.3f}%')
    if y[tr].sum() < 30 or y[te].sum() < 10:
        log('    正例过少，跳过')
        continue
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4,
                                         min_samples_leaf=100, random_state=7)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    log(f'    AUC = {roc_auc_score(y[te], p):.4f}')
    log(f'    {"阈值":>6}{"报出":>9}{"覆盖":>9}{"精确率":>9}{"召回":>8}{"报出时已涨中位":>14}')
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        m = p >= thr
        if m.sum() == 0:
            log(f'    {thr:>6.1f}{"无报出":>9}'); continue
        log(f'    {thr:>6.1f}{int(m.sum()):>9,}{100*m.mean():>8.2f}%{100*y[te][m].mean():>8.2f}%'
            f'{100*m[y[te] == 1].sum()/max((y[te] == 1).sum(), 1):>7.2f}%'
            f'{100*np.median(OV[te][m]):>13.2f}%')

# 置换重要性
log('\n' + '=' * 118)
log('【特征重要性】置换重要性（H=24，测试段）')
log('=' * 118)
Hh = 24
y = Y[Hh]
clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4,
                                     min_samples_leaf=100, random_state=7)
clf.fit(X[tr], y[tr])
base = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
rng = np.random.default_rng(7)
imp = []
for j, nm in enumerate(FN):
    Xt = X[te].copy(); perm = rng.permutation(len(Xt))
    Xt[:, j] = Xt[perm, j]
    imp.append((nm, base - roc_auc_score(y[te], clf.predict_proba(Xt)[:, 1])))
imp.sort(key=lambda t: -t[1])
for nm, v in imp:
    log(f'  {nm:<10}{v:>+10.4f}')

out = os.path.join(OUT, 'regime_live_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
