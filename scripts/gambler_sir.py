# -*- coding: utf-8 -*-
"""gambler_sir.py — 高等数学第一问：这个市场到底偏离随机游走多少？

用户判据（2026-10-17）：
    「我们需要引入高等数学的角度观察和思考……我们漏了高等数学的角度，
      也没有用高等数学的角度，把已验证的结论融合。」

在堆任何特征之前，必须先回答一个前置问题：
    **「局」是真实存在的动力学过程，还是我们把噪声看成了故事？**

两个标准检验（不依赖任何机器学习、不需要标注）：

【检验一】方差比检验 Variance Ratio（Lo & MacKinlay 1988）
    VR(q) = Var(q 期收益) / (q × Var(1 期收益))
        随机游走 ⇒ VR = 1
        趋势/动量（正自相关）⇒ VR > 1     ← 若「局」存在，局内应显著 >1
        均值回归 ⇒ VR < 1
    统计量 z = (VR-1) / sqrt(2(2q-1)(q-1) / (3qN))，|z| > 1.96 显著
    → 分别算「局内（低点→认定点）」与「局外（等长随机窗口）」的 VR

【检验二】累计成交额曲线是 S 型还是直线
    S 型（logistic 拟合 R² 明显高于线性）⇒ 存在传染机制（人传人），是 SIR 的物理证据
    线性 ⇒ 只是资金匀速流入，没有传染

这两条若都指"局内显著偏离随机、且成交额呈 S 型"，则「操盘局」有数学依据；
若都指向噪声，则此前所有"识别局"的努力都是在拟合噪声——必须承认并改向。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
BAR_MIN = 5
MAX_BARS = int(48 * 60 / BAR_MIN)
TRIG = 0.30
R = 0.10
QS = [6, 12, 24, 48, 96]

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_sir · 局内是否显著偏离随机游走（方差比检验）+ 成交额是否 S 型')
log('=' * 112)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'读入 {len(RAW)} 币   耗时 {time.time()-t0:.0f}s')

def vr_test(x, q):
    """x: 对数价格序列；返回 (VR, z)"""
    n = len(x)
    if n < q * 4:
        return np.nan, np.nan
    r = np.diff(x)                      # 1 期收益
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (len(r) - 1)
    if var1 <= 0:
        return np.nan, np.nan
    # q 期重叠收益
    rq = x[q:] - x[:-q]
    mq = q * mu
    varq = ((rq - mq) ** 2).sum() / (len(rq) - 1)
    VR = varq / (q * var1)
    N = len(r)
    theta = 2 * (2 * q - 1) * (q - 1) / (3.0 * q * N)
    z = (VR - 1) / np.sqrt(theta) if theta > 0 else np.nan
    return VR, z

def fit_s(x):
    """累计成交额曲线：比较 logistic(线性化) 与直线拟合的 R²
       logistic: y = L/(1+exp(-k(t-t0))) —— 用最小二乘网格搜 k,t0"""
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    t = np.arange(n, dtype='float64')
    y = np.cumsum(x)
    y = y / max(y[-1], 1e-9)
    # 线性
    A = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2_lin = 1 - ss_res / max(ss_tot, 1e-12)
    # logistic 网格
    best = -np.inf
    for k in (2, 4, 6, 8, 12, 16, 24, 32):
        for t0f in (0.2, 0.35, 0.5, 0.65, 0.8):
            t0 = n * t0f
            lg = 1.0 / (1.0 + np.exp(-k * (t - t0) / n))
            Al = np.vstack([lg, np.ones(n)]).T
            c, *_ = np.linalg.lstsq(Al, y, rcond=None)
            p = Al @ c
            sr = ((y - p) ** 2).sum()
            r2 = 1 - sr / max(ss_tot, 1e-12)
            if r2 > best:
                best = r2
    return r2_lin, best

recs = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < 600:
        continue
    lg = np.log(np.maximum(c, 1e-12))
    i = 0
    spans = []
    while i < n - 1:
        lo_idx = i; lo = l[i]; conf = -1; peak = -np.inf
        j = i; done = False
        while j < n - 1:
            j += 1
            if conf < 0:
                if l[j] < lo:
                    lo = l[j]; lo_idx = j
                if j - lo_idx > MAX_BARS:
                    i = lo_idx + 1; done = True; break
                if h[j] >= lo * (1 + TRIG):
                    conf = j; peak = h[j]
            else:
                if h[j] > peak:
                    peak = h[j]
                if j - conf > MAX_BARS or l[j] <= peak * (1 - R):
                    gain = peak / lo - 1
                    spans.append((lo_idx, j))
                    if conf - lo_idx >= 48:
                        seg = lg[lo_idx:conf + 1]
                        vrs = [vr_test(seg, qq) for qq in QS]
                        r2l, r2s = fit_s(q[lo_idx:conf + 1])
                        recs.append(dict(sym=sym, gain=gain, bars=conf - lo_idx,
                                         **{f'VR{k}': v for k, (v, z) in zip(QS, vrs)},
                                         **{f'z{k}': z for k, (v, z) in zip(QS, vrs)},
                                         r2_lin=r2l, r2_log=r2s))
                    i = j + 1; done = True; break
        if done:
            continue
        break
    # 局外对照：在该币非周期区间随机取等长窗口
    if len(spans) >= 2:
        mask = np.zeros(n, bool)
        for a, b in spans:
            mask[a:b + 1] = True
        free = np.flatnonzero(~mask)
        rng = np.random.default_rng(7)
        for _ in range(len(spans) * 2):
            if len(free) < 200:
                break
            st = int(rng.choice(free[:len(free) - 200]))
            L = int(np.clip(rng.choice([b - a for a, b in spans]), 48, 1200))
            if st + L >= n:
                continue
            seg = lg[st:st + L]
            vrs = [vr_test(seg, qq) for qq in QS]
            r2l, r2s = fit_s(q[st:st + L])
            recs.append(dict(sym=sym, gain=np.nan, bars=L,
                             **{f'VR{k}': v for k, (v, z) in zip(QS, vrs)},
                             **{f'z{k}': z for k, (v, z) in zip(QS, vrs)},
                             r2_lin=r2l, r2_log=r2s))
del RAW; gc.collect()

DF = pd.DataFrame(recs)
DF['inside'] = DF.gain.notna()
DF.to_csv(os.path.join(OUT, 'vr_test.csv'), index=False, encoding='utf-8-sig')
log(f'样本：局内 {int(DF.inside.sum()):,}   局外对照 {int((~DF.inside).sum()):,}   耗时 {time.time()-t0:.0f}s')

log('\n' + '=' * 112)
log('【检验一】方差比 VR(q)：随机游走 = 1.00；>1 趋势；<1 均值回归')
log('=' * 112)
log(f'  {"q(根)":>6}{"局内 VR":>10}{"局内 z":>9}{"局外 VR":>10}{"局外 z":>9}{"局内显著>1占比":>14}')
for qq in QS:
    a = DF[DF.inside][f'VR{qq}'].dropna(); b = DF[~DF.inside][f'VR{qq}'].dropna()
    za = DF[DF.inside][f'z{qq}'].dropna(); zb = DF[~DF.inside][f'z{qq}'].dropna()
    log(f'  {qq:>6}{a.median():>10.3f}{za.median():>9.2f}{b.median():>10.3f}{zb.median():>9.2f}'
        f'{100*(za > 1.96).mean():>13.1f}%')

log('\n' + '=' * 112)
log('【检验二】累计成交额曲线：线性拟合 R² vs S 型(logistic) R²')
log('=' * 112)
for lbl, d in (('局内（低点→认定）', DF[DF.inside]), ('局外对照', DF[~DF.inside])):
    a = d['r2_lin'].dropna(); b = d['r2_log'].dropna()
    k = min(len(a), len(b))
    if k == 0:
        continue
    diff = (b.values[:k] - a.values[:k])
    log(f'  {lbl:<18} 线性 R² 中位 {a.median():.4f}   S 型 R² 中位 {b.median():.4f}   '
        f'S 型更好者占 {100*(diff > 0.02).mean():.1f}%   平均提升 {diff.mean():+.4f}')

log('\n' + '=' * 112)
log('【判读】')
log('=' * 112)
log('  · 局内 VR 显著 >1 且明显高于局外 ⇒ 局内确实存在趋势性（「局」不是纯噪声）')
log('  · 若局内 VR ≈ 局外 VR ≈ 1 ⇒ 价格在 5m 尺度上接近随机游走，"识别局"是拟合噪声')
log('  · S 型显著优于线性 ⇒ 存在人传人的传染机制（SIR 的前提成立）')

out = os.path.join(OUT, 'sir_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
