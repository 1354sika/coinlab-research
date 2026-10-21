# -*- coding: utf-8 -*-
"""gambler_dyntp · 验证「止盈/止损不该定死，应随周期实时变化」

两种动态方案，与固定 TP=2%/SL=1% 对比：
  A 波动率动态：TP = 2% × (vstd/中位vstd), SL = 1% × 同比例
  B 预测空间动态：用回归模型预测未来 60 分钟最大涨幅/最大跌幅，
     TP = α × 预测涨幅, SL = β × 预测跌幅（网格搜索 α,β）
回测口径：用未来 60 分钟真实最高/最低价判断先触谁（双触保守算止损，超时按收盘价）。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
STEP = 100
FEE = 0.0004   # maker 双边

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_dyntp · 动态止盈/止损 vs 固定 2%/1%')
log('=' * 100)

XS, FMAX, FMIN, FCL, VSTD, TS = [], [], [], [], [], []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        fcl = c[HOLD:][:m]
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60, r240 = ret(1), ret(5), ret(15), ret(60), ret(240)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,volr,tbr,trr,vstd,hour])
        up = fmax / cc - 1
        dn = fmin / cc - 1
        ok = np.isfinite(F).all(axis=1) & np.isfinite(up) & np.isfinite(dn)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        FMAX.append(up[idx]); FMIN.append(dn[idx]); FCL.append(fcl[idx] / cc[idx] - 1)
        VSTD.append(vstd[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); fmax = np.concatenate(FMAX); fmin = np.concatenate(FMIN)
fcl = np.concatenate(FCL); vstd = np.concatenate(VSTD); t = np.concatenate(TS)
del XS, FMAX, FMIN, FCL, VSTD, TS; gc.collect()
log(f'样本 {len(X):,}  未来60分最大涨 中位 {np.nanmedian(fmax):.2%}  最大跌 中位 {np.nanmedian(fmin):.2%}')

from sklearn.ensemble import GradientBoostingRegressor
cut = np.median(t)
tr = t <= cut; te = t > cut
log(f'训练 {tr.sum():,}  测试 {te.sum():,}')

t_fit = time.time()
reg_up = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=7, subsample=0.8)
reg_dn = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=7, subsample=0.8)
reg_up.fit(X[tr], fmax[tr])
reg_dn.fit(X[tr], -fmin[tr])   # 预测下跌幅度（正数）
log(f'回归训练 {time.time()-t_fit:.0f}s')
pu = np.maximum(reg_up.predict(X[te]), 0.001)
pd_ = np.maximum(reg_dn.predict(X[te]), 0.001)
log(f'预测：涨幅 中位 {np.median(pu):.2%}  跌幅 中位 {np.median(pd_):.2%}')

# 真实值
fu = fmax[te]; fd = -fmin[te]; fc = fcl[te]
vs = vstd[te]

def backtest(tp, sl):
    """向量化回测：tp/sl 可为标量或与样本等长数组。双触保守算止损，都不触按 60 分收盘价。"""
    tp = np.broadcast_to(np.asarray(tp, dtype='float64'), fu.shape)
    sl = np.broadcast_to(np.asarray(sl, dtype='float64'), fd.shape)
    hit_tp = fu >= tp
    hit_sl = fd >= sl
    win = hit_tp & ~hit_sl
    lose = hit_sl
    timeout = ~hit_tp & ~hit_sl
    ret = np.where(win, tp, np.where(lose, -sl, fc))
    ret = ret - FEE
    return len(ret), win.mean(), ret.mean(), np.median(ret)

log('\n【固定基准】')
n, wr, m_, med = backtest(0.02, 0.01)
log(f'  TP=2% SL=1%   n={n:,}  胜率={wr:.1%}  均值={m_:.3%}  中位={med:.3%}')

log('\n【A 波动率动态】TP=2%×k, SL=1%×k（k=vstd/中位vstd，截断 0.5~3）')
vm = np.median(vs[np.isfinite(vs)])
for lo, hi in [(0.5, 2.0), (0.7, 1.5)]:
    k = np.clip(vs / vm, lo, hi)
    n, wr, m_, med = backtest(0.02 * k, 0.01 * k)
    log(f'  k∈[{lo},{hi}]  n={n:,}  胜率={wr:.1%}  均值={m_:.3%}  中位={med:.3%}')

log('\n【B 预测空间动态】TP=α×预测涨幅, SL=β×预测跌幅')
log(f'{"α":>5s}{"β":>6s}{"胜率":>8s}{"均值":>9s}{"中位":>9s}')
log('-' * 50)
best = None
for a in [0.5, 0.7, 1.0]:
    for b in [0.5, 0.7, 1.0]:
        tp = np.clip(a * pu, 0.005, 0.08)
        sl = np.clip(b * pd_, 0.0025, 0.04)
        n, wr, m_, med = backtest(tp, sl)
        log(f'{a:>5.1f}{b:>6.1f}{wr:>8.1%}{m_:>9.3%}{med:>9.3%}')
        if best is None or m_ > best[2]:
            best = (a, b, m_, wr, med)

log(f'\n【最佳动态】α={best[0]} β={best[1]}  均值={best[2]:.3%}  胜率={best[3]:.1%}  中位={best[4]:.3%}')

log('\n【对照：同一批样本上，固定 TP 2% / SL 1% 的均值】')
n, wr, m0, med0 = backtest(0.02, 0.01)
log(f'  固定 {m0:.3%}  vs  最佳动态 {best[2]:.3%}  → 差 {(best[2]-m0)*100:+.3f}pp')

log('\n【判读】')
log('  · 动态均值若高于固定 ≥0.05pp ⇒ 自适应有意义，值得做实时版本')
log('  · 若接近 ⇒ 收益由「选对时刻」决定，改 TP/SL 只是换皮')

out = os.path.join(OUT, 'dyntp_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
