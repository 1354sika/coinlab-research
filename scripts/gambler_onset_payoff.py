# -*- coding: utf-8 -*-
"""gambler_onset_payoff.py — 精确率只有 6.6%，那命中的能涨多少？（赔率决定成败）

由来（job-24708-43 gambler_onset_rule）：
    在"还没起涨（over<10%）"的样本里，最好的筛选是 amp≥3 & volr≥5：
    测试段 273 个信号，精确率 6.59%（基准 0.342%，lift 19.3 倍），每天 0.8 次。
    但离"能交易"的 25% 还差 4 倍。

【关键问题】精确率低不要紧，只要命中的赔率高：
    若 6.6% 的命中平均能涨 100%+，那 6.6%×100% 就可能盖住 93.4% 的小亏。

本脚本测三种筛选下的赔率结构：
    S1 amp≥3 & volr≥5          （精确率最高、最稀少）
    S2 amp≥3                   （每天 2.7 次）
    S3 amp≥2 & volr≥5          （每天 2.1 次）
    Z  随机（对照）
输出：后续 30 天最大涨幅分布、达到 +50%/+100%/+200% 的比例、
      以及在"不止损、持有 30 天""移动止损回落 30%"两种退出下的期望净。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; FEE = 0.0009
FUT = 8640      # 30 天 = 8640 根

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_onset_payoff · 择时信号命中后的赔率结构')
log('=' * 112)

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
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
del C; gc.collect()

SIG = {'S1': [], 'S2': [], 'S3': [], 'Z': []}
rng = np.random.default_rng(7)
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4 + FUT + 2:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    ampb = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    qb = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_ratio = amp / np.maximum(ampb, 1e-12)
        volr = q / np.maximum(qb, 1e-12)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    base_ok = (np.nan_to_num(over, nan=1.0) < 0.10) & np.isfinite(amp_ratio) & \
              np.isfinite(volr24) & (np.arange(n) < n - FUT - 2) & (np.arange(n) > WDAY * 31)
    masks = {
        'S1': base_ok & (amp_ratio >= 3.0) & (volr24 >= 5.0),
        'S2': base_ok & (amp_ratio >= 3.0),
        'S3': base_ok & (amp_ratio >= 2.0) & (volr24 >= 5.0),
    }
    for tag, m in masks.items():
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        if len(idx) > 1500:
            idx = idx[::max(1, len(idx)//1500)]
        wh = np.lib.stride_tricks.sliding_window_view(h, FUT)[idx]
        wl = np.lib.stride_tricks.sliding_window_view(l, FUT)[idx]
        wc = np.lib.stride_tricks.sliding_window_view(c, FUT)[idx]
        c0 = c[idx]
        peak = wh.max(1) / c0 - 1.0
        cummax = np.maximum.accumulate(wh, axis=1)
        dd = wc / np.maximum(cummax, 1e-12) - 1.0
        trig = dd <= -0.30
        it = np.minimum(np.argmax(trig, axis=1), FUT - 1)
        has = trig.any(1)
        pxt = np.take_along_axis(wl, it[:, None], axis=1)[:, 0]
        trail = np.where(has, pxt / c0 - 1.0, wc[:, -1] / c0 - 1.0) - FEE
        hold = wc[:, -1] / c0 - 1.0 - FEE
        tpos = np.searchsorted(ts, ot)[idx]
        te = tpos >= t_cut
        SIG[tag].append(dict(peak=peak[te], hold=hold[te], trail=trail[te], tpos=tpos[te]))
        del wh, wl, wc
    idx_r = rng.integers(WDAY*31, n - FUT - 2, size=min(800, n // 100))
    wh = np.lib.stride_tricks.sliding_window_view(h, FUT)[idx_r]
    wl = np.lib.stride_tricks.sliding_window_view(l, FUT)[idx_r]
    wc = np.lib.stride_tricks.sliding_window_view(c, FUT)[idx_r]
    c0 = c[idx_r]
    peak = wh.max(1) / c0 - 1.0
    cummax = np.maximum.accumulate(wh, axis=1)
    dd = wc / np.maximum(cummax, 1e-12) - 1.0
    trig = dd <= -0.30
    it = np.minimum(np.argmax(trig, axis=1), FUT - 1)
    has = trig.any(1)
    pxt = np.take_along_axis(wl, it[:, None], axis=1)[:, 0]
    trail = np.where(has, pxt / c0 - 1.0, wc[:, -1] / c0 - 1.0) - FEE
    hold = wc[:, -1] / c0 - 1.0 - FEE
    tpos = np.searchsorted(ts, ot)[idx_r]
    te = tpos >= t_cut
    SIG['Z'].append(dict(peak=peak[te], hold=hold[te], trail=trail[te], tpos=tpos[te]))
    del amp, ampb, volr, volr24, wh, wl, wc
del RAW; gc.collect()

log('\n' + '=' * 112)
log('【赔率结构】测试段，入场后 30 天内')
log('=' * 112)
log(f'  {"信号":<8}{"笔数":>8}{"峰值涨中位%":>13}{"p90%":>9}{"≥+50%":>8}{"≥+100%":>9}{"≥+200%":>9}'
    f'{"持有30天%":>11}{"移动止损30%%":>13}')
for tag, name in (('S1', 'amp3&vol5'), ('S2', 'amp3'), ('S3', 'amp2&vol5'), ('Z', '随机')):
    if not SIG[tag]:
        continue
    d = {k: np.concatenate([x[k] for x in SIG[tag]]) for k in ('peak', 'hold', 'trail')}
    n = len(d['peak'])
    if n < 20:
        continue
    log(f'  {name:<10}{n:>8,}{100*np.median(d["peak"]):>13.1f}'
        f'{100*np.percentile(d["peak"],90):>9.1f}'
        f'{100*np.mean(d["peak"]>=0.5):>8.1f}{100*np.mean(d["peak"]>=1.0):>9.1f}'
        f'{100*np.mean(d["peak"]>=2.0):>9.1f}'
        f'{100*np.mean(d["hold"]):>+11.2f}{100*np.mean(d["trail"]):>+13.2f}')

log('\n' + '=' * 112)
log('【判读】')
log('=' * 112)
log('  · 若 S1 的峰值涨中位 ≫ 随机，说明低精确率但高赔率，可能仍有价值')
log('  · 若两种退出期望都为负，说明赔率也救不了')

out = os.path.join(OUT, 'onset_payoff_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
