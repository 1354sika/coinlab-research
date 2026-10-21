# -*- coding: utf-8 -*-
"""gambler_crowd.py — 散户情绪的可量化投影（用户设想 2026-10-16）
用户观点：散户的追涨/杀跌心理会造成跟风与恐慌，这些都能被量化；我此前把它当"另一条负和的线"太简单了。
做法：从 5m 数据构造 8 个"散户情绪"指标，然后对比：
  A 全样本分布（常态）
  B 局启动前 12 根（1h）/ 48 根（4h）的分布
  C 哪个指标对"局即将启动"最有区分度（单变量 AUC）
数据字段：quote_volume（成交额）、trades（笔数）、taker_quote（主动买额）—— 后两个是刻画"谁在买"的关键
"""
import os, gc, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_crowd.txt')
GAIN = 0.20
WMAX = 288
LEADS = (12, 48)      # 局启动前 1h / 4h

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_crowd · 散户情绪的可量化投影（追涨/杀跌/跟风/恐慌）')
log('=' * 100)

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

NAMES = ['taker_buy_ratio', 'avg_trade_size', 'trade_intensity', 'vol_intensity',
         'vol_price_div', 'run_same_dir', 'vola_jump', 'fomo', 'panic', 'divergence']
rows = []          # 全样本指标值（抽样）
leads = {k: [] for k in LEADS}

for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low',
                                    'quote_volume', 'trades', 'taker_quote'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32', 'taker_quote': 'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        n = len(g)
        if n < 2000:
            continue
        c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64'); q = g['quote_volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64'); tq = g['taker_quote'].values.astype('float64')

        tbr = np.where(q > 0, tq / np.maximum(q, 1e-12), np.nan)          # 主动买占比
        ats = np.where(tr > 0, q / np.maximum(tr, 1e-12), np.nan)          # 平均单笔额
        ti = np.where(rmean(tr, 48) > 0, tr / np.maximum(rmean(tr, 48), 1e-12), np.nan)
        vi = np.where(rmean(q, 48) > 0, q / np.maximum(rmean(q, 48), 1e-12), np.nan)
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        vpd = vi / (1.0 + np.abs(np.nan_to_num(r1, nan=0.0)) * 100)        # 放量不涨
        sg = np.sign(np.nan_to_num(r1, nan=0.0))
        run = np.zeros(n)
        cur = 0
        for i in range(n):
            cur = (cur + 1) if (sg[i] != 0 and (i == 0 or sg[i] == sg[i - 1])) else (1 if sg[i] != 0 else 0)
            run[i] = cur
        rng = h - l
        atr = rmean(rng / np.maximum(c, 1e-12) * 100, 48)
        vola = (rng / np.maximum(c, 1e-12) * 100) / np.maximum(atr, 1e-9)
        fomo = np.nan_to_num(tbr, nan=0.5) * np.nan_to_num(vi, nan=1.0)
        panic = (1.0 - np.nan_to_num(tbr, nan=0.5)) * np.nan_to_num(vi, nan=1.0)
        div = np.nan_to_num(tbr, nan=0.5) - 0.5

        M = np.stack([tbr, ats, ti, vi, vpd, run, vola, fomo, panic, div], axis=1)
        ok = ~np.isnan(M).any(axis=1)
        samp = np.flatnonzero(ok)[::24]
        if len(samp):
            rows.append(M[samp].astype('float32'))

        # 局启动点
        W = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
        is_low = l <= W.min(axis=1) + 1e-12
        hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
        cand = np.flatnonzero(is_low & np.isfinite(hmax))
        last = -10 ** 9
        for t0 in cand:
            if t0 - last < WMAX:
                continue
            H = hmax[t0]; base = max(l[t0], 1e-12)
            if H / base - 1.0 < GAIN:
                continue
            i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
            thr = base + (H - base) * 0.5
            k2 = min(n - 1, i1 + 96)
            if not (k2 > i1 and l[i1:k2 + 1].min() <= thr):
                continue
            for L in LEADS:
                if t0 - L >= 0:
                    v = M[t0 - 1]           # 用启动前一根（严格无前视）
                    if not np.isnan(v).any():
                        leads[L].append(v.astype('float32'))
            last = i1
    del ch; gc.collect()
    if len(rows) % 20 == 0:
        log(f'  已处理  全样本块 {len(rows)}  耗时 {time.time()-t0t:.0f}s')

A = np.concatenate(rows) if rows else np.zeros((0, len(NAMES)))
log(f'全样本 {len(A):,}  局 {sum(len(v) for v in leads.values())}  耗时 {time.time()-t0t:.0f}s')

log('')
log('=' * 100)
log('【A】指标对照：全样本（常态） vs 局启动前 1h / 4h')
log('=' * 100)
log(f'{"指标":20s} {"全样本中位":>12s} {"前24h中位":>12s} {"前1h中位":>12s} {"倍数(前1h/常态)":>16s}')
for i, nm in enumerate(NAMES):
    base = np.nanmedian(A[:, i])
    v48 = np.nanmedian(np.array(leads[48])[:, i]) if leads[48] else np.nan
    v12 = np.nanmedian(np.array(leads[12])[:, i]) if leads[12] else np.nan
    ratio = v12 / base if (base and np.isfinite(base) and base != 0) else np.nan
    log(f'{nm:20s} {base:>12.4f} {v48:>12.4f} {v12:>12.4f} {ratio:>15.2f}x')

log('')
log('=' * 100)
log('【B】单变量区分度：用"局启动前一根"的指标值判断该币是否将进局（AUC）')
log('=' * 100)
CUT = 0.70
# 构造对照：前一根指标 vs 是否在随后 288 根内进局
from sklearn.metrics import roc_auc_score
for i, nm in enumerate(NAMES):
    # 简单做法：局前样本 vs 全样本 的 5 分位对比（AUC 近似）
    pos = np.array(leads[12])[:, i] if leads[12] else None
    if pos is None or len(pos) < 50:
        continue
    neg = A[:, i]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    ok = np.isfinite(s)
    if ok.sum() < 100:
        continue
    auc = roc_auc_score(y[ok], s[ok])
    log(f'  {nm:20s} 单变量 AUC={auc:.4f}')

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
