# -*- coding: utf-8 -*-
"""gambler_overlay · E1/E2：A1/A2 + ETH 15分钟跌≥1% 叠加
口径：做空 TP2%/SL1%，60 分钟窗口，双触算止损，赢率=w/n（含未触发按0）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 50; FEE = 0.0004
TP, SL = 0.02, 0.01

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_overlay · E1/E2：A1/A2 + ETH 15分钟跌≥1%')
log('=' * 100)

leader = {}
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32'},
                      chunksize=3_000_000):
    for sym in ('BTCUSDT','ETHUSDT'):
        g = ch[ch['symbol'] == sym]
        if len(g):
            g = g.sort_values('open_time')
            ot = g['open_time'].values.astype('int64')
            cl = g['close'].values.astype('float64')
            if sym in leader:
                leader[sym] = (np.concatenate([leader[sym][0], ot]), np.concatenate([leader[sym][1], cl]))
            else:
                leader[sym] = (ot, cl)
    del ch; gc.collect()
eth_ot, eth_cl = leader['ETHUSDT']
eth_r = np.full(len(eth_cl), np.nan)
eth_r[15:] = eth_cl[15:] / eth_cl[:-15] - 1.0
ETH = dict(zip(eth_ot.tolist(), eth_r.tolist()))
log(f'ETH {len(eth_ot):,} 分钟')

FS, FMX, FMN, TS, ER = [], [], [], [], []
sym_id = {}
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym in ('BTCUSDT','ETHUSDT') or len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r15 = ret(1), ret(15)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r15, volr, tbr, vstd, vstd1440, hour])
        fmx_rel = fmax / cc - 1.0
        fmn_rel = fmin / cc - 1.0
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmx_rel) & np.isfinite(fmn_rel)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        er = np.array([ETH.get(int(k), np.nan) for k in ot[idx]], dtype='float64')
        FS.append(F[idx].astype('float32'))
        FMX.append(fmx_rel[idx].astype('float32'))
        FMN.append(fmn_rel[idx].astype('float32'))
        TS.append(ot[idx]); ER.append(er)
    del ch; gc.collect()

X = np.vstack(FS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN)
t = np.concatenate(TS); er = np.concatenate(ER)
del FS, FMX, FMN, TS, ER; gc.collect()
r15, volr, tbr, vstd, vstd1440, hour = [X[:,i] for i in range(6)]
log(f'样本 {len(X):,}  币 {len(sym_id)}')

cut = np.median(t); trm = t <= cut; tem = t > cut
P90v = float(np.nanpercentile(vstd[trm], 90))
P50v = float(np.nanpercentile(vstd[trm], 50))
P90_1440 = float(np.nanpercentile(vstd1440[trm], 90))
log(f'训练段 p90(vstd)={P90v:.5f}  p50(vstd)={P50v:.5f}  p90(vstd1440)={P90_1440:.5f}')

W = (fmn <= -TP) & (fmx < SL)
L = (fmx >= SL)
A1 = (r15 > 0.02) & (volr < 0.5) & (vstd > P90v) & (tbr > 0.55)
A2 = (vstd < P50v) & (vstd1440 > P90_1440) & (tbr > 0.7) & (hour >= 0) & (hour < 6)
ETHD = er <= -0.01

log(f'\nETH 条件可用样本（有 er 值）：{np.isfinite(er).sum():,} / {len(er):,}')
log(f'ETH <= -1% 占比：{(np.isfinite(er) & ETHD).mean():.2%}')

def stat(mask, name):
    n = int(mask.sum())
    if n == 0:
        log(f'{name:28s} n=0'); return
    w = int((W & mask).sum()); l = int((L & mask).sum())
    wr = w/n; trig = (w+l)/n
    net = (w*TP - l*SL)/n - FEE
    # 每天次数（测试段约 28 天，STEP=50 采样；全量折算 ×50）
    days = (t[tem].max() - t[tem].min()) / 86400000.0
    per_day = n/days
    log(f'{name:28s} n={n:>6,}  赢率 {wr:>6.1%}  触发率 {trig:>6.1%}  净 {net:>+7.3%}  每天 {per_day:>6.2f} 次(采样)')

log('\n' + '=' * 100)
log('【测试段】E1/E2 叠加对比')
log('=' * 100)
stat(tem & A1, 'A1 单独')
stat(tem & A1 & ETHD, 'A1 + ETH跌1%')
stat(tem & A1 & ~ETHD, 'A1 但 ETH未跌1%')
stat(tem & A2, 'A2 单独')
stat(tem & A2 & ETHD, 'A2 + ETH跌1%')
stat(tem & A2 & ~ETHD, 'A2 但 ETH未跌1%')
stat(tem & ETHD, 'ETH跌1% 单独(全部币)')

log('\n' + '=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若 A1+ETH 净期望 > A1 单独且 n>=100 ⇒ 跨币过滤器有效')
log('  · 若样本坍缩到 n<100 ⇒ 叠加后机会太少，不值')
log('  · 每天次数为 STEP=50 采样口径，全量约 ×50')

out = os.path.join(OUT, 'overlay_e1e2_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
