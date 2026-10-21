# -*- coding: utf-8 -*-
"""gambler_atr_trade.py — 识别到局之后，怎么才抓得住：波动率自适应止损 + 等回调

现状（已跑完的三步）：
    ① 局真实存在：偏度 66×、CUSUM 2.67×、振幅 2.06×（六项全 p<1e-5）
    ② 用振幅做实时指标：未来 6h 起局 **AUC 0.9262**（旧价格特征仅 0.34）
    ③ 但把它变成交易：**全负**（最好 -0.09%/笔）
    ⇒ 瓶颈从"看不见"变成"抓不住"：振幅大时上下都大，固定 -15% 止损被打掉。

【本脚本】改两处，看能否把"看见"变成"赚到"：
    改① 止损：固定 15% → **波动率自适应 k×ATR96**（k=2/3/4）
    改② 入场：信号触发即买 → **等回调 r% 再买**（r=0 / 5%）

信号：amp_ratio = 8h 平均振幅 / 过去 30 天平均振幅 ≥ K（1.5/2/3）
     且 过去 24h 涨幅 < 30%（还没拉完）
退出：ATR 止损 / +50% 止盈 / 移动止损回落 30% / 5 天超时
扣费 0.09%；前 70% 训练、后 30% 测试；报每日出手次数
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; FEE = 0.0009
HOLD = 1440            # 5 天
TP = 0.50
TRAIL = 0.30

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_atr_trade · 波动率自适应止损 + 等回调')
log('=' * 116)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
del C; gc.collect()
log(f'币 {N}   全局 {T}   耗时 {time.time()-t0:.0f}s')

KS = [1.5, 2.0, 3.0]
KATR = [2.0, 3.0]
CALLBACK = [0.0, 0.05]
rows = []

for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < WDAY * 4 + HOLD + 2:
        continue
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    base = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        ar = amp / np.maximum(base, 1e-12)
    r24 = np.full(n, np.nan); r24[288:] = c[288:] / np.maximum(c[:-288], 1e-12) - 1.0
    atr = amp.copy()                       # 用"平均振幅"作为 ATR 的代理（占价格比例）
    for K in KS:
        sig = (np.isfinite(ar) & (ar >= K)
               & (np.nan_to_num(r24, nan=0.0) < 0.30)
               & (np.arange(n) < n - HOLD - 2))
        idx0 = np.flatnonzero(sig)
        if len(idx0) == 0:
            continue
        if len(idx0) > 1500:
            idx0 = idx0[::max(1, len(idx0)//1500)]
        for cb in CALLBACK:
            # 等回调：信号后 cb 比例的回落才入场（用后续窗口内的最低点近似）
            if cb > 0:
                # 取信号后 48 根内的最低点作为入场价；若没回调到位则不入场
                idx = []
                ent = []
                for k in idx0:
                    seg = l[k+1:k+49]
                    if len(seg) == 0:
                        continue
                    thr = c[k] * (1 - cb)
                    if seg.min() <= thr:
                        j = k + 1 + int(np.argmax(seg <= thr))
                        idx.append(j); ent.append(thr)
                idx = np.array(idx, dtype='int64'); ent = np.array(ent, dtype='float64')
            else:
                idx = idx0; ent = c[idx0]
            if len(idx) < 30:
                continue
            idx = idx[idx < n - HOLD - 2]
            ent = ent[:len(idx)]
            if len(idx) < 30:
                continue
            wh = np.lib.stride_tricks.sliding_window_view(h, HOLD)[idx].astype('float32')
            wl = np.lib.stride_tricks.sliding_window_view(l, HOLD)[idx].astype('float32')
            wc = np.lib.stride_tricks.sliding_window_view(c, HOLD)[idx].astype('float32')
            a0 = np.nan_to_num(atr[idx], nan=0.02)
            for ka in KATR:
                sl_frac = np.clip(ka * a0, 0.05, 0.60)
                sl_px = ent * (1 - sl_frac)
                hit_sl = wl <= sl_px[:, None]
                hit_tp = wh >= (ent * (1 + TP))[:, None]
                it = np.argmax(hit_tp, 1); il = np.argmax(hit_sl, 1)
                any_tp = hit_tp.any(1); any_sl = hit_sl.any(1)
                v = np.where(any_sl & (~any_tp | (il <= it)), -sl_frac,
                             np.where(any_tp, TP, wc[:, -1] / ent - 1.0)) - FEE
                tpos = np.searchsorted(ts, ot)[idx]
                for q in range(len(idx)):
                    rows.append((sym, int(tpos[q]), K, cb, ka, float(v[q])))
            del wh, wl, wc
    del amp, base, ar, atr
del RAW; gc.collect()

DF = pd.DataFrame(rows, columns=['sym', 'tpos', 'K', 'cb', 'ka', 'net'])
log(f'样本 {len(DF):,}   币 {DF.sym.nunique()}   耗时 {time.time()-t0:.0f}s')
te = DF.tpos.values >= t_cut
days = (DF.tpos.max() - DF.tpos.min()) / 86_400_000

log('\n' + '=' * 116)
log('【结果】测试段（后 30%）')
log('=' * 116)
log(f'  {"K":>5}{"回调":>6}{"ATR×k":>7}{"笔数":>8}{"期望净%":>10}{"中位%":>9}'
    f'{"胜率%":>8}{"≥+50%":>8}{"≥+100%":>9}{"每日":>8}')
best = []
for K in KS:
    for cb in CALLBACK:
        for ka in KATR:
            d = DF[(DF.K == K) & (DF.cb == cb) & (DF.ka == ka) & te]
            if len(d) < 50:
                continue
            v = d.net.values
            e = float(np.nanmean(v))
            best.append((K, cb, ka, len(d), e))
            log(f'  {K:>5.1f}{cb*100:>5.0f}%{ka:>7.1f}{len(v):>8,}{100*e:>+10.2f}'
                f'{100*np.nanmedian(v):>9.2f}{100*np.nanmean(v > 0):>8.1f}'
                f'{100*np.nanmean(v >= 0.5):>8.1f}{100*np.nanmean(v >= 1.0):>9.1f}'
                f'{len(v)/max(days*0.3,1):>8.1f}')

log('\n' + '=' * 116)
log('【对比基线】固定 -15% 止损（上一版，job-24708-33）为 -0.09 ~ -2.65%/笔')
log('=' * 116)
if best:
    best.sort(key=lambda x: -x[4])
    K, cb, ka, n, e = best[0]
    log(f'  最优格：K={K} 回调={cb*100:.0f}% ATR×{ka}  →  {n:,} 笔  期望净 {100*e:+.2f}%')
    log(f'  结论：{"波动率自适应止损有效" if e > 0 else "仍为负 —— 止损方式不是瓶颈"}')

out = os.path.join(OUT, 'atr_trade_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
