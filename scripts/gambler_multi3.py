# -*- coding: utf-8 -*-
"""gambler_multi3.py — 验证：46% 概率的"最危险点"是不是真落在操盘局里？
用户要求（2026-10-16）：
  "直观判断这些点是不是真的落在已知的操盘局里，这是我们的模型需要知道的"

做法：
  1) 复算最危险条件（C6 & S4 & S6 = 基线 371 个点 / 46.09%）
  2) 独立标记"已知局"（g20 口径：5 根内低点 → 288 根内最高涨>=20% → 之后自峰腰斩）
  3) 交叉验证（三个数）：
       A 精确率：371 个点里，多少落在某个局内？
       B 提前量：落在局内的那些点，出现在局的什么进度（相对峰值）？
       C 覆盖率：全部局里，有多少至少被一个危险点覆盖？
  4) 出图：时间轴 × 币种 的散点图（危险点 vs 局）
"""
import os, gc, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_multi3.txt')
PNG = os.path.join(EV, 'gambler_multi3.png')

WMAX = 288
GAIN = 0.20

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_multi3 · 最危险点 vs 已知操盘局（交叉验证）')
log('=' * 100)

cols = {}
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'}, chunksize=5_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        cols[sym] = (g['open_time'].values.astype('int64'),
                     g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                     g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                     g['trades'].values.astype('float64'))
    del ch; gc.collect()

syms = sorted(cols)
log(f'币 {len(syms)}   耗时 {time.time()-t0t:.0f}s')

# 对齐
C = pd.DataFrame({s: pd.Series(cols[s][1], index=cols[s][0]) for s in syms}).sort_index()
Q = pd.DataFrame({s: pd.Series(cols[s][4], index=cols[s][0]) for s in syms}).sort_index()
TR = pd.DataFrame({s: pd.Series(cols[s][5], index=cols[s][0]) for s in syms}).sort_index()
T, N = C.shape
ts = C.index.values
MAJOR = {'BTCUSDT', 'ETHUSDT', 'WBETHUSDT'}
tgt = np.array([i for i, s in enumerate(syms) if s not in MAJOR])

R1 = C.pct_change(12).values
Rn = np.nan_to_num(R1, nan=0.0)
r1 = np.nan_to_num(C.pct_change().values, nan=0.0)

btc = Rn[:, syms.index('BTCUSDT')]
btc_r6 = np.zeros(T); btc_r6[6:] = btc[6:] - btc[:-6]
C1 = np.repeat((btc_r6 <= -0.01)[:, None], N, axis=1)
C2 = Rn <= -0.03
v24 = Q.rolling(24, min_periods=6).mean().values
vr = np.where(v24 > 0, Q.values / np.maximum(v24, 1e-12), 1.0)
C3 = vr >= 3.0
t24 = TR.rolling(24, min_periods=6).mean().values
trr = np.where(t24 > 0, TR.values / np.maximum(t24, 1e-12), 1.0)
C4 = trr >= 2.0
dn_cnt = (Rn <= -0.03)[:, tgt].sum(axis=1)
C5 = np.repeat((dn_cnt >= 10)[:, None], N, axis=1)
vol24 = pd.DataFrame(np.abs(Rn)).rolling(24, min_periods=6).mean().values
C6 = np.abs(r1) > 3.0 * np.maximum(vol24, 1e-4)
dn_evt = (Rn <= -0.03).astype('float32')
S4 = pd.DataFrame(dn_evt).rolling(288, min_periods=24).sum().values >= 2
S6 = Rn <= -0.06
DANGER = C1 & C2 & C3 & C4 & C5 & C6 & S4 & S6
log(f'危险条件点 {int(DANGER.sum()):,}   耗时 {time.time()-t0t:.0f}s')

# 标记已知局（逐币，用完整轴索引）
IN_REG = np.zeros((T, N), dtype=bool)
REG_INFO = []      # (sym_idx, t0_idx, i1_idx)
for si, s in enumerate(syms):
    ot, c, h, l, q, tr = cols[s]
    n = len(c)
    if n < 2000:
        continue
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
        # 映射到对齐轴
        pos0 = np.searchsorted(ts, ot[t0]); pos1 = np.searchsorted(ts, ot[i1])
        if pos0 < T and pos1 < T:
            IN_REG[pos0:min(T, pos1 + 1), si] = True
            REG_INFO.append((si, pos0, pos1))
        last = i1
log(f'已知局 {len(REG_INFO):,} 个   耗时 {time.time()-t0t:.0f}s')

# 交叉验证
ii, jj = np.nonzero(DANGER)
log('')
log('=' * 100)
log('【A】精确率：危险点里有多少落在某个局内？')
log('=' * 100)
inreg = IN_REG[ii, jj]
log(f'  危险点 {len(ii):,}   落在局内 {int(inreg.sum()):,}（{100*inreg.mean():.1f}%）')
log(f'  对照：全体（币×时刻）在局内的比例 = {100*IN_REG.mean():.2f}%')

log('')
log('=' * 100)
log('【B】提前量：落在局内的危险点，出现在局的什么位置？')
log('=' * 100)
progs = []
for k in np.flatnonzero(inreg):
    t = ii[k]; s = jj[k]
    for (si, p0, p1) in REG_INFO:
        if si == s and p0 <= t <= p1:
            progs.append((t - p0) / max(p1 - p0, 1))
            break
if progs:
    P = np.array(progs)
    log(f'  样本 {len(P):,}')
    log(f'  进度中位 {100*np.median(P):.1f}%   均值 {100*P.mean():.1f}%')
    for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
        log(f'    进度 {lo:.0%}~{hi:.0%}: {int(((P>=lo)&(P<hi)).sum()):,}（{100*((P>=lo)&(P<hi)).mean():.1f}%）')

log('')
log('=' * 100)
log('【C】覆盖率：已知局里有多少至少被一个危险点覆盖？')
log('=' * 100)
hit_reg = 0
for (si, p0, p1) in REG_INFO:
    if DANGER[p0:p1 + 1, si].any():
        hit_reg += 1
log(f'  已知局 {len(REG_INFO):,}   被危险点覆盖 {hit_reg:,}（{100*hit_reg/max(len(REG_INFO),1):.1f}%）')

# 出图
log('')
log('=' * 100)
log('【D】出图')
log('=' * 100)
try:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    day = ts // 86400000
    # 已知局（灰）
    for (si, p0, p1) in REG_INFO:
        ax.plot([day[p0], day[p1]], [si, si], color='0.75', lw=0.7, zorder=1)
    # 危险点（深色散点）
    ax.scatter(day[ii], jj, s=6, c='0.15', marker='o', zorder=3, label='危险点（C6&S4&S6）')
    # 命中局内的危险点（空心圈）
    ax.scatter(day[ii[inreg]], jj[inreg], s=26, facecolors='none',
               edgecolors='black', lw=0.9, zorder=4, label='落在局内')
    ax.set_xlabel('time (days since epoch)')
    ax.set_ylabel('coin index')
    ax.set_title('gambler: danger points vs known manipulation regimes')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(PNG)
    plt.close()
    log(f'  图已存 {PNG}')
except Exception as e:
    log(f'  出图失败：{e}')

log('')
log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
