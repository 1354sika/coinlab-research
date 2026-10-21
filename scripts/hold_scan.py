# -*- coding: utf-8 -*-
"""持有期扫描：拉长时间跨度能不能放大收益？「一夜翻十倍」有没有可能？

固定信号（急跌5% + 量比>3 + 多币>=2 + 币自身7天跌超20%不进场），只改持有期 H：
  1小时 / 4小时 / 12小时 / 24小时(一夜) / 3天 / 7天 / 30天
每档给出：单笔净收益、胜率、最大单笔、最小单笔、K=3复利终值、回撤、分年正负

判据：
  · 若单笔收益随 H 增长 -> 拉长有道理（固定成本被摊薄）
  · 若单笔收益不增或下降 -> 拉长无解
  · 「一夜翻十倍」= 单笔 +900%；看实测最大值离它多远
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_5m_3y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

DEP, VOLR, CONCUR = 0.05, 3.0, 2
FEE_M = 0.0002
D7 = 288 * 7
CAP = 1000.0
HOLDS = [(12, '1小时'), (48, '4小时'), (144, '12小时'), (288, '24小时(一夜)'),
         (864, '3天'), (2016, '7天'), (8640, '30天')]

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float64', 'quote_volume': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P, QV = px.values, qv.values
T, N = P.shape
tms = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60
s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
coin_7d = P / pd.DataFrame(P).shift(D7).values - 1.0

CV = volr > VOLR
CD = streak_ret < -DEP
conc = (CD & CV).sum(axis=1)[:, None] * np.ones((1, N))
SIGBASE = CD & CV & (conc >= CONCUR)


def dedup_idx(S, hold):
    bi, bj = [], []
    last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            js = np.where(sel)[0]
            bi.append(np.full(len(js), i)); bj.append(js)
            last[js] = i
    if not bi:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(bi), np.concatenate(bj)


def compound(ii, jj, net, K=3):
    o = np.argsort(ii, kind='stable')
    ii, net = ii[o], net[o]
    slot = np.full(K, CAP / K); free = np.zeros(K, np.int64)
    hist = [CAP]; taken = 0
    for t in range(len(net)):
        i = ii[t]
        av = np.where(free <= i)[0]
        if len(av) == 0:
            continue
        k = av[0]
        slot[k] *= (1 + net[t]); free[k] = i + hold_bars
        taken += 1
        hist.append(slot.sum())
    h = np.array(hist)
    dd = ((h - np.maximum.accumulate(h)) / np.maximum.accumulate(h)).min() if len(h) > 1 else 0
    return h[-1], dd, taken


say('\n' + '=' * 104)
say('【持有期扫描】信号固定（跌幅5%% / 量比>3 / 多币>=2 / 币7天跌>20%%不进场），只改持有期')
say('  %-14s %6s %10s %9s %10s %11s %11s %10s %8s' % (
    '持有期', '成交', '单笔净%', '胜率%', '中位%', '最大单笔%', '最小单笔%', 'K3终值', '回撤%'))
res = {}
for hold_bars, lb in HOLDS:
    ii, jj = dedup_idx(SIGBASE, hold_bars)
    if len(ii) < 30:
        say('  %-14s 样本不足' % lb); continue
    FWD = np.full((T, N), np.nan); FWD[:-hold_bars] = P[hold_bars:] / P[:-hold_bars] - 1.0
    okf = np.isfinite(FWD[ii, jj])
    ii2, jj2 = ii[okf], jj[okf]
    keep = ~(coin_7d[ii2, jj2] < -0.20)
    ii2, jj2 = ii2[keep], jj2[keep]
    if len(ii2) < 30:
        say('  %-14s 过滤后不足' % lb); continue
    y = FWD[ii2, jj2]
    net = y - 2 * FEE_M
    end, dd, taken = compound(ii2, jj2, net, K=3)
    yr = pd.to_datetime(tms[ii2], unit='ms', utc=True).year.values
    yy = sorted(set(yr))
    pos = sum(1 for k in yy if (yr == k).sum() >= 5 and net[yr == k].mean() > 0)
    tot = sum(1 for k in yy if (yr == k).sum() >= 5)
    res[lb] = (net, end, dd, taken, '%d/%d' % (pos, tot))
    say('  %-14s %6d %+9.3f%% %8.1f%% %+9.3f%% %+10.2f%% %+10.2f%% %10.2f %7.1f%%  (正年 %s)' % (
        lb, len(ii2), 100 * net.mean(), 100 * (net > 0).mean(), 100 * np.median(net),
        100 * net.max(), 100 * net.min(), end, 100 * dd, '%d/%d' % (pos, tot)))

say('\n【关键判据】单笔收益是否随持有期增长？')
prev = None
for hold_bars, lb in HOLDS:
    if lb not in res:
        continue
    g = res[lb][0].mean()
    s_ = '' if prev is None else '（相对上一档 %+.1f%%）' % (100 * (g - prev))
    say('  %-14s 单笔 = %+.3f%%%s' % (lb, 100 * g, s_))
    prev = g

say('\n【「一夜翻十倍」可行性】需要单笔 >= +900%')
best = None
for hold_bars, lb in HOLDS:
    if lb not in res:
        continue
    mx = res[lb][0].max()
    if best is None or mx > best[0]:
        best = (mx, lb)
if best:
    say('  实测最大单笔 = %+.1f%%（持有期 %s）' % (100 * best[0], best[1]))
    say('  距离 +900%% 差 %.0f 倍' % (900 / (100 * best[0]) if best[0] > 0 else float('inf')))
    say('  即：翻十倍需要单笔 %.0f 倍于实测最好那一笔' % (900 / (100 * best[0])))
    # 极端单笔占比
    for hold_bars, lb in HOLDS:
        if lb not in res:
            continue
        net = res[lb][0]
        say('  %-14s 单笔 >+10%% 的占比 = %.2f%%（%d/%d）' % (
            lb, 100 * (net > 0.10).mean(), int((net > 0.10).sum()), len(net)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
lbs = [lb for _, lb in HOLDS if lb in res]
ax[0].bar(range(len(lbs)), [100 * res[lb][0].mean() for lb in lbs])
ax[0].set_xticks(range(len(lbs))); ax[0].set_xticklabels(lbs, rotation=35, fontsize=8)
ax[0].axhline(0, color='k', lw=.8); ax[0].set_title('mean net % per trade by holding period')
ax[1].bar(range(len(lbs)), [res[lb][1] for lb in lbs])
ax[1].set_xticks(range(len(lbs))); ax[1].set_xticklabels(lbs, rotation=35, fontsize=8)
ax[1].axhline(CAP, color='k', lw=.8, ls='--'); ax[1].set_title('K=3 equity from 1000 CNY')
ax[2].hist(100 * res[lbs[0]][0], bins=50, alpha=.7, label=lbs[0])
if len(lbs) > 3:
    ax[2].hist(100 * res[lbs[3]][0], bins=50, alpha=.7, label=lbs[3])
ax[2].set_yscale('log'); ax[2].legend(fontsize=8); ax[2].set_title('per-trade return % (log count)')
plt.tight_layout()
p = os.path.join(EV, 'hold_scan.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'hold_scan.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
