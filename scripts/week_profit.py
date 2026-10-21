# -*- coding: utf-8 -*-
"""1000 元入局，一周能赚多少？—— 用 3 年实测交易序列按周聚合，做三种仓位方案的资金模拟

纪律：
  ① 一周样本极少（3 年 = 156 周，有效交易 285 笔）-> 必须给"分布"而不是"一个数"
  ② 三种仓位方案并列：全押 / 每笔投 10% / 每笔固定 100 元
  ③ 扣费 maker；事件去重；无前视
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

DEP, VOLR, CONCUR, HOLD = 0.05, 10.0, 4, 12
FEE_M = 0.0002
D7 = 288 * 7

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

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60
s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
FWD = np.full((T, N), np.nan); FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
CV = volr > VOLR; CD = streak_ret < -DEP
conc = (CD & CV).sum(axis=1)[:, None] * np.ones((1, N))
SIG = CD & CV & (conc >= CONCUR)
coin_7d = P / pd.DataFrame(P).shift(D7).values - 1.0


def dedup(S, hold=HOLD):
    out = np.zeros_like(S); last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            out[i] = sel; last[sel] = i
    return out


ii, jj = np.where(dedup(SIG))
y = FWD[ii, jj]; ok = np.isfinite(y)
ii, jj, y = ii[ok], jj[ok], y[ok]
keep = ~(coin_7d[ii, jj] < -0.20)
ii, jj, y = ii[keep], jj[keep], y[keep]
net = y - 2 * FEE_M
dt = pd.to_datetime(tms[ii], unit='ms', utc=True)
say('有效交易 %d 笔，跨度 %s ~ %s' % (len(net), dt.min().date(), dt.max().date()))

wk = dt.to_period('W')
wkv = np.asarray(wk)
weeks = sorted(set(wk))
days = (dt.max() - dt.min()).days
say('总跨度 %d 天 = %.0f 周；其中有交易的周 = %d' % (days, days / 7, len(weeks)))
cnt = pd.Series(1, index=wk).groupby(level=0).sum()
say('每周笔数：中位 %d  均值 %.2f  最多 %d  （无交易的周 %d 个）' % (
    cnt.median(), cnt.mean(), cnt.max(), int(days / 7) - len(weeks)))

CAP = 1000.0


def sim(mode):
    eq = CAP; hist = []
    for w in weeks:
        yw = net[wkv == np.asarray(w)]
        if mode == 'allin':
            for r in yw: eq *= (1 + r)
        elif mode == 'p10':
            for r in yw: eq += eq * 0.10 * r
        elif mode == 'fix100':
            for r in yw: eq += 100.0 * r
        hist.append(eq)
    return np.array(hist), eq


say('\n【三种仓位方案：1000 元本金，%d 个有交易的周】' % len(weeks))
say('  %-18s %14s %12s %12s' % ('方案', '3年后(元)', '总收益%', '最大回撤%'))
res = {}
for mode, nm in (('allin', '全押（复利）'), ('p10', '每笔投 10%'), ('fix100', '每笔固定 100 元')):
    hist, end = sim(mode)
    peak = np.maximum.accumulate(hist)
    dd = ((hist - peak) / peak).min()
    res[mode] = (hist, end, dd)
    say('  %-18s %14.2f %11.1f%% %11.1f%%' % (nm, end, 100 * (end / CAP - 1), 100 * dd))

say('\n【核心问题：一周能赚多少】（每笔投本金 10%，即 100 元）')
wkret = np.array([np.prod(1 + 0.10 * net[wkv == np.asarray(w)]) - 1 for w in weeks])
say('  有交易的周 %d 个 / 全期约 %d 周 -> %.0f%% 的周根本没有机会开仓' % (
    len(wkret), int(days / 7), 100 * (1 - len(wkret) / (days / 7))))
say('  周收益率：中位 %+.3f%%  均值 %+.3f%%  P5 %+.3f%%  P95 %+.3f%%  最差 %+.3f%%  最好 %+.3f%%' % (
    100 * np.median(wkret), 100 * wkret.mean(), 100 * np.percentile(wkret, 5),
    100 * np.percentile(wkret, 95), 100 * wkret.min(), 100 * wkret.max()))
say('  → 1000 元本金：中位 %+.2f 元/周   均值 %+.2f 元/周   最差 %.2f 元   最好 %+.2f 元' % (
    CAP * np.median(wkret), CAP * wkret.mean(), CAP * wkret.min(), CAP * wkret.max()))
say('  亏损周占比 = %.0f%%（%d/%d）' % (100 * (wkret < 0).mean(), int((wkret < 0).sum()), len(wkret)))

say('\n【同一批周，若是每笔全押（这是上限，不是建议）】')
wr2 = np.array([np.prod(1 + net[wkv == np.asarray(w)]) - 1 for w in weeks])
say('  周收益率：中位 %+.3f%%  均值 %+.3f%%  最差 %+.3f%%  最好 %+.3f%%' % (
    100 * np.median(wr2), 100 * wr2.mean(), 100 * wr2.min(), 100 * wr2.max()))
say('  → 1000 元：中位 %+.2f 元   最差 %.2f 元   最好 %+.2f 元' % (
    CAP * np.median(wr2), CAP * wr2.min(), CAP * wr2.max()))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].bar(range(len(weeks)), 100 * wkret); ax[0].axhline(0, color='k', lw=.8)
ax[0].set_title('weekly return % (10% per trade)'); ax[0].set_xlabel('week #')
ax[1].plot(res['p10'][0], label='10% per trade'); ax[1].plot(res['allin'][0], label='all-in')
ax[1].axhline(CAP, color='k', lw=.8, ls='--'); ax[1].legend(fontsize=8)
ax[1].set_title('equity from 1000 CNY (3y)')
ax[2].hist(100 * wkret, bins=30); ax[2].axvline(0, color='k', lw=.8)
ax[2].set_title('distribution of weekly return %')
plt.tight_layout()
p = os.path.join(EV, 'week_profit.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'week_profit.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
