# -*- coding: utf-8 -*-
"""全量机制测试：盈利侧（做多）+ 防跌侧（做空）+ 二维安全区 + 随机对照

机制清单（本会话已累积确认）：
  M1 做多：急跌(段内<-5%) + 巨量(量比>10) + 多币同时(>=4) -> 持 60 分钟
     过滤器：币自身 7 天跌超 20% 不进场（训练/测试双改善的唯一规则）
  M2 做空：急涨(段内>+5%) + 巨量 + 多币同时 -> 持 60 分钟（对称假设）
  M3 二维：市场7天跌幅 x 币自身7天跌幅 -> 做多净收益（找安全区）
  M4 随机对照：每个机制在同一年随机取同样笔数，看机制落在什么分位

数据：klines_5m_3y.csv（60 币 / 2023-10 ~ 2026-10 / 1252 万行）
纪律：事件去重（持仓期内不重复开仓）、无前视、扣费双档（maker 0.02% / taker 0.10% 单边）
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
FEE_M, FEE_T = 0.0002, 0.0010
D1, D7 = 288, 288 * 7
rng = np.random.default_rng(20261015)

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
syms = list(px.columns)
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))

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

CV = volr > VOLR
CD = streak_ret < -DEP
CU = streak_ret > DEP
conc_d = (CD & CV).sum(axis=1)[:, None] * np.ones((1, N))
conc_u = (CU & CV).sum(axis=1)[:, None] * np.ones((1, N))
SIG_L = CD & CV & (conc_d >= CONCUR)
SIG_S = CU & CV & (conc_u >= CONCUR)

mkt_r = np.nanmedian(ret1, axis=1)
mkt_7d = pd.Series(mkt_r).rolling(D7).sum().values
mkt_1d = pd.Series(mkt_r).rolling(D1).sum().values
pxdf = pd.DataFrame(P)
coin_7d = P / pxdf.shift(D7).values - 1.0
say('信号：做多 %d 个 bar-币，做空 %d 个' % (int(SIG_L.sum()), int(SIG_S.sum())))


def dedup(S, hold=HOLD):
    out = np.zeros_like(S); last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            out[i] = sel; last[sel] = i
    return out


def pack(S, direction):
    D = dedup(S)
    ii, jj = np.where(D)
    y = FWD[ii, jj]
    ok = np.isfinite(y)
    ii, jj, y = ii[ok], jj[ok], y[ok]
    return ii, jj, y * direction


iiL, jjL, yL = pack(SIG_L, +1)
yrL = pd.to_datetime(tms[iiL], unit='ms', utc=True).year.values
f5L = coin_7d[iiL, jjL]; fmL = mkt_7d[iiL]; f1L = mkt_1d[iiL]
keepL = ~(f5L < -0.20)

iiS, jjS, yS = pack(SIG_S, -1)
yrS = pd.to_datetime(tms[iiS], unit='ms', utc=True).year.values
f5S = coin_7d[iiS, jjS]
keepS_up = ~(f5S > 0.20)

say('\n' + '=' * 90)
say('【M1】盈利侧：急跌+巨量+多币同时 -> 做多（持 60 分钟）')
say('  %-26s %5s %11s %11s %8s %10s' % ('口径', 'n', '净maker%', '净taker%', '胜率%', '中位%'))
for nm, k in (('不加过滤', np.ones(len(yL), bool)), ('币7天跌>20%不进场', keepL)):
    if k.sum() < 10:
        continue
    g = yL[k].mean()
    say('  %-26s %5d %+10.3f%% %+10.3f%% %7.1f%% %+9.3f%%' % (
        nm, k.sum(), 100 * (g - 2 * FEE_M), 100 * (g - 2 * FEE_T),
        100 * (yL[k] > 0).mean(), 100 * np.median(yL[k])))
say('  逐年（加过滤）：')
for y in sorted(set(yrL)):
    m = (yrL == y) & keepL
    if m.sum() < 5:
        continue
    g = yL[m].mean()
    say('    %d  n=%3d  毛=%+7.3f%%  净maker=%+7.3f%%  胜率=%.1f%%  中位=%+.3f%%' % (
        y, m.sum(), 100 * g, 100 * (g - 2 * FEE_M), 100 * (yL[m] > 0).mean(), 100 * np.median(yL[m])))
yy = sorted(set(yrL)); npos = sum(1 for y in yy if ((yrL == y) & keepL).sum() >= 5 and (yL[(yrL == y) & keepL].mean() - 2 * FEE_M) > 0)
ntot = sum(1 for y in yy if ((yrL == y) & keepL).sum() >= 5)
say('  → 净为正年份 %d/%d' % (npos, ntot))

say('\n【M2】防跌侧：急涨+巨量+多币同时 -> 做空（对称假设）')
say('  %-26s %5s %11s %11s %8s %10s' % ('口径', 'n', '净maker%', '净taker%', '胜率%', '中位%'))
for nm, k in (('不加过滤', np.ones(len(yS), bool)), ('币7天涨>20%不进场', keepS_up)):
    if k.sum() < 10:
        continue
    g = yS[k].mean()
    say('  %-26s %5d %+10.3f%% %+10.3f%% %7.1f%% %+9.3f%%' % (
        nm, k.sum(), 100 * (g - 2 * FEE_M), 100 * (g - 2 * FEE_T),
        100 * (yS[k] > 0).mean(), 100 * np.median(yS[k])))
say('  逐年（不加过滤）：')
for y in sorted(set(yrS)):
    m = yrS == y
    if m.sum() < 5:
        continue
    g = yS[m].mean()
    say('    %d  n=%3d  毛=%+7.3f%%  净maker=%+7.3f%%  胜率=%.1f%%  中位=%+.3f%%' % (
        y, m.sum(), 100 * g, 100 * (g - 2 * FEE_M), 100 * (yS[m] > 0).mean(), 100 * np.median(yS[m])))
yyS = sorted(set(yrS)); nposS = sum(1 for y in yyS if (yrS == y).sum() >= 5 and (yS[yrS == y].mean() - 2 * FEE_M) > 0)
ntotS = sum(1 for y in yyS if (yrS == y).sum() >= 5)
say('  → 净为正年份 %d/%d' % (nposS, ntotS))

say('\n【M4】随机对照：同年同量随机取点，机制落在什么分位')
for nm, ii_, yv_ in (('M1 做多', iiL, yL), ('M2 做空', iiS, yS)):
    yrs = pd.to_datetime(tms[ii_], unit='ms', utc=True).year.values
    allr = []
    for y in sorted(set(yrs)):
        m = yrs == y
        n = int(m.sum())
        yb = np.where(pd.to_datetime(tms, unit='ms', utc=True).year == y)[0]
        yb = yb[yb < T - HOLD]
        if len(yb) < n:
            continue
        for _ in range(60):
            a = rng.choice(yb, n, replace=True); b = rng.integers(0, N, n)
            v = FWD[a, b]; v = v[np.isfinite(v)]
            if len(v):
                allr.append(v.mean())
    allr = np.array(allr)
    if len(allr):
        say('  %-10s 机制净=%+.3f%%  随机中位=%+.3f%%  随机P95=%+.3f%%  位置=%.1f%%' % (
            nm, 100 * (yv_.mean() - 2 * FEE_M), 100 * np.median(allr),
            100 * np.percentile(allr, 95), 100 * (allr < yv_.mean()).mean()))

say('\n【M3】二维安全区：市场7天 x 币自身7天 -> 做多净maker%（括号内 n）')
mk_bins = [(-1e9, -0.25, '市场<-25%'), (-0.25, -0.15, '-25~-15%'), (-0.15, -0.05, '-15~-5%'), (-0.05, 1e9, '>-5%')]
ck_bins = [(-1e9, -0.20, '币<-20%'), (-0.20, -0.10, '-20~-10%'), (-0.10, 0.0, '-10~0%'), (0.0, 1e9, '>0%')]
say('  %-14s' % '市场\\币' + ''.join('%18s' % c[2] for c in ck_bins))
for a, b, lb in mk_bins:
    row = '  %-14s' % lb
    for c, d, _ in ck_bins:
        m = (fmL >= a) & (fmL < b) & (f5L >= c) & (f5L < d)
        if m.sum() < 10:
            row += '%18s' % '-'
        else:
            row += '%18s' % ('%+.2f%%(n=%d)' % (100 * (yL[m].mean() - 2 * FEE_M), m.sum()))
    say(row)

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax = ax.ravel()
ys = sorted(set(yrL))
a1 = [100 * (yL[(yrL == y) & keepL].mean() - 2 * FEE_M) if ((yrL == y) & keepL).sum() else np.nan for y in ys]
b1 = [100 * (yL[yrL == y].mean() - 2 * FEE_M) if (yrL == y).sum() else np.nan for y in ys]
ax[0].bar(np.arange(len(ys)) - 0.2, b1, 0.4, label='M1 long (no filter)')
ax[0].bar(np.arange(len(ys)) + 0.2, a1, 0.4, label='M1 long (+filter)')
ax[0].set_xticks(range(len(ys))); ax[0].set_xticklabels(ys)
ax[0].axhline(0, color='k', lw=.8); ax[0].legend(fontsize=8); ax[0].set_title('M1 long: per-year net(maker) %')
ys2 = sorted(set(yrS))
b2 = [100 * (yS[yrS == y].mean() - 2 * FEE_M) if (yrS == y).sum() else np.nan for y in ys2]
ax[1].bar(range(len(ys2)), b2); ax[1].set_xticks(range(len(ys2))); ax[1].set_xticklabels(ys2)
ax[1].axhline(0, color='k', lw=.8); ax[1].set_title('M2 short: per-year net(maker) %')
ax[2].hist(100 * yL, bins=60, alpha=.6, label='M1 long'); ax[2].hist(100 * yS, bins=60, alpha=.6, label='M2 short')
ax[2].axvline(0, color='k', lw=.8); ax[2].legend(fontsize=8); ax[2].set_title('per-trade net % distribution')
idx = np.argsort(iiL)
ax[3].plot(np.cumprod(1 + 0.01 * (yL[idx] - 2 * FEE_M)), label='M1 long(all)')
ax[3].plot(np.cumprod(1 + 0.01 * (yS[np.argsort(iiS)] - 2 * FEE_M)), label='M2 short(all)')
ax[3].set_title('equity (1% per trade, maker)'); ax[3].legend(fontsize=8)
plt.tight_layout()
p = os.path.join(EV, 'full_mechanisms.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
with open(os.path.join(EV, 'full_mechanisms.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
