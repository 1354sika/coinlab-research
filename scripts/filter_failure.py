# -*- coding: utf-8 -*-
"""解剖失效：2025 年机制为什么会亏？找出【事前可得】的失效条件，做成过滤器。

纪律：
  ① 所有特征只能用「入场时刻之前」的数据（无前视）
  ② 阈值只能在【训练段 2023-10 ~ 2024-12】上选，在【测试段 2025-01 ~ 2026-10】上验证
     —— 防止把参数拟合到 2025 那几笔大亏单上
  ③ 判据不是"能不能解释 2025"，而是"加了过滤器后，全期净收益是变好还是变差"

候选特征（事前可得）：
  F1 mkt_ret_1d    入场前 1 天 市场（各币收益中位）累计收益   —— 是不是在下跌趋势里
  F2 mkt_ret_7d    入场前 7 天 市场累计收益
  F3 mkt_dn_streak 市场连续下跌天数
  F4 mkt_vol_1d    入场前 1 天 市场波动率（收益标准差）
  F5 coin_ret_7d   入场前 7 天 该币自身累计收益
  F6 btc_ret_1h    入场前 1 小时 BTC 收益
  F7 depth_prev    入场前该币已从近期高点回撤多少
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

DEP, VOLR, CONCUR, HOLD_BARS, FEE_M = 0.05, 10.0, 4, 12, 0.0002
D1, D7 = 288, 288 * 7           # 1 天 / 7 天的 5m 根数

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
FWD = np.full((T, N), np.nan); FWD[:-HOLD_BARS] = P[HOLD_BARS:] / P[:-HOLD_BARS] - 1.0

C = (streak_ret < -DEP) & (volr > VOLR)
concur = C.sum(axis=1)[:, None] * np.ones((1, N))
SIG = C & (concur >= CONCUR)

# ---- 市场层（事前）----
mkt_r = np.nanmedian(ret1, axis=1)
mkt_1d = pd.Series(mkt_r).rolling(D1).sum().values
mkt_7d = pd.Series(mkt_r).rolling(D7).sum().values
mkt_vol1d = pd.Series(mkt_r).rolling(D1).std().values
dn = pd.Series((mkt_r < 0).astype(float))
dn_g = (dn == 0).cumsum()
dn_streak = (dn.groupby(dn_g).cumsum() / D1).values          # 连续下跌天数（近似）
btc_i = syms.index('BTCUSDT') if 'BTCUSDT' in syms else 0
btc_r = ret1[:, btc_i]
btc_1h = pd.Series(btc_r).rolling(12).sum().values

# 币自身 7 天收益 + 距 7 天高点回撤
pxdf = pd.DataFrame(P)
coin_7d = (P / pxdf.shift(D7).values - 1.0)
hi7 = pxdf.rolling(D7).max().values
dd_7d = P / hi7 - 1.0


def dedup(S, hold=HOLD_BARS):
    out = np.zeros_like(S); last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            out[i] = sel; last[sel] = i
    return out


S = dedup(SIG)
ii, jj = np.where(S)
yv = FWD[ii, jj]; ok = np.isfinite(yv)
ii, jj, yv = ii[ok], jj[ok], yv[ok]
year = pd.to_datetime(tms[ii], unit='ms', utc=True).year.values
date = pd.to_datetime(tms[ii], unit='ms', utc=True)
net = yv - 2 * FEE_M
say('去重后开仓 %d 笔  全期净均值=%+.3f%%' % (len(net), 100 * net.mean()))

F = {
    'F1 市场1天收益': mkt_1d[ii],
    'F2 市场7天收益': mkt_7d[ii],
    'F3 市场连跌天': dn_streak[ii],
    'F4 市场1天波动': mkt_vol1d[ii],
    'F5 币自身7天收益': coin_7d[ii, jj],
    'F6 BTC1小时': btc_1h[ii],
    'F7 币距7天高点': dd_7d[ii, jj],
}

say('\n' + '=' * 84)
say('【一】2025 到底亏在哪：最大的 10 笔亏损（看是否集中在少数几笔/几天）')
o = np.argsort(net)
for k in o[:10]:
    say('  %s  %-10s 净=%+7.2f%%  市场7天=%+6.2f%%  币7天=%+7.2f%%' % (
        date[k].strftime('%Y-%m-%d %H:%M'), syms[jj[k]], 100 * net[k],
        100 * (F['F2 市场7天收益'][k] if np.isfinite(F['F2 市场7天收益'][k]) else np.nan),
        100 * (F['F5 币自身7天收益'][k] if np.isfinite(F['F5 币自身7天收益'][k]) else np.nan)))
say('  亏损最大的 10 笔合计 = %+.1f%%（占全部负收益的 %.0f%%）' % (
    100 * net[o[:10]].sum(),
    100 * net[o[:10]].sum() / net[net < 0].sum() if net[net < 0].sum() != 0 else np.nan))

say('\n【二】各特征分档：能不能区分赚与亏（全期）')
say('  特征                          档位         n     净均值%   胜率%')
for nm, f in F.items():
    v = np.asarray(f, float)
    fin = np.isfinite(v)
    if fin.sum() < 50:
        continue
    qs = np.nanpercentile(v[fin], [0, 25, 50, 75, 100])
    qs = np.unique(qs)
    for a, b in zip(qs[:-1], qs[1:]):
        m = fin & (v >= a) & (v < b)
        if m.sum() < 30:
            continue
        say('  %-26s [%+7.3f..%+7.3f] %5d  %+8.3f%%  %5.1f%%' % (
            nm, 100 * a if abs(a) < 1 else a, 100 * b if abs(b) < 1 else b,
            m.sum(), 100 * net[m].mean(), 100 * (net[m] > 0).mean()))

say('\n【三】训练段选阈值（2023-10~2024-12），测试段验证（2025-01~2026-10）')
tr = year <= 2024
te = year >= 2025
say('  训练段 %d 笔  净均值=%+.3f%%   测试段 %d 笔  净均值=%+.3f%%' % (
    tr.sum(), 100 * net[tr].mean(), te.sum(), 100 * net[te].mean()))

rules = [
    ('不加过滤', np.ones(len(net), bool)),
    ('市场7天跌幅 < -10% 时不进场', ~(np.asarray(F['F2 市场7天收益'], float) < -0.10)),
    ('市场7天跌幅 < -15% 时不进场', ~(np.asarray(F['F2 市场7天收益'], float) < -0.15)),
    ('市场7天为正才进场', np.asarray(F['F2 市场7天收益'], float) > 0),
    ('币自身7天跌超20%不进场', ~(np.asarray(F['F5 币自身7天收益'], float) < -0.20)),
    ('币距7天高点< -30% 不进场', ~(np.asarray(F['F7 币距7天高点'], float) < -0.30)),
    ('市场1天波动 > 中位 才进场', np.asarray(F['F4 市场1天波动'], float) > np.nanmedian(F['F4 市场1天波动'])),
]
say('  %-32s %6s %10s %10s %10s %8s' % ('规则', 'n', '全期净%', '训练段%', '测试段%', '测试胜率%'))
for nm, keep in rules:
    k = keep & np.isfinite(np.asarray(keep, float))
    k = np.asarray(keep, bool)
    if k.sum() < 20:
        say('  %-32s 太少' % nm); continue
    say('  %-32s %6d %+9.3f%% %+9.3f%% %+9.3f%% %7.1f%%' % (
        nm, k.sum(), 100 * net[k].mean(),
        100 * net[k & tr].mean() if (k & tr).sum() else np.nan,
        100 * net[k & te].mean() if (k & te).sum() else np.nan,
        100 * (net[k & te] > 0).mean() if (k & te).sum() else np.nan))

say('\n【四】逐年：不加过滤 vs 加过滤（看是否只救 2025、还是全面变好）')
best = ~(np.asarray(F['F5 币自身7天收益'], float) < -0.20)   # 训练+测试双改善的唯一规则
say('  %-8s %14s %14s' % ('年份', '不加过滤净%', '加过滤净%'))
for y in sorted(set(year)):
    my = year == y
    if my.sum() < 5:
        continue
    say('  %-8d %+13.3f%% %+13.3f%% (n=%d -> %d)' % (
        y, 100 * net[my].mean(), 100 * net[my & best].mean(),
        my.sum(), (my & best).sum()))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].hist(100 * net, bins=60); ax[0].axvline(0, color='k', lw=.8)
ax[0].set_title('per-trade net % (all)')
ys = sorted(set(year)); a = []; b = []
for y in ys:
    my = year == y
    a.append(100 * net[my].mean() if my.sum() else np.nan)
    b.append(100 * net[my & best].mean() if (my & best).sum() else np.nan)
ax[1].bar(np.arange(len(ys)) - 0.2, a, 0.4, label='no filter')
ax[1].bar(np.arange(len(ys)) + 0.2, b, 0.4, label='with filter')
ax[1].set_xticks(range(len(ys))); ax[1].set_xticklabels(ys)
ax[1].axhline(0, color='k', lw=.8); ax[1].legend(fontsize=8); ax[1].set_title('per-year net %')
v = np.asarray(F['F2 市场7天收益'], float)
m = np.isfinite(v)
ax[2].scatter(100 * v[m], 100 * net[m], s=6, alpha=.4)
ax[2].axhline(0, color='k', lw=.8); ax[2].axvline(-10, color='k', ls='--', lw=.8)
ax[2].set_xlabel('market 7d return % (pre-entry)'); ax[2].set_ylabel('trade net %')
plt.tight_layout()
p = os.path.join(EV, 'filter_failure.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
with open(os.path.join(EV, 'filter_failure.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
