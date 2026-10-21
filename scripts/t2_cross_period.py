# -*- coding: utf-8 -*-
"""跨牛熊外部验证：把已确认机制（急跌+巨量+多币同时 -> 做多）放到 3 年 5m 数据上，分年重测。

为什么用 5m：1m x 3 年 x 60 币 = 1.57 亿行（约 4 小时拉取）；5m 只需 1252 万行（已拉完）。
代价：粒度变粗（5m 一根），参数需按 5m 口径重新标定，但「急跌+巨量 -> 反弹」的结构不变。

对照：每个年份都同时给出
  ① 机制（急跌+巨量+多币同时）做多
  ② 随机对照（同一年、同样事件数、随机挑时点）
  ③ 全样本基准（同年所有 5m 的均值）—— 用来判断"是不是只是市场整体在涨"
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

BARS_PER_MIN = 5          # 5m 数据
DEP = 0.05                # 段内跌幅门槛
VOLR = 10.0               # 量比（相对过去 60 根 = 5 小时）
CONCUR = 4                # 同分钟触发数
HOLD_BARS = 12            # 持有 12 根 5m = 60 分钟
FEE_M = 0.0002
rng = np.random.default_rng(20261015)

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float64', 'quote_volume': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values
QV = qv.values
T, N = P.shape
tms = px.index.values
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan)
ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60

s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0

FWD = np.full((T, N), np.nan)
FWD[:-HOLD_BARS] = P[HOLD_BARS:] / P[:-HOLD_BARS] - 1.0

C = (streak_ret < -DEP) & (volr > VOLR)
concur = C.sum(axis=1)[:, None] * np.ones((1, N))
SIG = C & (concur >= CONCUR)

years = pd.to_datetime(tms, unit='ms', utc=True).year.values
say('信号（bar x 币，未去重）= %d' % int(SIG.sum()))
say('年份分布：' + ' '.join('%d:%d' % (y, int((years == y).sum())) for y in sorted(set(years))))


def dedup(S, hold=HOLD_BARS):
    out = np.zeros_like(S)
    last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        row = S[i]
        sel = row & ((i - last) >= hold)
        if sel.any():
            out[i] = sel
            last[sel] = i
    return out


S = dedup(SIG)
ii, jj = np.where(S)
yv = FWD[ii, jj]
ok = np.isfinite(yv)
ii, jj, yv = ii[ok], jj[ok], yv[ok]
yr = years[ii]
say('去重后开仓 = %d 笔' % len(yv))

say('\n' + '=' * 84)
say('【一】分年：机制 vs 随机对照 vs 全样本基准')
say('  年份   机制n   机制净maker%  随机对照均值%   随机最好%   全样本基准%   机制胜率%')
allbase = np.nanmean(FWD)
for y in sorted(set(yr)):
    m = yr == y
    if m.sum() < 5:
        continue
    n = int(m.sum())
    g = yv[m].mean()
    net = 100 * (g - 2 * FEE_M)
    # 随机对照：同年随机取 n 个 (bar,币) 组合
    ybars = np.where(years == y)[0]
    ybars = ybars[ybars < T - HOLD_BARS]
    rr = []
    for _ in range(200):
        a = rng.choice(ybars, n, replace=True)
        b = rng.integers(0, N, n)
        v = FWD[a, b]
        v = v[np.isfinite(v)]
        if len(v):
            rr.append(v.mean())
    rr = np.array(rr)
    base = np.nanmean(FWD[ybars])
    say('  %d  %6d   %+9.3f%%    %+9.3f%%   %+8.3f%%    %+8.3f%%    %6.1f%%' % (
        y, n, net, 100 * rr.mean(), 100 * rr.max(), 100 * base, 100 * (yv[m] > 0).mean()))

say('\n【二】机制扣费后的分年净收益（maker / taker）')
for y in sorted(set(yr)):
    m = yr == y
    if m.sum() < 5:
        continue
    g = yv[m].mean()
    say('  %d  毛=%+.3f%%  净maker=%+.3f%%  净taker=%+.3f%%  中位=%+.3f%%  胜率=%.1f%%' % (
        y, 100 * g, 100 * (g - 2 * FEE_M), 100 * (g - 2 * 0.001),
        100 * np.median(yv[m]), 100 * (yv[m] > 0).mean()))

say('\n【三】逐年正负（判断是否"跨牛熊都成立"）')
pos = sum(1 for y in sorted(set(yr)) if (yr == y).sum() >= 5
          and yv[yr == y].mean() - 2 * FEE_M > 0)
tot = sum(1 for y in sorted(set(yr)) if (yr == y).sum() >= 5)
say('  → 净收益为正的年份 %d/%d' % (pos, tot))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ys = sorted(set(yr)); nets = []; bases = []
for y in ys:
    m = yr == y
    nets.append(100 * (yv[m].mean() - 2 * FEE_M) if m.sum() else np.nan)
    ybars = np.where(years == y)[0]; ybars = ybars[ybars < T - HOLD_BARS]
    bases.append(100 * np.nanmean(FWD[ybars]))
ax[0].bar(np.arange(len(ys)) - 0.2, nets, 0.4, label='mechanism net(maker)')
ax[0].bar(np.arange(len(ys)) + 0.2, bases, 0.4, label='market base')
ax[0].set_xticks(range(len(ys))); ax[0].set_xticklabels(ys)
ax[0].axhline(0, color='k', lw=.8); ax[0].legend(fontsize=8); ax[0].set_title('per-year: mechanism vs market')
idx = np.argsort(ii)
ax[1].plot(np.cumprod(1 + 0.01 * (yv[idx] - 2 * FEE_M)))
ax[1].set_title('equity (1%% per trade, maker)'); ax[1].set_xlabel('trade #')
ax[2].hist(100 * yv, bins=60); ax[2].axvline(0, color='k', lw=.8)
ax[2].set_title('per-trade return %%')
plt.tight_layout()
p = os.path.join(EV, 'cross_period.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
with open(os.path.join(EV, 'cross_period.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
