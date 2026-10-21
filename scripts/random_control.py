# -*- coding: utf-8 -*-
"""多重检验对照（回答用户："几乎所有机制都只在某一样本的特殊时期有效"）
问题：我扫了很多「特征 x 阈值」组合，挑出看起来最好的 —— 这本身可能在挖噪声。
做法：
  A) 随机生成 N 个「假信号」：从特征池随机挑一个特征、随机挑一个分位阈值、随机挑方向
     用与真信号【完全相同】的回测流程（事件级去重、等权组合、扣 maker、持 60 分钟）
     得到「随机策略」收益分布。
  B) 把真信号（volr>10 & 段长>=3 做多等）放进这个分布，看它落在什么分位。
  C) 把已验证的机制在 6 个独立时段（各 2 个月）上重测，看有几个时段仍然有效。
判据：真信号若落在随机分布 P99 之外 -> 可能是真；落在中间 -> 很可能是数据挖掘产物。
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, DQ, FEE_M, NSIM = 60, 0.90, 0.0002, 200
t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
QV = qv.values.astype(np.float64)
qv60 = qv.rolling(60).mean().values
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
MA60 = px.rolling(60).mean().values
madev = P / MA60 - 1.0
vol60 = px.pct_change().rolling(60).std().values.astype(np.float64)
R5 = (px / px.shift(5) - 1.0).values.astype(np.float64)
R15 = (px / px.shift(15) - 1.0).values.astype(np.float64)
R60 = (px / px.shift(60) - 1.0).values.astype(np.float64)
RNG15 = ((px.rolling(15).max() - px.rolling(15).min()) / px).values.astype(np.float64)
FWD = np.full((T, N), np.nan)
FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)

POOL = {
    'R5': R5, 'R15': R15, 'R60': R60, 'streak_ret': streak_ret, 'volr': volr,
    'MA60dev': madev, 'vol60': vol60, 'Rng15': RNG15,
}
say('特征池: %s' % ', '.join(POOL.keys()))


def backtest(mask):
    """事件级去重 + 等权组合（时间等权、复利、扣 maker），返回净值与开仓数"""
    pos = np.zeros((T, N), np.float32)
    nopen = 0
    for j in range(N):
        ii = np.flatnonzero(mask[:, j])
        if len(ii) == 0:
            continue
        last = -1
        for k in ii:
            if k <= last:
                continue
            pos[k:k + HOLD, j] = 1.0
            nopen += 1
            last = k + HOLD - 1
    if nopen == 0:
        return np.nan, 0
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (pos[:-1] > 0).sum(axis=1).astype(np.float64)
    eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    d = np.abs(np.diff(pos, axis=0, prepend=0))
    fs = (d > 1e-6).sum(axis=1) * FEE_M
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = eq - fse
    return float(np.cumprod(1 + net)[-1] - 1), nopen


say('\n=== A) %d 个随机假信号（随机特征 x 随机分位阈值 x 随机方向）===' % NSIM)
rs = np.random.RandomState(20261015)
keys = list(POOL.keys())
sims = []
for i in range(NSIM):
    name = keys[rs.randint(len(keys))]
    F = POOL[name]
    q = rs.uniform(0.02, 0.10)          # 取最极端 2%~10% 的分位
    side = rs.choice([-1, 1])
    fin = F[np.isfinite(F)]
    if side > 0:
        thr = np.quantile(fin, 1 - q)
        m = (F >= thr) & sel
    else:
        thr = np.quantile(fin, q)
        m = (F <= thr) & sel
    nav, n = backtest(m)
    if np.isfinite(nav) and n >= 30:
        sims.append((nav, n, name, q, side))
    if (i + 1) % 40 == 0:
        say('  ... %d/%d 完成，有效 %d，用时 %.0fs' % (i + 1, NSIM, len(sims), time.time() - t0))

sims.sort()
navs = np.array([x[0] for x in sims])
say('\n  有效随机策略 = %d 个' % len(navs))
say('  随机策略净值分布: 均值=%+.1f%%  中位=%+.1f%%  P75=%+.1f%%  P90=%+.1f%%  P99=%+.1f%%  最大=%+.1f%%' % (
    100*navs.mean(), 100*np.median(navs), 100*np.percentile(navs, 75),
    100*np.percentile(navs, 90), 100*np.percentile(navs, 99), 100*navs.max()))
say('  负数占比 = %.1f%%' % (100*(navs < 0).mean()))

say('\n=== B) 真信号 vs 随机分布 ===')
real = {}
for nm, sig in (
    ('真信号: 急跌+volr>10（无段长）', (streak_ret < -0.05) & (volr > 10) & sel),
    ('真信号: 急跌+volr>10+段长>=3', (streak_ret < -0.05) & (volr > 10) & (seglen >= 3) & sel),
    ('真信号: 急涨+volr>10（做空）', (streak_ret > +0.05) & (volr > 10) & sel),
    ('真信号: 低位+远离均线', (R60 <= np.nanquantile(R60[np.isfinite(R60)], 0.4)) &
     (madev <= np.nanquantile(madev[np.isfinite(madev)], 0.2)) & sel),
):
    nav, n = backtest(sig)
    real[nm] = nav
    if not np.isfinite(nav):
        continue
    pct = 100 * (navs < nav).mean()
    say('  %-32s 净值=%+9.1f%%  开仓=%5d  -> 在随机分布中位于 P%.0f' % (nm, 100*nav, n, pct))
say('\n  判读：真信号若落在 P95 以内，说明"扫阈值挑出来的好结果"和随机挑没本质区别。')

say('\n=== C) 机制在 6 个独立时段（各 2 个月）上的稳定性 ===')
mo = pd.to_datetime(t, unit='ms', utc=True).dt.strftime('%Y-%m')
months = sorted(mo.unique())
blocks = [months[i:i + 2] for i in range(0, len(months), 2)]
mechs = {
    '急跌+volr>10': (streak_ret < -0.05) & (volr > 10) & sel,
    '急涨+volr>10': (streak_ret > +0.05) & (volr > 10) & sel,
}
say('  %-16s' % '机制' + ''.join('%14s' % ('/'.join(b)) for b in blocks))
for nm, m in mechs.items():
    row = '  %-16s' % nm
    for b in blocks:
        mm = np.isin(mo, b)[:, None] & np.ones((1, N), bool)
        y = FWD[m & mm]; y = y[np.isfinite(y)]
        row += '%14s' % ('%+.2f%%(n%d)' % (100*y.mean(), len(y)) if len(y) >= 20 else '-')
    say(row)

with open(os.path.join(EV, 'random_control.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
