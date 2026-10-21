# -*- coding: utf-8 -*-
"""期望收益率 -> 组合回测：把「单笔期望」变成「资金曲线」
策略：P(跌>0.5%) >= TS 时做空、P(涨>0.5%) >= TL 时做多，持有 H 分钟，扣 maker 双边费
输出：交易数 / 总收益 / 年化 / 夏普 / 最大回撤 / 分月稳定性
"""
import os, gc, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = r'D:\中信证券\coinlab\data'
EV   = r'D:\中信证券\coinlab\evidence'
FEE  = 0.0002        # maker 单边 0.02% -> 往返 0.04%
TL   = 0.39          # 做多阈值（model_edge 实测）
TS   = 0.31          # 做空阈值（model_edge 实测）
MIN_PER_YEAR = 525600.0

def say(*a): print(*a, flush=True)

say('=' * 100)
say('期望收益率 · 组合回测  (做多 p>=%.2f / 做空 p>=%.2f / maker 双边 %.2f%%)'
    % (TL, TS, 2 * FEE * 100))
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=5)
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
T, N = close.shape
say('matrix %d x %d' % (T, N))

ret1 = close.pct_change()
mkt1 = ret1.median(axis=1)
disp = ret1.std(axis=1)

def rep(s):
    return pd.DataFrame(np.repeat(s.values[:, None], N, axis=1),
                        index=close.index, columns=close.columns)

volr = qv / qv.rolling(60, min_periods=20).median()
ext_vol = (volr > 5).astype('float32')
feats = {
    'ret1':    ret1, 'ret5': close.pct_change(5),
    'ret15':   close.pct_change(15), 'ret60': close.pct_change(60),
    'volr':    volr,
    'dist_ma': close / close.rolling(60, min_periods=30).mean() - 1,
    'mkt1':    rep(mkt1), 'disp': rep(disp),
    'mktvol':  rep(mkt1.rolling(60, min_periods=20).std()),
    'ext_vol': ext_vol, 'n_pump': rep(ext_vol.sum(axis=1).astype('float32')),
    'qv_rank': qv.rank(axis=1, pct=True),
}
feats['rk_ret'] = feats['ret15'].rank(axis=1, pct=True)
names = list(feats.keys())

EDGES = [-np.inf, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, np.inf]
nxt = close.shift(-1) / close - 1.0
R = nxt.values.ravel().astype('float32')
Y = np.digitize(R, EDGES[1:-1]).astype('int8')

X = np.stack([feats[n].values.ravel() for n in names], axis=1).astype('float32')
t_idx_all = np.repeat(np.arange(T), N)
ok = np.isfinite(X).all(axis=1) & np.isfinite(R)
del feats; gc.collect()
X, Y, R = X[ok], Y[ok], R[ok]
t_idx = t_idx_all[ok]
X = np.ascontiguousarray(X)
del ok, t_idx_all; gc.collect()
say('可用样本 %s' % format(len(X), ','))

a = int(len(X) * 0.6)
TRMAX = 6000000
if a > TRMAX:
    tr = np.random.default_rng(7).choice(a, TRMAX, replace=False); tr.sort()
else:
    tr = slice(0, a)
m = HistGradientBoostingClassifier(max_iter=70, learning_rate=0.08, max_depth=6,
                                   max_bins=128, min_samples_leaf=200,
                                   l2_regularization=1.0, random_state=42,
                                   early_stopping=False)
m.fit(X[tr], Y[tr]); del tr; gc.collect()
say('训练完成')

b = len(X)
P = np.zeros((b - a, 8), dtype='float32')
STEP = 2000000
for s0 in range(a, b, STEP):
    s1 = min(s0 + STEP, b)
    pp = m.predict_proba(X[s0:s1])
    tmp = np.zeros((s1 - s0, 8), dtype='float32')
    for ci, cls in enumerate(m.classes_):
        tmp[:, int(cls)] = pp[:, ci]
    P[s0 - a:s1 - a] = tmp
del X; gc.collect()

Rv = R[a:b].astype('float64')
Tv = t_idx[a:b]
del R, Y, t_idx; gc.collect()

pup05 = P[:, [6, 7]].sum(axis=1).astype('float64')
pdn05 = P[:, [0, 1]].sum(axis=1).astype('float64')
del P; gc.collect()

sig_l = pup05 >= TL
sig_s = pdn05 >= TS
say('测试样本 %s   做多信号 %s   做空信号 %s   合计 %s'
    % (format(len(Rv), ','), format(int(sig_l.sum()), ','),
       format(int(sig_s.sum()), ','), format(int((sig_l | sig_s).sum()), ',')))

pr_l = Rv - 2 * FEE
pr_s = -Rv - 2 * FEE
mask = sig_l | sig_s
t_ = Tv[mask]
r_ = np.where(sig_l[mask], pr_l[mask], pr_s[mask])

uniq, inv = np.unique(t_, return_inverse=True)
sums = np.bincount(inv, weights=r_)
cnts = np.bincount(inv)
port = sums / cnts
dts = close.index[uniq]                      # uniq 本身即时间行号，直接取时间戳
span_min = (dts[-1] - dts[0]).total_seconds() / 60.0

say('\n' + '=' * 100)
say('【一】单笔统计（扣 maker 双边 %.2f%%）' % (2 * FEE * 100))
say('=' * 100)
for nm, sg, pr in [('做多', sig_l, pr_l), ('做空', sig_s, pr_s)]:
    n = int(sg.sum())
    if n == 0: continue
    say('  %s  笔数 %10s  E[净]/笔 %+.4f%%  中位 %+.4f%%  胜率 %.2f%%'
        % (nm, format(n, ','), pr[sg].mean() * 100, np.median(pr[sg]) * 100,
           (pr[sg] > 0).mean() * 100))
say('  合计 %10s 笔  E[净]/笔 %+.4f%%' % (format(int(mask.sum()), ','), r_.mean() * 100))

say('\n【二】组合资金曲线（按分钟等权全押所有信号）')
say('=' * 100)
nav = np.cumprod(1.0 + port)
tot = nav[-1] - 1.0
yrs = span_min / MIN_PER_YEAR          # 用真实时间跨度算年化
ann = nav[-1] ** (1 / yrs) - 1 if yrs > 0 else float('nan')
shp = port.mean() / port.std() * np.sqrt(MIN_PER_YEAR) if port.std() > 0 else float('nan')
dd = float((nav / np.maximum.accumulate(nav) - 1).min())
say('  有信号分钟数 = %s（占总跨度分钟 %.2f%%）  时间跨度 %.1f 天（%s ~ %s）'
    % (format(len(port), ','), len(port) / max(1.0, span_min) * 100,
       span_min / 1440, dts[0], dts[-1]))
say('  总收益 = %+.1f%%   年化 = %+.1f%%   夏普 = %.2f   最大回撤 = %.1f%%'
    % (tot * 100, ann * 100, shp, dd * 100))
say('  每分钟平均信号数 = %.2f   单分钟最多 = %d'
    % (cnts.mean(), int(cnts.max())))
say('  注：在「有信号的分钟」上连续复利，无信号分钟按 0 计')

say('\n【三】分月稳定性（月度累计净收益 %）')
say('=' * 100)
s_port = pd.Series(port, index=dts)
mo = s_port.groupby(s_port.index.strftime('%Y-%m')).agg(['count', 'sum', 'mean'])
pos = 0
for k, row in mo.iterrows():
    if row['sum'] > 0: pos += 1
    say('  %s  %6d 分钟   月累计 %+7.3f%%   均 %+.4f%%'
        % (k, int(row['count']), row['sum'] * 100, row['mean'] * 100))
say('  -> 正收益月 = %d/%d' % (pos, len(mo)))

say('\n【四】阈值敏感性（同口径）')
say('=' * 100)
say('  %-6s %-6s %10s %12s %12s' % ('TL', 'TS', '笔数', '总收益%', '年化%'))
for tl, ts in [(0.30, 0.25), (0.35, 0.28), (0.39, 0.31), (0.45, 0.35), (0.50, 0.40)]:
    sl = pup05 >= tl; ss = pdn05 >= ts
    mk = sl | ss
    if mk.sum() < 100: continue
    rr = np.where(sl[mk], pr_l[mk], pr_s[mk])
    tt = Tv[mk]
    u2, i2 = np.unique(tt, return_inverse=True)
    p2 = np.bincount(i2, weights=rr) / np.bincount(i2)
    nv2 = np.cumprod(1 + p2)
    y2 = len(p2) / MIN_PER_YEAR
    a2 = nv2[-1] ** (1 / y2) - 1 if y2 > 0 else float('nan')
    say('  %-6.2f %-6.2f %10s %+11.1f%% %+11.1f%%'
        % (tl, ts, format(int(mk.sum()), ','), (nv2[-1] - 1) * 100, a2 * 100))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
ax = axes[0]
ax.plot(dts, nav, lw=1.4)
ax.set_yscale('log')
ax.set_title('组合净值（等权全押 / maker 双边 %.2f%%）' % (2 * FEE * 100))
ax.set_ylabel('净值（对数轴）'); ax.grid(alpha=.3)
ax = axes[1]
cols = ['C2' if v > 0 else 'C3' for v in mo['sum']]
ax.bar(range(len(mo)), mo['sum'].values * 100, color=cols)
ax.set_xticks(range(len(mo)))
ax.set_xticklabels(mo.index, rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='k', lw=1)
ax.set_title('月度累计净收益 %'); ax.grid(alpha=.3)
plt.tight_layout()
p_ = os.path.join(EV, 'edge_backtest.png')
plt.savefig(p_, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p_)
say('TOTAL done')
