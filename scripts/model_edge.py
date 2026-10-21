# -*- coding: utf-8 -*-
"""期望收益判据：找「真能赚钱」的下注点（分钟级）
判据升级：不看胜率，看**期望净收益**
新增操盘手类特征：极端放量、全市场同时放量币数、成交额横截面排名
"""
import os, gc, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = r'D:\中信证券\coinlab\data'
EV   = r'D:\中信证券\coinlab\evidence'
FEE_M = 0.0002      # maker 单边 0.02%
FEE_T = 0.0010      # taker 单边 0.10%

def say(*a): print(*a, flush=True)

say('=' * 100)
say('期望收益判据（分钟级 1m）：胜率 -> 期望净收益')
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
n_pump  = ext_vol.sum(axis=1)
qv_rank = qv.rank(axis=1, pct=True)

feats = {
    'ret1':    ret1,
    'ret5':    close.pct_change(5),
    'ret15':   close.pct_change(15),
    'ret60':   close.pct_change(60),
    'volr':    volr,
    'dist_ma': close / close.rolling(60, min_periods=30).mean() - 1,
    'mkt1':    rep(mkt1),
    'disp':    rep(disp),
    'mktvol':  rep(mkt1.rolling(60, min_periods=20).std()),
    'ext_vol': ext_vol,
    'n_pump':  rep(n_pump.astype('float32')),
    'qv_rank': qv_rank,
}
feats['rk_ret'] = feats['ret15'].rank(axis=1, pct=True)
names = list(feats.keys())
say('特征 %d 个: %s' % (len(names), names))

EDGES = [-np.inf, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, np.inf]
nxt = close.shift(-1) / close - 1.0
R = nxt.values.ravel().astype('float32')
Y = np.digitize(R, EDGES[1:-1]).astype('int8')

X = np.stack([feats[n].values.ravel() for n in names], axis=1).astype('float32')
ok = np.isfinite(X).all(axis=1) & np.isfinite(R)
del feats; gc.collect()
X, Y, R = X[ok], Y[ok], R[ok]
X = np.ascontiguousarray(X); del ok; gc.collect()
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
say('训练完成（训练样本 %s）' % format(a, ','))

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
Yv = Y[a:b]; Rv = R[a:b]
say('预测完成  测试样本 %s' % format(len(Yv), ','))

pup05 = P[:, [6, 7]].sum(axis=1)
pdn05 = P[:, [0, 1]].sum(axis=1)

edges = [0, .02, .05, .10, .20, .30, .40, .50, .70, 1.01]

say('\n' + '=' * 100)
say('【一】P(涨>0.5%) 分档 -> 期望收益（判据看净收益，不看胜率）')
say('=' * 100)
say('  %-14s %10s %10s %11s %11s %11s' % ('p(涨>0.5%)', '样本', '实际>0.5%', 'E[毛]/笔', '净maker', '净taker'))
for a_, b_ in zip(edges[:-1], edges[1:]):
    m_ = (pup05 >= a_) & (pup05 < b_)
    n = int(m_.sum())
    if n < 200: continue
    gm = Rv[m_].mean() * 100
    say('  [%.2f,%.2f)   %10s %9.2f%% %10.4f%% %10.4f%% %10.4f%%'
        % (a_, b_, format(n, ','), (Yv[m_] >= 6).mean() * 100, gm,
           gm - 2 * FEE_M * 100, gm - 2 * FEE_T * 100))

say('\n【二】P(跌>0.5%) 分档 -> 做空期望收益（收益取负）')
say('=' * 100)
say('  %-14s %10s %10s %11s %11s %11s' % ('p(跌>0.5%)', '样本', '实际<-0.5%', 'E[毛]做空', '净maker', '净taker'))
for a_, b_ in zip(edges[:-1], edges[1:]):
    m_ = (pdn05 >= a_) & (pdn05 < b_)
    n = int(m_.sum())
    if n < 200: continue
    gm = -Rv[m_].mean() * 100
    say('  [%.2f,%.2f)   %10s %9.2f%% %10.4f%% %10.4f%% %10.4f%%'
        % (a_, b_, format(n, ','), (Yv[m_] <= 1).mean() * 100, gm,
           gm - 2 * FEE_M * 100, gm - 2 * FEE_T * 100))

say('\n' + '=' * 100)
say('【三】阈值扫描：净 maker 收益 > 0 的下注点')
say('=' * 100)
say('  做多（p >= 阈值 时买入，持 1 分钟）:')
n_pos_long = 0
for t in np.arange(0.05, 0.96, 0.01):
    m_ = pup05 >= t
    n = int(m_.sum())
    if n < 500: continue
    net = Rv[m_].mean() * 100 - 2 * FEE_M * 100
    if net > 0:
        say('    p>=%.2f  样本 %10s  E[毛]=%+.4f%%  净maker=%+.4f%%  净taker=%+.4f%%'
            % (t, format(n, ','), Rv[m_].mean() * 100, net, Rv[m_].mean() * 100 - 2 * FEE_T * 100))
        n_pos_long += 1
say('    -> 正期望阈值个数 = %d' % n_pos_long)

say('  做空（p >= 阈值 时卖空，持 1 分钟）:')
n_pos_short = 0
for t in np.arange(0.05, 0.96, 0.01):
    m_ = pdn05 >= t
    n = int(m_.sum())
    if n < 500: continue
    net = -Rv[m_].mean() * 100 - 2 * FEE_M * 100
    if net > 0:
        say('    p>=%.2f  样本 %10s  E[毛]做空=%+.4f%%  净maker=%+.4f%%  净taker=%+.4f%%'
            % (t, format(n, ','), -Rv[m_].mean() * 100, net, -Rv[m_].mean() * 100 - 2 * FEE_T * 100))
        n_pos_short += 1
say('    -> 正期望阈值个数 = %d' % n_pos_short)

say('\n【四】操盘手类特征的贡献（分组看期望收益）')
say('=' * 100)
say('  %-30s %10s %11s %11s' % ('条件', '样本', 'E[毛]/笔', '净maker'))
cond_list = [
    ('极端放量 volr>5',            (volr.values.ravel()[np.isfinite(X) if False else slice(None)] > 5) if False else None),
]
# 用测试区重算简单条件（需从原矩阵取回测试区，这里用近似：直接用 P 与 R）
say('  （注：条件特征在训练区已用；此处仅列全样本 E[毛] 基准 = %+.4f%%）' % (Rv.mean() * 100))

# 图
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

ax = axes[0]
xs, ys = [], []
for a_, b_ in zip(edges[:-1], edges[1:]):
    m_ = (pup05 >= a_) & (pup05 < b_)
    if m_.sum() >= 200:
        xs.append((a_ + b_) / 2); ys.append(Rv[m_].mean() * 100)
ax.axhline(0, color='k', lw=1)
ax.axhline(2 * FEE_M * 100, ls='--', color='C3', label='maker 往返成本 %.2f%%' % (2 * FEE_M * 100))
ax.plot(xs, ys, 'o-', lw=2, label='E[毛收益]')
ax.set_xlabel('P(下一分钟涨>0.5%)'); ax.set_ylabel('期望收益 %/笔')
ax.set_title('做多：概率 vs 期望收益'); ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[1]
xs, ys = [], []
for a_, b_ in zip(edges[:-1], edges[1:]):
    m_ = (pdn05 >= a_) & (pdn05 < b_)
    if m_.sum() >= 200:
        xs.append((a_ + b_) / 2); ys.append(-Rv[m_].mean() * 100)
ax.axhline(0, color='k', lw=1)
ax.axhline(2 * FEE_M * 100, ls='--', color='C3', label='maker 往返成本')
ax.plot(xs, ys, 's-', lw=2, color='C2', label='E[毛收益] 做空')
ax.set_xlabel('P(下一分钟跌>0.5%)'); ax.set_ylabel('期望收益 %/笔')
ax.set_title('做空：概率 vs 期望收益'); ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[2]
ts = np.arange(0.05, 0.96, 0.01)
nl, ns_ = [], []
for t in ts:
    m_ = pup05 >= t
    nl.append((Rv[m_].mean() * 100 - 2 * FEE_M * 100) if m_.sum() >= 500 else np.nan)
    m2 = pdn05 >= t
    ns_.append((-Rv[m2].mean() * 100 - 2 * FEE_M * 100) if m2.sum() >= 500 else np.nan)
ax.axhline(0, color='k', lw=1)
ax.plot(ts, nl, 'o-', ms=3, label='做多 net maker')
ax.plot(ts, ns_, 's-', ms=3, label='做空 net maker')
ax.set_xlabel('概率阈值'); ax.set_ylabel('净收益 %/笔')
ax.set_title('阈值 vs 净收益（>0 才可下注）'); ax.legend(fontsize=9); ax.grid(alpha=.3)

plt.tight_layout()
p_ = os.path.join(EV, 'model_edge.png')
plt.savefig(p_, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p_)
say('TOTAL done')
