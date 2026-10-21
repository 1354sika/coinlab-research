# -*- coding: utf-8 -*-
"""分钟级「下一次价格变化」概率预测：方向 + 幅度分布
数据：klines_1m_1y.csv（60 币 x 365 天 x 1m）
预测对象：**下一分钟**收益 r_{t+1} 落在哪一档（8 档多分类）
输出：P(涨>0.2%) / P(涨>0.5%) / P(涨>1%) / P(跌同样) 的校准与 75% 达成情况
纪律：严格 walk-forward（训练=历史，测试=其后段），绝不随机划分
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = r'D:\中信证券\coinlab\data'
EV   = r'D:\中信证券\coinlab\evidence'

# 8 档：<=-1% / -1~-0.5 / -0.5~-0.2 / -0.2~0 / 0~0.2 / 0.2~0.5 / 0.5~1 / >1
EDGES = [-np.inf, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, np.inf]
LBL   = ['<-1%', '-1~-0.5%', '-0.5~-0.2%', '-0.2~0%',
         '0~0.2%', '0.2~0.5%', '0.5~1%', '>1%']
UP_BINS = [4, 5, 6, 7]      # 涨档索引
DN_BINS = [0, 1, 2, 3]      # 跌档索引

def say(*a):
    print(*a, flush=True)

say('=' * 100)
say('分钟级「下一次价格变化」概率预测：方向 + 幅度分布（8 档多分类）')
say('=' * 100)

# ---------- 1 读 ----------
f = os.path.join(DATA, 'klines_1m_1y.csv')
say('读 %s' % f)
df = pd.read_csv(f, usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
say('行数=%s  币数=%d' % (format(len(df), ','), df.symbol.nunique()))
df = df.sort_values(['symbol', 'open_time'])

close = df.pivot(index='dt', columns='symbol', values='close').astype('float32')
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
close = close.ffill(limit=5)
T, N = close.shape
say('matrix %d x %d = %.1fM 单元' % (T, N, T * N / 1e6))

# ---------- 2 特征（15 个） ----------
ret1 = close.pct_change()
mkt1 = ret1.median(axis=1)
disp = ret1.std(axis=1)

def rep(s):
    return pd.DataFrame(np.repeat(s.values[:, None], N, axis=1),
                        index=close.index, columns=close.columns)

feats = {
    'ret1':   ret1,
    'ret5':   close.pct_change(5),
    'ret15':  close.pct_change(15),
    'ret60':  close.pct_change(60),
    'volr':   qv / qv.rolling(60, min_periods=20).median(),
    'dist_ma': close / close.rolling(60, min_periods=30).mean() - 1,
    'mkt1':   rep(mkt1),
    'mkt5':   rep(mkt1.rolling(5, min_periods=2).sum()),
    'disp':   rep(disp),
    'mktvol': rep(mkt1.rolling(60, min_periods=20).std()),
}
feats['rk_ret']  = feats['ret15'].rank(axis=1, pct=True)
feats['rk_volr'] = feats['volr'].rank(axis=1, pct=True)

def seglen_1d(up):
    """O(n) 全向量化：连续同值段的长度（涨段为正、跌段为负）"""
    n = up.shape[0]
    chg = np.r_[True, up[1:] != up[:-1]]
    seg_id = np.cumsum(chg) - 1
    start = np.where(chg)[0][seg_id]
    run = (np.arange(n) - start + 1).astype('float32')
    return np.where(up, run, -run)

upv = (close.pct_change(15).values > 0)
segv = np.stack([seglen_1d(upv[:, j]) for j in range(N)], axis=1).astype('float32')
feats['seglen'] = pd.DataFrame(segv, index=close.index, columns=close.columns)
feats['hour']   = rep(pd.Series(close.index.hour, index=close.index).astype('float32'))
feats['dow']    = rep(pd.Series(close.index.dayofweek, index=close.index).astype('float32'))

names = list(feats.keys())
say('特征 %d 个' % len(names))

# ---------- 3 标签：下一分钟收益 8 档 ----------
nxt = close.shift(-1) / close - 1.0
R = nxt.values.ravel().astype('float32')
Y = np.digitize(R, EDGES[1:-1]).astype('int8')     # 0..7

X = np.stack([feats[n].values.ravel() for n in names], axis=1).astype('float32')
ok = np.isfinite(X).all(axis=1) & np.isfinite(R)
del feats, ret1, mkt1, disp, upv, segv
import gc as _gc; _gc.collect()
X, Y, R = X[ok], Y[ok], R[ok]
X = np.ascontiguousarray(X)
del ok; _gc.collect()
say('可用样本 = %s' % format(len(X), ','))

say('\n【下一分钟收益的实际分布（全样本，基准）】')
say('  %-12s %12s %10s' % ('档位', '样本', '占比'))
for i, l in enumerate(LBL):
    m = (Y == i)
    say('  %-12s %12s %9.3f%%' % (l, format(int(m.sum()), ','), m.mean() * 100))
p_up = ((Y >= 4)).mean() * 100
p_dn = ((Y <= 3)).mean() * 100
say('  → 下一分钟上涨概率 = %.2f%%   下跌概率 = %.2f%%' % (p_up, p_dn))

# ---------- 4 walk-forward 多分类 ----------
NSEG = 4
edges_i = np.linspace(0, len(X), NSEG + 2).astype(int)   # 6 个边界点 -> 5 段
say('\n' + '=' * 100)
say('【walk-forward】训练=历史，测试=其后 1/%d 段' % (NSEG + 1))
say('=' * 100)

PROB = np.full((len(X), 8), np.nan, dtype='float32')
TRAIN_MAX = 6000000       # 每段训练最多用 600 万（内存约束；GBDT 下远超充分）
for k in range(1, NSEG + 1):
    a, b = edges_i[k], edges_i[k + 1]
    if edges_i[k] < 200000 or b - a < 50000:
        continue
    m = HistGradientBoostingClassifier(
        max_iter=70, learning_rate=0.08, max_depth=6, max_bins=128,
        min_samples_leaf=200, l2_regularization=1.0,
        random_state=42, early_stopping=False)
    if a > TRAIN_MAX:
        tr = np.random.default_rng(1000 + k).choice(a, TRAIN_MAX, replace=False)
        tr.sort()
    else:
        tr = slice(0, a)
    m.fit(X[tr], Y[tr])
    del tr; _gc.collect()
    pp = np.empty((b - a, len(m.classes_)), dtype='float32')
    STEP = 2000000
    for s0 in range(a, b, STEP):
        s1 = min(s0 + STEP, b)
        pp[s0 - a:s1 - a] = m.predict_proba(X[s0:s1])
    out = np.zeros((b - a, 8), dtype='float32')
    for ci, cls in enumerate(m.classes_):
        out[:, int(cls)] = pp[:, ci]
    PROB[a:b] = out
    del pp, out; _gc.collect()
    say('  段%d 训练=%s 测试=%s 完成' % (k, format(a, ','), format(b - a, ',')))

msk = np.isfinite(PROB[:, 0])
P = PROB[msk]; Yv = Y[msk]; Rv = R[msk]
say('有模型样本 = %s' % format(int(msk.sum()), ','))

# ---------- 5 校准：P(涨>0.5%) 的可靠性 ----------
say('\n' + '=' * 100)
say('【校准 1】预测 P(下一分钟涨 >0.5%%) 的可靠性（这是"涨多少"的概率）')
say('=' * 100)
say('  %-14s %12s %12s %12s'
    % ('预测概率档', '样本', '实际涨>0.5%率', '实际均值'))
pup05 = P[:, [6, 7]].sum(axis=1)           # 严格 >0.5%：档 6,7 合计
for a_, b_ in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    m = (pup05 >= a_) & (pup05 < b_)
    n = int(m.sum())
    if n < 200:
        continue
    say('  [%.2f,%.2f)     %12s %11.2f%% %11.4f%%'
        % (a_, b_, format(n, ','), (Yv[m] >= 6).mean() * 100, Rv[m].mean() * 100))

say('\n【校准 2】预测 P(下一分钟涨 >0.2%%) 的可靠性')
pup02 = P[:, [5, 6, 7]].sum(axis=1)
for a_, b_ in [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    m = (pup02 >= a_) & (pup02 < b_)
    n = int(m.sum())
    if n < 200:
        continue
    say('  [%.2f,%.2f)     %12s %11.2f%% %11.4f%%'
        % (a_, b_, format(n, ','), (Yv[m] >= 5).mean() * 100, Rv[m].mean() * 100))

say('\n【校准 3】预测 P(下一分钟跌 >0.5%%) 的可靠性')
pdn05 = P[:, [0, 1]].sum(axis=1)
for a_, b_ in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.01)]:
    m = (pdn05 >= a_) & (pdn05 < b_)
    n = int(m.sum())
    if n < 200:
        continue
    say('  [%.2f,%.2f)     %12s %11.2f%% %11.4f%%'
        % (a_, b_, format(n, ','), (Yv[m] <= 1).mean() * 100, Rv[m].mean() * 100))

# ---------- 6 方向概率的校准 ----------
say('\n' + '=' * 100)
say('【校准 4】P(下一分钟上涨) 的分档可靠性 —— 回答「能不能到 75%%」')
say('=' * 100)
pup = P[:, UP_BINS].sum(axis=1)
say('  %-14s %12s %12s %12s' % ('预测涨概率', '样本', '实际上涨率', '净maker/笔'))
for a_, b_ in [(0, .3), (.3, .4), (.4, .45), (.45, .5), (.5, .55),
               (.55, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (pup >= a_) & (pup < b_)
    n = int(m.sum())
    if n < 200:
        continue
    say('  [%.2f,%.2f)     %12s %11.2f%% %11.4f%%'
        % (a_, b_, format(n, ','), (Yv >= 4).mean() * 100, Rv[m].mean() * 100 - 0.02))

say('\n【阈值扫描：P(涨) 从高到低，何时跌破 75%】')
hit = None
for t in np.arange(0.50, 0.98, 0.005):
    m = pup >= t
    if m.sum() >= 50 and (Yv[m] >= 4).mean() >= 0.75:
        hit = t
        break
for t in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
    m = pup >= t
    n = int(m.sum())
    if n < 30:
        say('  p>=%.2f   样本 %s（不足）' % (t, format(n, ',')))
        continue
    say('  p>=%.2f   样本 %11s  实际涨率 %6.2f%%  净maker %+.4f%%'
        % (t, format(n, ','), (Yv[m] >= 4).mean() * 100, Rv[m].mean() * 100 - 0.02))
if hit is not None:
    m = pup >= hit
    say('\n  ★ 首次达到 >=75%% 上涨率的阈值 = P(涨) >= %.3f' % hit)
    say('     样本 %s 笔（占 %.4f%%）  实际涨率 %.2f%%'
        % (format(int(m.sum()), ','), m.sum() / len(pup) * 100, (Yv[m] >= 4).mean() * 100))
else:
    best = 0
    for t in np.arange(0.5, 0.99, 0.005):
        m = pup >= t
        if m.sum() >= 50:
            best = max(best, (Yv[m] >= 4).mean() * 100)
    say('\n  ✗ 全阈值内最高概率区的上涨率只到 %.2f%% —— 未达 75%%' % best)

# ---------- 7 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

ax = axes[0]
xs, ys = [], []
for a_, b_ in [(0, .3), (.3, .4), (.4, .45), (.45, .5), (.5, .55),
               (.55, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (pup >= a_) & (pup < b_)
    if m.sum() >= 200:
        xs.append((a_ + b_) / 2); ys.append((Yv[m] >= 4).mean() * 100)
ax.plot([0, 1], [0, 100], 'k:', lw=1, label='完美校准')
ax.plot(xs, ys, 'o-', lw=2, color='C0')
ax.axhline(75, ls='--', lw=1.5, color='C3', label='目标 75%')
ax.set_xlabel('模型预测 P(下一分钟上涨)'); ax.set_ylabel('实际上涨率 %')
ax.set_title('方向概率校准'); ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[1]
xs, ys = [], []
for a_, b_ in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    m = (pup05 >= a_) & (pup05 < b_)
    if m.sum() >= 200:
        xs.append((a_ + b_) / 2); ys.append((Yv[m] >= 6).mean() * 100)
ax.plot([0, 1], [0, 100], 'k:', lw=1, label='完美校准')
ax.plot(xs, ys, 'o-', lw=2, color='C2')
ax.set_xlabel('模型预测 P(下一分钟涨>0.5%)'); ax.set_ylabel('实际上涨>0.5%率 %')
ax.set_title('幅度概率校准（涨>0.5%）'); ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[2]
ns, ws = [], []
for t in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
    m = pup >= t
    if m.sum() >= 30:
        ns.append(m.sum()); ws.append((Yv[m] >= 4).mean() * 100)
ax.plot(ns, ws, 'o-', lw=2, color='C3')
ax.axhline(75, ls='--', lw=1.5, label='目标 75%')
ax.set_xscale('log'); ax.set_xlabel('样本数'); ax.set_ylabel('上涨率 %')
ax.set_title('阈值 vs 样本 vs 上涨率'); ax.legend(fontsize=9); ax.grid(alpha=.3)

plt.tight_layout()
p_ = os.path.join(EV, 'model_next_move.png')
plt.savefig(p_, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p_)
say('TOTAL done')
