# -*- coding: utf-8 -*-
"""分钟级概率模型（目标：高概率区真实胜率 >= 75%）
数据：klines_1m_1y.csv（60 币 x 365 天 x 1m）
方法：严格 walk-forward（训练=历史，测试=其后一段），禁止随机划分
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV   = r'D:\中信证券\coinlab\evidence'
FWD  = 5          # 预测未来 5 分钟方向（分钟级）
DEP  = 0.02       # 连续段门槛

def say(*a):
    print(*a, flush=True)

say('=' * 96)
say('分钟级概率模型  数据=klines_1m_1y.csv  目标=高概率区真实胜率>=75%')
say('=' * 96)

# ---------- 1 读 ----------
f = os.path.join(DATA, 'klines_1m_1y.csv')
say('读 %s' % f)
df = pd.read_csv(f, usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
say('行数=%d  币数=%d  跨度 %s ~ %s' % (len(df), df.symbol.nunique(),
      df.dt.min(), df.dt.max()))
df = df.sort_values(['symbol', 'open_time'])

close = df.pivot(index='dt', columns='symbol', values='close').astype('float32')
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
close = close.ffill(limit=5)
T, N = close.shape
say('matrix %d x %d = %.1fM 单元' % (T, N, T * N / 1e6))

# ---------- 2 特征 ----------
ret1 = close.pct_change()
mkt1 = ret1.median(axis=1)                      # 市场收益（中位）
disp = ret1.std(axis=1)                         # 横截面离散度

def roll_sum(s, w):
    return s.rolling(w, min_periods=max(2, w // 2)).sum()

def roll_std(s, w):
    return s.rolling(w, min_periods=max(2, w // 2)).std()

feats = {}
feats['ret1']   = ret1
feats['ret5']   = close.pct_change(5)
feats['ret15']  = close.pct_change(15)
feats['ret60']  = close.pct_change(60)
feats['ret240'] = close.pct_change(240)

# 量比：本分钟量 / 过去 60 分钟量中位
qv60 = qv.rolling(60, min_periods=20).median()
feats['volr'] = qv / qv60

# 离均线
ma60 = close.rolling(60, min_periods=30).mean()
feats['dist_ma'] = close / ma60 - 1

# 连续段（跌/涨，用 ret15 判定）
up = (close.pct_change(15) > 0).astype('int8')
grp_u = (up != up.shift()).cumsum()
seglen_up = up.groupby(grp_u).cumcount() + 1
seglen_up = seglen_up.where(up == 1, -seglen_up)   # 涨段为正、跌段为负
seglen_up.index = close.index
feats['seglen'] = seglen_up.values.astype('float32').reshape(-1, 1) * np.ones((1, N))

# 市场 & 状态
feats['mkt1']   = pd.DataFrame(np.repeat(mkt1.values[:, None], N, axis=1),
                               index=close.index, columns=close.columns)
feats['disp']   = pd.DataFrame(np.repeat(disp.values[:, None], N, axis=1),
                               index=close.index, columns=close.columns)
feats['mktvol'] = pd.DataFrame(np.repeat(
    roll_std(mkt1, 60).values[:, None], N, axis=1),
    index=close.index, columns=close.columns)

# 横截面排名（每个时刻内 ret15 / volr 的百分位）
feats['rk_ret']  = feats['ret15'].rank(axis=1, pct=True)
feats['rk_volr'] = feats['volr'].rank(axis=1, pct=True)

# 时间特征
hh = close.index.hour.to_numpy()
dd = close.index.dayofweek.to_numpy()
feats['hour'] = pd.DataFrame(np.repeat(hh[:, None], N, axis=1),
                             index=close.index, columns=close.columns)
feats['dow']  = pd.DataFrame(np.repeat(dd[:, None], N, axis=1),
                             index=close.index, columns=close.columns)

names = ['ret1','ret5','ret15','ret60','ret240','volr','dist_ma','seglen',
         'mkt1','disp','mktvol','rk_ret','rk_volr','hour','dow']
say('特征 %d 个：%s' % (len(names), names))

# ---------- 3 标签 ----------
fwd = close.shift(-FWD) / close - 1.0
y = (fwd > 0).astype('int8')

X = np.stack([feats[n].values.ravel() for n in names], axis=1).astype('float32')
Y = y.values.ravel().astype('int8')
RV = fwd.values.ravel().astype('float32')      # 用于算收益

ok = np.isfinite(X).all(axis=1) & np.isfinite(RV)
X, Y, RV = X[ok], Y[ok], RV[ok]
say('可用样本 = %s' % format(len(X), ','))

# ---------- 4 walk-forward ----------
NSEG = 4
edges = np.linspace(0, len(X), NSEG + 1).astype(int)
say('\n' + '=' * 96)
say('【walk-forward】训练=历史，测试=其后 1/%d 段（绝不随机划分）' % (NSEG + 1))
say('=' * 96)
say('  测试段  训练n  测试n  AUC  p>0.5胜率  p>0.6胜率  p>0.7胜率  p>0.7净maker')

res = []
test_p = np.full(len(X), np.nan, dtype='float32')

for k in range(1, NSEG + 1):
    tr_lo, tr_hi = 0, edges[k]
    te_lo, te_hi = edges[k], edges[k + 1]
    if tr_hi - tr_lo < 100000 or te_hi - te_lo < 50000:
        continue
    m = HistGradientBoostingClassifier(
        max_iter=90, learning_rate=0.08, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0,
        random_state=42, early_stopping=False)
    m.fit(X[tr_lo:tr_hi], Y[tr_lo:tr_hi])
    p = m.predict_proba(X[te_lo:te_hi])[:, 1].astype('float32')
    test_p[te_lo:te_hi] = p
    auc = roc_auc_score(Y[te_lo:te_hi], p)
    rv = RV[te_lo:te_hi]; yv = Y[te_lo:te_hi]
    def w(mask):
        n = int(mask.sum())
        if n < 20:
            return n, float('nan'), float('nan')
        return n, float(yv[mask].mean() * 100), float(rv[mask].mean() * 100 - 0.02)
    n5, w5, _ = w(p > 0.5)
    n6, w6, _ = w(p > 0.6)
    n7, w7, g7 = w(p > 0.7)
    res.append((k, tr_hi - tr_lo, te_hi - te_lo, auc, w5, w6, w7, g7))
    say('  #%d  %9s %9s  %.4f  %6.1f%%  %6.1f%%  %6.1f%%  %+7.4f%%'
        % (k, format(tr_hi - tr_lo, ','), format(te_hi - te_lo, ','),
           auc, w5, w6, w7, g7))

if res:
    auc_all = roc_auc_score(Y[np.isfinite(test_p)], test_p[np.isfinite(test_p)])
    say('  合并 AUC = %.4f   有模型样本 %s'
        % (auc_all, format(int(np.isfinite(test_p).sum()), ',')))

# ---------- 5 概率分档（合并） ----------
mask = np.isfinite(test_p)
P = test_p[mask]; Yv = Y[mask]; RVv = RV[mask]
say('\n' + '=' * 96)
say('【概率分档 —— 回答「到 75% 需要多高概率、多少样本」】')
say('=' * 96)
say('  %-12s %10s %10s %12s %12s %12s'
    % ('概率区间', '样本', '实际胜率', '毛收益/笔', '净maker/笔', '净taker/笔'))
bins = [0, .40, .45, .50, .55, .60, .65, .70, .75, .80, 1.01]
for a, b in zip(bins[:-1], bins[1:]):
    m = (P >= a) & (P < b)
    n = int(m.sum())
    if n < 50:
        continue
    say('  %-12s %10s %9.2f%% %11.4f%% %11.4f%% %11.4f%%'
        % ('[%.2f,%.2f)' % (a, b), format(n, ','), Yv[m].mean() * 100,
           RVv[m].mean() * 100, RVv[m].mean() * 100 - 0.02,
           RVv[m].mean() * 100 - 0.20))

# ---------- 6 阈值扫描：找 >=75% 的截点 ----------
say('\n' + '=' * 96)
say('【阈值扫描 —— 从高到低放宽，看什么时候胜率跌破 75%/70%/65%】')
say('=' * 96)
say('  %-10s %10s %10s %12s %12s'
    % ('p>=', '样本', '实际胜率', '毛/笔', '净maker/笔'))
th = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
for t in th:
    m = P >= t
    n = int(m.sum())
    if n < 30:
        say('  %-10.2f %10s   (样本不足)' % (t, format(n, ',')))
        continue
    say('  %-10.2f %10s %9.2f%% %11.4f%% %11.4f%%'
        % (t, format(n, ','), Yv[m].mean() * 100,
           RVv[m].mean() * 100, RVv[m].mean() * 100 - 0.02))

hit75 = None
for t in np.arange(0.50, 0.96, 0.005):
    m = P >= t
    if m.sum() >= 50 and Yv[m].mean() >= 0.75:
        hit75 = t
        break
if hit75 is not None:
    m = P >= hit75
    say('\n  ★ 首次达到 >=75%% 胜率的阈值 = p >= %.3f' % hit75)
    say('     样本 %s 笔（占测试区 %.4f%%）  胜率 %.2f%%  净maker %+.4f%%'
        % (format(int(m.sum()), ','), m.sum() / len(P) * 100,
           Yv[m].mean() * 100, RVv[m].mean() * 100 - 0.02))
else:
    mx = 0
    for t in np.arange(0.50, 0.99, 0.005):
        m = P >= t
        if m.sum() >= 50:
            mx = max(mx, Yv[m].mean() * 100)
    say('\n  ✗ 全阈值范围内最高概率区的胜率只到 %.2f%% —— 未达 75%%' % mx)

# ---------- 7 时间外推：高概率区按月分布 ----------
if hit75 is not None:
    t = hit75
    m = P >= t
    idx = close.index[ok][mask][m]
    mo = pd.Series(idx).dt.strftime('%Y-%m').values
    s = pd.Series(Yv[m]).groupby(mo).agg(['count', 'mean'])
    say('\n【高概率区（p>=%.2f）按月分布 —— 看是否只在个别月份有效】' % t)
    for k_, r_ in s.iterrows():
        say('  %s  n=%5d  胜率=%.1f%%' % (k_, r_['count'], r_['mean'] * 100))
    pos = int((s['mean'] > 0.5).sum())
    say('  → 胜率>50%% 的月份 = %d/%d' % (pos, len(s)))

# ---------- 8 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
ax = axes[0]
ax.hist(P[Yv == 1], bins=60, alpha=.55, label='实际涨', density=True)
ax.hist(P[Yv == 0], bins=60, alpha=.55, label='实际跌', density=True)
ax.set_xlabel('model p(up)'); ax.set_ylabel('density')
ax.set_title('预测概率分布（分钟级 FWD=%d）' % FWD); ax.legend(fontsize=9)

ax = axes[1]
xs, ys = [], []
for a, b in zip(bins[:-1], bins[1:]):
    m_ = (P >= a) & (P < b)
    if m_.sum() >= 50:
        xs.append((a + b) / 2); ys.append(Yv[m_].mean() * 100)
ax.plot(xs, ys, 'o-', lw=2)
ax.axhline(75, ls='--', lw=1.6, label='目标 75%')
ax.axhline(50, ls=':', lw=1.2, label='掷硬币 50%')
ax.set_xlabel('model p'); ax.set_ylabel('实际胜率 %')
ax.set_title('校准曲线'); ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[2]
ns, ws = [], []
for t in th:
    m_ = P >= t
    if m_.sum() >= 30:
        ns.append(m_.sum()); ws.append(Yv[m_].mean() * 100)
ax.plot(ns, ws, 'o-', lw=2, color='C3')
ax.axhline(75, ls='--', lw=1.6, label='目标 75%')
ax.set_xscale('log'); ax.set_xlabel('样本数（p 阈值越严、样本越少）')
ax.set_ylabel('实际胜率 %')
ax.set_title('阈值 vs 样本 vs 胜率'); ax.legend(fontsize=9); ax.grid(alpha=.3)

plt.tight_layout()
p_ = os.path.join(EV, 'model_proba_1m.png')
plt.savefig(p_, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p_)
say('TOTAL done')
