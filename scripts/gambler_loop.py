# -*- coding: utf-8 -*-
"""gambler 闭环回测 —— 回答"跟随操盘手，什么时候下注、什么时候退出、能不能赚钱"

闭环规则（全部只用 t 时刻可见信息）：
  入场：P(涨·未来2h≥+20%) ≥ 0.70  且  P(崩·未来2h≤−20%) < 0.50   → 按当根收盘价买入
  退出：之后任一时刻 P(崩) ≥ 0.50  → 卖出；或持有满 24 根（2h）→ 卖出
  成本：maker 双边 0.04%（保守起见也报 taker 0.20% 口径）
  对照：同一时段"随机持有同样根数"的基线

数据：klines_5m_502.csv（304 币 × 3 年 5m）
切分：前 60% 时间训练模型，后 40% 时间做闭环（严格 walk-forward）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
W5, THR, PRE, POST, FWD = 48, 0.5, 48, 24, 24
DN, UP = -0.20, 0.20
P_ENTER, P_EXIT, MAXHOLD = 0.70, 0.50, 24
FEE_M, FEE_T = 0.0004, 0.0020
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 闭环回测：入场 P(涨)≥%.2f → 退出 P(崩)≥%.2f 或持满 %d 根' % (P_ENTER, P_EXIT, MAXHOLD))
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
def _piv(c):
    m = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
    return m.where(m > 0).ffill(limit=3)
close, high, low = _piv('close'), _piv('high'), _piv('low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
qv = qv.where(qv >= 0)
del df; gc.collect()
T, N = close.shape
tm = close.index.values
CV, HV, LV = close.values, high.values, low.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
qv60 = pd.DataFrame(qv).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)
volr = qv.values / qv60
ret60 = np.full((T, N), np.nan, np.float32); ret60[60:] = CV[60:] / CV[:-60] - 1.0
ret240 = np.full((T, N), np.nan, np.float32); ret240[240:] = CV[240:] / CV[:-240] - 1.0
vol240 = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
d4 = (np.abs(ret1) > 3.0 * vol240).astype(np.float32)
sgn = np.sign(np.nan_to_num(ret1))
d1 = np.zeros((T, N), np.float32); d7 = np.zeros((T, N), np.float32)
idxr = np.arange(T)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (idxr - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
    fz = np.r_[False, CV[1:, j] == CV[:-1, j]]
    lf = np.maximum.accumulate(np.where(~fz, idxr, -1))
    d7[:, j] = np.where(fz, idxr - lf, 0)
d8 = np.repeat(pd.Series((ret1 == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
fwd_min = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fwd_max = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gainW = (fwd_high / close - 1.0).values

# 事件（局）+ 样本
P = gainW >= THR
ev = []
for j in range(N):
    col = P[:, j]
    if not col.any():
        continue
    starts = np.where(col & ~np.r_[False, col[:-1]])[0]
    last = -10 ** 9
    for i in starts:
        if i - last < W5:
            continue
        pk = i + int(np.nanargmax(HV[i:i + W5, j]))
        ev.append((j, int(i), int(pk))); last = int(i)
rf, cf, evi = [], [], []
for k, (j, i, pk) in enumerate(ev):
    for t in range(max(0, i - PRE), min(T - 1, pk + POST)):
        if not (np.isfinite(CV[t, j]) and np.isfinite(fwd_min[t, j]) and np.isfinite(fwd_max[t, j])):
            continue
        rf.append(t); cf.append(j); evi.append(k)
rf = np.array(rf); cf = np.array(cf); evi = np.array(evi)
I = np.array([ev[k][1] for k in evi])
c0 = CV[rf, cf]
X = np.column_stack([
    (rf - I).astype('float32'), c0 / CV[I, cf] - 1.0, HV[rf, cf] / CV[I, cf] - 1.0,
    volr[rf, cf], d1[rf, cf], d4[rf, cf], d7[rf, cf],
    ret60[rf, cf], ret240[rf, cf], vol240[rf, cf], d8[rf, cf],
]).astype('float32')
yup = (fwd_max[rf, cf] / c0 - 1.0 >= UP).astype(np.int8)
ydn = (fwd_min[rf, cf] / c0 - 1.0 <= DN).astype(np.int8)
say('局 %d 个 / 样本 %s  %.0fs' % (len(ev), format(len(X), ','), time.time() - t0))

cut = int(T * 0.6)
tr = rf < cut; te = ~tr
mu = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_depth=4,
                                    min_samples_leaf=100, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], yup[tr])
md = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_depth=4,
                                    min_samples_leaf=100, l2_regularization=1.0,
                                    random_state=42, early_stopping=False).fit(X[tr], ydn[tr])
Pup = np.full((T, N), np.nan, np.float32); Pdn = np.full((T, N), np.nan, np.float32)
Pup[rf[te], cf[te]] = mu.predict_proba(X[te])[:, 1]
Pdn[rf[te], cf[te]] = md.predict_proba(X[te])[:, 1]
say('概率图完成（测试段 = 后 40%% 时间）  %.0fs' % (time.time() - t0))

# ---- 闭环模拟：一币同时只持一笔 ----
trades = []
busy_until = np.full(N, -1)
for t in range(cut, T - 1):
    for j in np.where(np.isfinite(Pup[t]))[0]:
        if t <= busy_until[j]:
            continue
        if not (Pup[t, j] >= P_ENTER and Pdn[t, j] < 0.5):
            continue
        entry = t
        exit_t = min(T - 1, t + MAXHOLD); reason = 'timeout'
        for u in range(t + 1, min(T - 1, t + MAXHOLD) + 1):
            if np.isfinite(Pdn[u, j]) and Pdn[u, j] >= P_EXIT:
                exit_t = u; reason = 'signal'; break
        r = float(CV[exit_t, j] / CV[entry, j] - 1.0)
        trades.append((entry, j, exit_t, exit_t - entry, r, reason))
        busy_until[j] = exit_t
say('闭环交易数 = %d  %.0fs' % (len(trades), time.time() - t0))

if trades:
    tr_arr = np.array([(t[3], t[4]) for t in trades])
    hold, ret = tr_arr[:, 0], tr_arr[:, 1]
    say('\n' + '=' * 100)
    say('【闭环结果】测试段（后 40%% 时间，304 币）')
    say('=' * 100)
    say('  交易笔数 %d ；平均持有 %.1f 根（%.0f 分钟）；每天 %.1f 笔'
        % (len(trades), hold.mean(), hold.mean() * 5, len(trades) / ((T - cut) * 5 / 1440.0)))
    say('  胜率 %.1f%%；毛收益 均值 %+.3f%%  中位 %+.3f%%  P10 %+.2f%%  P90 %+.2f%%'
        % (100 * (ret > 0).mean(), 100 * ret.mean(), 100 * np.median(ret),
           100 * np.percentile(ret, 10), 100 * np.percentile(ret, 90)))
    say('  净收益（maker 双边 0.04%%）：均值 %+.3f%%/笔   合计 %+.1f%%'
        % (100 * (ret.mean() - FEE_M), 100 * (ret - FEE_M).sum()))
    say('  净收益（taker 双边 0.20%%）：均值 %+.3f%%/笔   合计 %+.1f%%'
        % (100 * (ret.mean() - FEE_T), 100 * (ret - FEE_T).sum()))
    say('  退出原因：信号 %d 笔 / 超时 %d 笔；信号退出平均收益 %+.3f%%，超时 %+.3f%%'
        % (sum(1 for x in trades if x[5] == 'signal'), sum(1 for x in trades if x[5] == 'timeout'),
           100 * np.mean([x[4] for x in trades if x[5] == 'signal']) if any(x[5] == 'signal' for x in trades) else 0,
           100 * np.mean([x[4] for x in trades if x[5] == 'timeout']) if any(x[5] == 'timeout' for x in trades) else 0))
    # 随机基线：同币同持有期
    rng = np.random.default_rng(0)
    base = []
    for _ in range(20000):
        j = int(rng.integers(0, N)); t = int(rng.integers(cut, T - MAXHOLD))
        base.append(float(CV[t + MAXHOLD, j] / CV[t, j] - 1.0))
    base = np.array(base)
    say('  对照：同币随机时刻持有 24 根 → 均值 %+.3f%%、胜率 %.1f%%'
        % (100 * base.mean(), 100 * (base > 0).mean()))
    pd.DataFrame(trades, columns=['entry_row', 'col', 'exit_row', 'hold', 'ret', 'reason']).to_csv(
        os.path.join(EV, 'gambler_trades.csv'), index=False, encoding='utf-8')
    say('\n  saved: gambler_trades.csv')

open(os.path.join(EV, 'gambler_loop.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
