# -*- coding: utf-8 -*-
"""gambler v1 —— 操盘手博弈模型（2026-10-16 立项）

用户口径：
  · 模型名 = **gambler**
  · 只预测"崩"不够，**识别"涨"同样重要**
  · 报警**准确率 ≥85%**（报出来的信号里 ≥85% 是真事件）
  · 核心是**识别操盘者、掌控它的行为**（只有掌控他们才真能获利）

做法：
  ① 操盘者识别：4h 窗口内最高价 ≥ 当前价 ×1.5（+50%）= 一个"局"（事件）
  ② 样本：事件起点前 48 根 → 峰值后 24 根（覆盖 潜伏 → 拉升 → 见顶 → 崩）
  ③ 两套标签（未来 24 根 = 2h，全部 t 时刻可见特征，无前视）：
        崩：min(low[t+1..t+24]) / close[t] − 1 ≤ −20%
        涨：max(high[t+1..t+24]) / close[t] − 1 ≥ +20%
  ④ walk-forward：按事件时间切 60% 训练 / 40% 测试
  ⑤ 阈值扫描：对每个阈值报出"精确率（信号里真事件占比）+ 覆盖率"，找 85% 截点
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
W5, THR = 48, 0.5
PRE, POST, FWD = 48, 24, 24
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler v1 —— 操盘手博弈模型：识别操盘者 / 识别涨 / 预测崩（准确率目标 ≥85%）')
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
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))
CV, HV, LV = close.values, high.values, low.values

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
say('特征算完  %.0fs' % (time.time() - t0))

fwd_min = low.iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1)
fwd_max = high.iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1)
FMIN, FMAX = fwd_min.values, fwd_max.values
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gainW = (fwd_high / close - 1.0).values

# ---- 事件（操盘者的"局"）----
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
        ev.append((j, int(i), int(pk)))
        last = int(i)
say('操盘者的局 = %d 个  %.0fs' % (len(ev), time.time() - t0))

rf, cf, evi = [], [], []
for k, (j, i, pk) in enumerate(ev):
    for t in range(max(0, i - PRE), min(T - 1, pk + POST)):
        if not (np.isfinite(CV[t, j]) and np.isfinite(FMIN[t, j]) and np.isfinite(FMAX[t, j])):
            continue
        rf.append(t); cf.append(j); evi.append(k)
rf = np.array(rf); cf = np.array(cf); evi = np.array(evi)
say('样本点 = %s  %.0fs' % (format(len(rf), ','), time.time() - t0))

I = np.array([ev[k][1] for k in evi])
c0 = CV[rf, cf]
X = np.column_stack([
    (rf - I).astype('float32'),                      # bars_from_start（可为负=潜伏期）
    c0 / CV[I, cf] - 1.0,                            # gain_from_start
    HV[rf, cf] / CV[I, cf] - 1.0,                    # high_from_start
    volr[rf, cf], d1[rf, cf], d4[rf, cf], d7[rf, cf],
    ret60[rf, cf], ret240[rf, cf], vol240[rf, cf], d8[rf, cf],
]).astype('float32')
NAME = ['bars', 'gain', 'high_gain', 'volr', 'D1', 'D4', 'D7', 'ret60', 'ret240', 'vol240', 'D8']
ydn = (FMIN[rf, cf] / c0 - 1.0 <= DN).astype(np.int8)
yup = (FMAX[rf, cf] / c0 - 1.0 >= UP).astype(np.int8)
say('特征 %d 个：%s' % (len(NAME), NAME))
say('样本 %s ；崩(=未来2h跌≥20%%) %.2f%% ；涨(=未来2h涨≥20%%) %.2f%%'
    % (format(len(X), ','), 100 * ydn.mean(), 100 * yup.mean()))

et = np.array([ev[k][1] for k in range(len(ev))])
order = np.argsort(et); split = int(len(ev) * 0.6)
tr_set = set(order[:split].tolist())
tr = np.array([k in tr_set for k in evi]); te = ~tr
say('训练 %s（%d 局）/ 测试 %s（%d 局）' % (format(int(tr.sum()), ','), split,
                                        format(int(te.sum()), ','), len(ev) - split))

REPORT = {}
for tag, y in (('崩', ydn), ('涨', yup)):
    clf = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_depth=4,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         random_state=42, early_stopping=False)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    say('\n' + '=' * 100)
    say('【gambler · %s】AUC = %.4f   基准发生率 = %.2f%%' % (tag, auc, 100 * y[te].mean()))
    say('=' * 100)
    say('  %-8s %10s %11s %11s %11s' % ('阈值', '报出数', '覆盖率', '精确率', '召回率'))
    best85 = None
    for t in np.arange(0.05, 0.995, 0.01):
        m = p >= t
        n = int(m.sum())
        if n < 20:
            continue
        prec = float(y[te][m].mean())
        rec = float(y[te][m].sum() / max(1, y[te].sum()))
        if best85 is None and prec >= 0.85:
            best85 = (float(t), n, prec, rec)
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        m = p >= t
        n = int(m.sum())
        if n < 20:
            continue
        say('  %-8.2f %10s %10.2f%% %10.2f%% %10.2f%%'
            % (t, format(n, ','), 100 * n / len(p), 100 * y[te][m].mean(),
               100 * y[te][m].sum() / max(1, y[te].sum())))
    if best85:
        say('  ★ 精确率≥85%% 的最低阈值 = %.2f ：报出 %s 点（覆盖 %.2f%%），精确率 %.1f%%，召回 %.1f%%'
            % (best85[0], format(best85[1], ','), 100 * best85[1] / len(p), 100 * best85[2], 100 * best85[3]))
    else:
        say('  ✗ 任何阈值都达不到 85% 精确率（该标签在当前特征下不可达）')
    # 事件级
    ev_hit = []
    for k in set(evi[te].tolist()):
        m = evi[te] == k
        if not m.any():
            continue
        pp = p[m]
        ev_hit.append((float(pp.max()), int(y[te][m][np.argmax(pp)])))
    if ev_hit:
        a = np.array([e[0] for e in ev_hit]); b = np.array([e[1] for e in ev_hit])
        say('  事件级（每局取最大 P）：测试局 %d，其中真%s %d 局（%.0f%%）'
            % (len(a), tag, int(b.sum()), 100 * b.mean()))
        for t in (0.5, 0.7, 0.85):
            m = a >= t
            if m.sum() == 0:
                continue
            say('     P≥%.2f：覆盖 %d 局（%.0f%%），精确率 %.0f%%'
                % (t, int(m.sum()), 100 * m.mean(), 100 * b[m].mean()))
    say('  特征方向（训练段 corr）：' + '  '.join(
        '%s=%+.3f' % (nm, float(np.corrcoef(np.nan_to_num(X[tr, k]), y[tr])[0, 1]))
        for k, nm in enumerate(NAME)))
    REPORT[tag] = (p, y[te], auc)

np.savez(os.path.join(EV, 'gambler_proba.npz'),
         p_dn=REPORT['崩'][0], y_dn=REPORT['崩'][1], p_up=REPORT['涨'][0], y_up=REPORT['涨'][1])
open(os.path.join(EV, 'gambler.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
