# -*- coding: utf-8 -*-
"""v4-2：让「什么时候走」也由模型决定（抄底入场专用）

前提：v4 合成已通（跌5%买 + 凯利 + 急停 → 期末 4.34 倍、回撤 −10.6%、11/13 季度正）。
唯一还是死板的：持满 176 根就走。本脚本让它自己判断。

做法：
  入场 = 近 5 根跌 5% 买
  每个入场的每 STEP 根采样一个决策点：
    状态 = 浮盈 / 持有根数 / 波动 / 动量 / 段内位置 / 量比
    标签 = 从这里再持 K 根，是否比现在退更好（训练时可用未来，标准监督学习）
  训练/测试按时间切 70/30，测试段绝不看未来
  测试：逐根查分类器，p(继续) < 0.5 就走；对比固定持 176

输出：evidence/revert_exit.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('RE_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\revert_exit.txt'
FEE = 0.0004
SLIP = float(os.environ.get('RE_SLIP', '0.0005'))
NSYM = int(os.environ.get('RE_NSYM', '60'))
COST = FEE + SLIP
DROP = float(os.environ.get('RE_DROP', '0.05'))
MAX_H = int(os.environ.get('RE_MAXH', '576'))
KF = int(os.environ.get('RE_K', '50'))
STEP = int(os.environ.get('RE_STEP', '8'))
NMAX = int(os.environ.get('RE_NMAX', '1500000'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('v4-2 · 让模型自己决定什么时候走（抄底入场专用）')
say(f'  {os.path.basename(DATA)}  跌 {DROP:.0%} 买  成本 {COST:.3%}  K={KF} 步长={STEP}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def state_vec(C, V, e, j):
    ret = C[j] / C[e] - 1.0
    hold = j - e
    seg = C[max(0, j - 100):j + 1]
    r1 = np.diff(np.log(np.maximum(seg, 1e-12)))
    v20 = np.std(r1[-20:]) * 100 if len(r1) >= 20 else 0.0
    v100 = np.std(r1) * 100 if len(r1) >= 30 else 0.0
    m5 = (C[j] / C[max(0, j - 5)] - 1.0) * 100 if j >= 5 else 0.0
    m20 = (C[j] / C[max(0, j - 20)] - 1.0) * 100 if j >= 20 else 0.0
    lo = C[e:j + 1].min(); hi = C[e:j + 1].max()
    pos = (C[j] - lo) / (hi - lo) if hi > lo else 0.5
    v20v = V[max(0, j - 20):j + 1].mean(); v100v = V[max(0, j - 100):j + 1].mean()
    volr = v20v / v100v if v100v > 0 else 1.0
    dd = (C[j] - hi) / hi * 100 if hi > 0 else 0.0
    return [ret * 100.0, np.log1p(hold), v20, v100, m5, m20, pos, volr, dd]

say('  构造决策样本 ...')
X_rows, y_rows, t_rows = [], [], []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    i = 5
    while i < T - MAX_H - KF:
        if C[i] / C[i - 5] - 1.0 <= -DROP:
            e = i
            for j in range(e + 12, e + MAX_H + 1, STEP):
                if j + KF >= T:
                    break
                sv = state_vec(C, V, e, j)
                if not np.isfinite(sv).all():
                    continue
                lab = 1 if C[j + KF] > C[j] else 0
                X_rows.append(sv); y_rows.append(lab); t_rows.append(ts[j])
                if len(X_rows) >= NMAX:
                    break
            i += MAX_H      # 一个入场只算一次，跳过整段
        else:
            i += 1
    if len(X_rows) >= NMAX:
        say(f'  达到样本上限 {NMAX:,}')
        break

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows)
tt = np.array(t_rows)
say(f'  样本 {len(X):,}  正 {y.mean():.1%}  {time.time()-t0:.0f}s')
if len(X) < 50000:
    say('  ❌ 样本不足'); sys.exit(1)

order = np.argsort(tt)
X, y, tt = X[order], y[order], tt[order]
cut_i = int(len(X) * 0.7)
cut_t = tt[cut_i]
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                     min_samples_leaf=80, random_state=11)
clf.fit(X[:cut_i], y[:cut_i])
auc_tr = roc_auc_score(y[:cut_i], clf.predict_proba(X[:cut_i])[:, 1])
auc_te = roc_auc_score(y[cut_i:], clf.predict_proba(X[cut_i:])[:, 1])
say(f'  AUC 训练 {auc_tr:.3f}  测试 {auc_te:.3f}  （切片 {pd.to_datetime(cut_t, unit="ms")}）')

say('')
say('  测试段回放 ...')
res_fix, res_mod, res_max = [], [], []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    i = 5
    while i < T - MAX_H:
        if ts[i] >= cut_t and C[i] / C[i - 5] - 1.0 <= -DROP:
            e = i; entry = C[e]
            if entry <= 0:
                i += 1; continue
            jmax = min(e + MAX_H, T - 1)
            exit_j = jmax
            for j in range(e + 12, jmax + 1, STEP):
                sv = state_vec(C, V, e, j)
                if not np.isfinite(sv).all():
                    continue
                if clf.predict_proba(np.array([sv], dtype=np.float32))[0, 1] < 0.5:
                    exit_j = j; break
            res_mod.append(C[exit_j] / entry - 1.0 - COST)
            j176 = min(e + 176, T - 1)
            res_fix.append(C[j176] / entry - 1.0 - COST)
            res_max.append(C[jmax] / entry - 1.0 - COST)
            i += MAX_H
        else:
            i += 1

say('')
say('=' * 100)
say('【一】样本外对比')
say('=' * 100)
say(f'  {"策略":<18s} {"笔数":>7s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"t值":>7s}')
for name, v in [('固定持176', res_fix), ('模型逐根决策', res_mod), ('持到MAX_H', res_max)]:
    a = np.array(v)
    if len(a) < 30:
        say(f'  {name:<18s} {len(a):>7}  样本不足'); continue
    mu = a.mean(); sd = a.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(a))) if sd > 0 else 0
    say(f'  {name:<18s} {len(a):>7,} {mu*100:>+8.3f}% {np.median(a)*100:>+8.3f}% {(a>0).mean():>6.1%} {tv:>+7.2f}')

say('')
say('=' * 100)
say('【二】四段（模型逐根决策）')
say('=' * 100)
a = np.array(res_mod)
if len(a) >= 40:
    seg = pd.qcut(np.arange(len(a)), 4, labels=False)
    for s in range(4):
        m = a[seg == s]
        say(f'  段{s+1}  n={len(m):>5,}  单笔 {m.mean()*100:>+7.3f}%  t={m.mean()/(m.std(ddof=1)/np.sqrt(len(m))):>+6.2f}')
    same = all(a[seg == s].mean() > 0 for s in range(4))
    say(f'  {"✅ 四段同正" if same else "❌ 有段非正"}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 若"模型逐根决策"优于"固定持176" ⇒ 退出可以学')
say('  · 若 优于 且四段同正 ⇒ 这就是要长出来的退出结构')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
