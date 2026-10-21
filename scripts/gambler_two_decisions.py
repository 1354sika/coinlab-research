# -*- coding: utf-8 -*-
"""两个决策的完整模型：下注 / 不下注 + 走 / 继续持

用户定义：赌徒只会两件事 —— 进场下注、退出。
本脚本把这两件事合成一个时间流模型，严格无前视：

  决策1（下注）：
    候选入场 = zigzag 确认底（价格从近期低点反弹 Z%）
    状态 = volr（近60/近1440 均量比）等
    规则 = volr >= 训练段分位数阈值 才允许下注
           （阈值只在训练段确定，测试段只查表）

  决策2（退出）：
    训练段：对每个入场每 STEP 根采样，标签 = 继续持 K 根是否比现在退更好
    测试段：逐根查分类器，p(继续) < 0.5 就走；最多持 MAX_H

  对比基线：
    B1 全部入场 + 固定持 176
    B2 全部入场 + 持到 MAX_H
    M1 只下注 volr 高档 + 固定持 176
    M2 只下注 volr 高档 + 模型退出   ← 目标
    M3 全部入场 + 模型退出

输出：evidence/two_decisions.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('TD_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\two_decisions.txt'
Z = float(os.environ.get('TD_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('TD_SLIP', '0.0005'))
NSYM = int(os.environ.get('TD_NSYM', '60'))
COST = FEE + SLIP
MAX_H = int(os.environ.get('TD_MAXH', '576'))
KF = int(os.environ.get('TD_K', '50'))
STEP = int(os.environ.get('TD_STEP', '8'))
NMAX = int(os.environ.get('TD_NMAX', '1500000'))
GATE_Q = float(os.environ.get('TD_Q', '0.80'))   # 只在训练段 volr 前 20% 下注

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('两个决策的完整模型 · 下注/不下注 + 走/继续持')
say(f'  {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}  下注门槛=训练段 volr 前{(1-GATE_Q)*100:.0f}%')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def zigzag_entry(C, z):
    T = len(C)
    segs = []
    trend = 0
    hi_i = lo_i = 0
    hi_v = lo_v = C[0]
    entry_i = None
    for i in range(1, T):
        c = C[i]
        if trend >= 0:
            if c > hi_v:
                hi_v, hi_i = c, i
            if c <= hi_v * (1 - z):
                if trend == 1 and entry_i is not None:
                    segs.append((lo_i, entry_i, hi_i))
                trend = -1
                lo_v, lo_i = c, i
                entry_i = None
        if trend <= 0:
            if c < lo_v:
                lo_v, lo_i = c, i
            if c >= lo_v * (1 + z):
                trend = 1
                hi_v, hi_i = c, i
                entry_i = i
    return segs

def state_vec(C, V, e, j):
    """j 时刻能看到的状态（只用 <= j 的数据）"""
    ret = C[j] / C[e] - 1.0
    hold = j - e
    seg = C[max(0, j - 100):j + 1]
    r1 = np.diff(np.log(np.maximum(seg, 1e-12)))
    v20 = np.std(r1[-20:]) * 100 if len(r1) >= 20 else 0.0
    v100 = np.std(r1) * 100 if len(r1) >= 30 else 0.0
    m5 = (C[j] / C[max(0, j - 5)] - 1.0) * 100 if j >= 5 else 0.0
    m20 = (C[j] / C[max(0, j - 20)] - 1.0) * 100 if j >= 20 else 0.0
    lo = C[e:j + 1].min()
    hi = C[e:j + 1].max()
    pos = (C[j] - lo) / (hi - lo) if hi > lo else 0.5
    v20v = V[max(0, j - 20):j + 1].mean()
    v100v = V[max(0, j - 100):j + 1].mean()
    volr = v20v / v100v if v100v > 0 else 1.0
    dd = (C[j] - hi) / hi * 100 if hi > 0 else 0.0
    return [ret * 100.0, np.log1p(hold), v20, v100, m5, m20, pos, volr, dd]

# ---------- 收集样本（含时间戳，稍后按时间切） ----------
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
    sv = pd.Series(V)
    volr = (sv.rolling(60).mean() / np.maximum(sv.rolling(1440).mean(), 1e-12)).values
    for (a, e, b) in zigzag_entry(C, Z):
        if not np.isfinite(volr[e]):
            continue
        jmax = min(e + MAX_H, T - 1 - KF)
        if jmax <= e + 12:
            continue
        for j in range(e + 12, jmax + 1, STEP):
            sv_ = state_vec(C, V, e, j)
            if not np.isfinite(sv_).all():
                continue
            lab = 1 if C[j + KF] > C[j] else 0
            X_rows.append(sv_ + [volr[e]])
            y_rows.append(lab)
            t_rows.append(ts[j])
            if len(X_rows) >= NMAX:
                break
        if len(X_rows) >= NMAX:
            break
    if len(X_rows) >= NMAX:
        say(f'  达到样本上限 {NMAX:,}')
        break

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows)
tt = np.array(t_rows)
say(f'  样本 {len(X):,}  正 {y.mean():.1%}  {time.time()-t0:.0f}s')

order = np.argsort(tt)
X, y, tt = X[order], y[order], tt[order]
cut_i = int(len(X) * 0.7)
cut_t = tt[cut_i]
Xtr, ytr = X[:cut_i], y[:cut_i]
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                     min_samples_leaf=80, random_state=11)
clf.fit(Xtr, ytr)
say(f'  退出分类器训练完（切片时刻 {pd.to_datetime(cut_t, unit="ms")}）  {time.time()-t0:.0f}s')

# ---------- 下注门槛：只用训练段的 volr 分布 ----------
volr_tr = Xtr[:, -1]
gate = np.quantile(volr_tr, GATE_Q)
say(f'  下注门槛 volr >= {gate:.3f}（训练段前 {(1-GATE_Q)*100:.0f}%）')

# ---------- 时间流回放（测试段） ----------
say('  测试段回放 ...')
res = {k: [] for k in ['B1', 'B2', 'M1', 'M2', 'M3']}
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    sv = pd.Series(V)
    volr = (sv.rolling(60).mean() / np.maximum(sv.rolling(1440).mean(), 1e-12)).values
    for (a, e, b) in zigzag_entry(C, Z):
        if ts[e] < cut_t or not np.isfinite(volr[e]):
            continue
        entry = C[e]
        if entry <= 0:
            continue
        jmax = min(e + MAX_H, T - 1)
        if jmax <= e + 12:
            continue
        j176 = min(e + 176, T - 1)
        r176 = C[j176] / entry - 1.0 - COST
        rmax = C[jmax] / entry - 1.0 - COST
        # 模型退出
        exit_j = jmax
        for j in range(e + 12, jmax + 1, STEP):
            sv_ = state_vec(C, V, e, j)
            if not np.isfinite(sv_).all():
                continue
            p = clf.predict_proba(np.array([sv_ + [volr[e]]], dtype=np.float32))[0, 1]
            if p < 0.5:
                exit_j = j
                break
        rmod = C[exit_j] / entry - 1.0 - COST
        res['B1'].append(r176)
        res['B2'].append(rmax)
        res['M3'].append(rmod)
        if volr[e] >= gate:
            res['M1'].append(r176)
            res['M2'].append(rmod)

say('')
say('=' * 100)
say('【一】样本外结果（测试段）')
say('=' * 100)
names = {'B1': '全部入场 + 固定持176', 'B2': '全部入场 + 持到MAX_H',
         'M3': '全部入场 + 模型退出', 'M1': '只下注volr高档 + 固定持176',
         'M2': '只下注volr高档 + 模型退出'}
say(f'  {"策略":<28s} {"笔数":>8s} {"单笔均值":>9s} {"中位":>9s} {"胜率":>7s} {"t值":>7s}')
for k in ['B1', 'B2', 'M3', 'M1', 'M2']:
    v = np.array(res[k])
    if len(v) < 50:
        say(f'  {names[k]:<28s} {len(v):>8,}  样本不足')
        continue
    mu = v.mean(); sd = v.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(v))) if sd > 0 else 0
    say(f'  {names[k]:<28s} {len(v):>8,} {mu*100:>+8.3f}% {np.median(v)*100:>+8.3f}% {(v>0).mean():>6.1%} {tv:>+7.2f}')

say('')
say('=' * 100)
say('【二】四段稳定性')
say('=' * 100)
for k in ['B1', 'M1', 'M2']:
    v = np.array(res[k])
    if len(v) < 200:
        continue
    seg = pd.qcut(np.arange(len(v)), 4, labels=False)
    row = [f'{v[seg == s].mean()*100:>+7.3f}%' for s in range(4)]
    same = all(v[seg == s].mean() > 0 for s in range(4))
    say(f'  {names[k]:<28s} 段1 {row[0]}  段2 {row[1]}  段3 {row[2]}  段4 {row[3]}   {"✅四段同正" if same else "❌有段非正"}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 目标：M2（只下注活跃期 + 模型退出）单笔 > 0 且四段同正')
say('  · 若 M2 > M1 且 M2 > M3 ⇒ 两个决策合起来确实比单个好')
say('  · 若 M2 仍 <= 0 ⇒ 现有状态维度不足以支撑"只学两件事"，需要新牌')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
