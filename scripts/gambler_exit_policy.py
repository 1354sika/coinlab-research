# -*- coding: utf-8 -*-
"""让模型自己选择如何退出：逐根决策（严格无前视）

核心问题：每一段成功交易的最优退出点都不同 -> 固定规则必然失败
做法：
  1. 每个入场的每根 bar 都是一个决策样本
  2. 状态 = 此刻看得到的（浮盈 / 已持有根数 / 波动 / 动量 / 段内位置 / 量比）
  3. 标签 = "继续持 K 根" 是否比 "现在退" 更好（训练时可用未来 K 根 = 标准监督学习）
  4. 训练分类器，测试时逐根决策：说"继续"就持，说"退"就走
  5. 训练/测试按时间切（前 70% / 后 30%），测试段绝不看未来

输出：evidence/exit_policy.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('EP_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\exit_policy.txt'
Z = float(os.environ.get('EP_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('EP_SLIP', '0.0005'))
NSYM = int(os.environ.get('EP_NSYM', '60'))
COST = FEE + SLIP
MAX_H = int(os.environ.get('EP_MAXH', '576'))
KF = int(os.environ.get('EP_K', '50'))      # 反事实窗口
STEP = int(os.environ.get('EP_STEP', '8'))  # 采样步长
NMAX = int(os.environ.get('EP_NMAX', '2000000'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('退出策略 · 逐根决策（模型自己选何时走）')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}  K={KF} 步长={STEP} 币={NSYM}')
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

FEATS = ['ret', 'hold_log', 'vol20', 'vol100', 'mom5', 'mom20', 'pos_in_run', 'volr', 'dd_from_hi']

def state_vec(C, V, e, j, run_lo, run_hi):
    """j 时刻能看到的状态（只用 <= j 的数据）"""
    ret = C[j] / C[e] - 1.0
    hold = j - e
    r1 = np.diff(np.log(np.maximum(C[max(0, j-100):j+1], 1e-12)))
    v20 = np.std(r1[-20:]) if len(r1) >= 20 else np.nan
    v100 = np.std(r1) if len(r1) >= 30 else np.nan
    m5 = C[j] / C[max(0, j-5)] - 1.0 if j >= 5 else 0.0
    m20 = C[j] / C[max(0, j-20)] - 1.0 if j >= 20 else 0.0
    lo = min(C[e:j+1]) if j > e else C[e]
    hi = max(C[e:j+1]) if j > e else C[e]
    pos = (C[j] - lo) / (hi - lo) if hi > lo else 0.5
    v20v = np.mean(V[max(0, j-20):j+1])
    v100v = np.mean(V[max(0, j-100):j+1])
    volr = v20v / v100v if v100v > 0 else 1.0
    dd = (C[j] - hi) / hi if hi > 0 else 0.0
    return [ret * 100.0, np.log1p(hold), v20 * 100, v100 * 100, m5 * 100, m20 * 100, pos, volr, dd * 100]

X_rows, y_rows, t_rows = [], [], []
say('  构造决策样本 ...')
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    for (a, e, b) in zigzag_entry(C, Z):
        jmax = min(e + MAX_H, T - 1 - KF)
        if jmax <= e + 12:
            continue
        for j in range(e + 12, jmax + 1, STEP):
            sv = state_vec(C, V, e, j, a, b)
            if not np.isfinite(sv).all():
                continue
            future_ret = C[j + KF] / C[j] - 1.0
            lab = 1 if future_ret > 0 else 0
            X_rows.append(sv)
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
say(f'  样本 {len(X):,}  标签均值 {y.mean():.1%}  {time.time()-t0:.0f}s')
if len(X) < 50000:
    say('  ❌ 样本不足')
    sys.exit(1)

order = np.argsort(tt)
X, y = X[order], y[order]
cut = int(len(X) * 0.7)
Xtr, ytr = X[:cut], y[:cut]
Xte, yte = X[cut:], y[cut:]
say(f'  训练 {len(Xtr):,}（正 {ytr.mean():.1%}）  测试 {len(Xte):,}（正 {yte.mean():.1%}）')

clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                     min_samples_leaf=80, random_state=11)
clf.fit(Xtr, ytr)
auc_tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
auc_te = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
say('')
say('=' * 100)
say('【一】退出决策能不能学')
say('=' * 100)
say(f'  AUC 训练 {auc_tr:.3f}   测试 {auc_te:.3f}')
say('  （这次不是预测方向，是判断"继续持 vs 现在退"，0.55 以上就有用）')

# ---- 用模型逐根决策，算实际收益 ----
say('')
say('  测试集上回放：模型逐根决定何时退 ...')
cut_t = np.sort(tt)[int(len(tt) * 0.7)]
res_model, res_fix176, res_hold = [], [], []
t_res = []

for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    for (a, e, b) in zigzag_entry(C, Z):
        if ts[e] < cut_t:
            continue
        entry = C[e]
        if entry <= 0:
            continue
        jmax = min(e + MAX_H, T - 1)
        if jmax <= e + 12:
            continue
        # 模型逐根决策
        exit_j = jmax
        for j in range(e + 12, jmax + 1, STEP):
            sv = np.array([state_vec(C, V, e, j, a, b)], dtype=np.float32)
            if not np.isfinite(sv).all():
                continue
            p = clf.predict_proba(sv)[0, 1]
            if p < 0.5:
                exit_j = j
                break
        r_model = C[exit_j] / entry - 1.0 - COST
        j176 = min(e + 176, T - 1)
        r176 = C[j176] / entry - 1.0 - COST
        rhold = C[jmax] / entry - 1.0 - COST
        res_model.append(r_model)
        res_fix176.append(r176)
        res_hold.append(rhold)
        t_res.append(ts[e])

say(f'  样本外入场 {len(res_model):,}')
say('')
say('=' * 100)
say('【二】模型退出 vs 固定规则（样本外）')
say('=' * 100)
cm = np.array(res_model); c1 = np.array(res_fix176); ch = np.array(res_hold)
for name, v in [('模型逐根决策', cm), ('固定持176', c1), ('持到MAX_H', ch)]:
    mu = v.mean(); sd = v.std(ddof=1)
    tv = mu / (sd / np.sqrt(len(v))) if sd > 0 else 0
    say(f'  {name:<14s} n={len(v):>7,}  单笔 {mu*100:>+7.3f}%  中位 {np.median(v)*100:>+7.3f}%  胜率 {(v>0).mean():>6.1%}  t={tv:>+6.2f}')

say('')
say('=' * 100)
say('【三】四段稳定性（模型退出）')
say('=' * 100)
seg_arr = pd.qcut(np.arange(len(cm)), 4, labels=False)
for sg in range(4):
    m = (seg_arr == sg)
    say(f'  段{sg+1}  n={m.sum():>6,}  单笔 {cm[m].mean()*100:>+7.3f}%  t={cm[m].mean()/(cm[m].std(ddof=1)/np.sqrt(m.sum())):>+6.2f}')
same = all(cm[seg_arr == sg].mean() > 0 for sg in range(4))
say(f'  {"✅ 四段同正" if same else "❌ 有段非正"}')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 若"模型逐根决策"优于"固定持176"且四段同正 ⇒ 退出可以学，这就是要长出来的东西')
say('  · 若仍劣于固定规则 ⇒ 这些状态维度还不够判退出，需加入更多信息')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
