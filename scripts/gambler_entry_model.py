# -*- coding: utf-8 -*-
"""v4-3：让「什么时候进场」也由模型决定（不再硬编码跌 5%）

用户批评：入场和退出都该由模型决定，不是我把 5% 拍进去。
本脚本把入场改成逐根决策，和退出完全对称：

  每个采样时刻问：现在开仓、持 T 根，会不会赚？
    状态 = 5根跌幅 / 15根跌幅 / 60根跌幅 / 240根跌幅 / 波动 / 量比
           / 区间位置 / 市场活跃度 / 连续下跌根数
    标签 = 现在开仓持 T 根的净收益 > 0（训练时用未来，标准监督学习）
  训练/测试按时间切 70/30，测试段绝不看未来
  测试：p > 阈值就开仓；对比「硬编码跌5%」

关键：模型自己决定「跌多少才值得买」——它能看到跌幅，但阈值由它学。

输出：evidence/entry_model.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('EM_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\entry_model.txt'
FEE = 0.0004
SLIP = float(os.environ.get('EM_SLIP', '0.0005'))
NSYM = int(os.environ.get('EM_NSYM', '60'))
COST = FEE + SLIP
HOLD = int(os.environ.get('EM_HOLD', '176'))
STEP = int(os.environ.get('EM_STEP', '20'))
NMAX = int(os.environ.get('EM_NMAX', '2000000'))
DROP_REF = 0.05

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('v4-3 · 入场也由模型决定（逐根决策，不硬编码跌5%）')
say(f'  {os.path.basename(DATA)}  持 {HOLD} 根  成本 {COST:.3%}  采样步长 {STEP}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

FEATS = 10

def feats_at(C, V, i, vabs):
    """i 时刻能看到的状态（只用 <= i）"""
    if i < 250:
        return None
    r5 = C[i] / C[i - 5] - 1.0
    r15 = C[i] / C[i - 15] - 1.0
    r60 = C[i] / C[i - 60] - 1.0
    r240 = C[i] / C[max(0, i - 240)] - 1.0
    seg = C[i - 240:i + 1]
    r1 = np.diff(np.log(np.maximum(seg, 1e-12)))
    sig = np.std(r1) * 100
    v20 = V[i - 20:i + 1].mean(); v240 = V[i - 240:i + 1].mean()
    volr = v20 / v240 if v240 > 0 else 1.0
    lo = C[i - 240:i].min(); hi = C[i - 240:i].max()
    pos = (C[i] - lo) / (hi - lo) if hi > lo else 0.5
    # 连续下跌根数
    dn = 0
    for k in range(1, 21):
        if C[i - k + 1] < C[i - k]:
            dn += 1
        else:
            break
    return [r5 * 100, r15 * 100, r60 * 100, r240 * 100, sig, volr, pos, vabs[i], dn / 20.0, C[i] / C[max(0, i - 1440)] - 1.0]

say('  构造入场样本 ...')
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
    vabs = (sv.rolling(1440).mean() / max(sv.mean(), 1e-12)).values
    for i in range(250, T - HOLD, STEP):
        f = feats_at(C, V, i, vabs)
        if f is None or not np.isfinite(f).all():
            continue
        r = C[i + HOLD] / C[i] - 1.0 - COST
        X_rows.append(f); y_rows.append(1 if r > 0 else 0); t_rows.append(ts[i])
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
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                     min_samples_leaf=100, random_state=11)
clf.fit(X[:cut_i], y[:cut_i])
auc_tr = roc_auc_score(y[:cut_i], clf.predict_proba(X[:cut_i])[:, 1])
auc_te = roc_auc_score(y[cut_i:], clf.predict_proba(X[cut_i:])[:, 1])
say(f'  AUC 训练 {auc_tr:.3f}  测试 {auc_te:.3f}')

# ---- 特征重要性（用 permutation 快速估） ----
NAMES = ['r5', 'r15', 'r60', 'r240', 'sig', 'volr', 'pos', 'vabs', 'dndays', 'r1440']
say('')
say('=' * 100)
say('【一】模型最看重什么（permutation 重要性，越大越重要）')
say('=' * 100)
base = auc_te
imps = []
rng = np.random.default_rng(11)
Xte = X[cut_i:][:200000]
yte = y[cut_i:][:200000]
p0 = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
for k in range(FEATS):
    Xp = Xte.copy()
    rng.shuffle(Xp[:, k])
    a = roc_auc_score(yte, clf.predict_proba(Xp)[:, 1])
    imps.append((NAMES[k], p0 - a))
for n, v in sorted(imps, key=lambda x: -x[1]):
    say(f'  {n:<8s} {v:>+8.4f}')

# ---- 测试段回放：模型选 vs 硬编码跌5% ----
say('')
say('  测试段回放 ...')
res_mod = {q: [] for q in (0.5, 0.6, 0.7, 0.8)}
res_fix = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    ts = g['open_time'].values
    T = len(C)
    if T < 3000:
        continue
    sv = pd.Series(V)
    vabs = (sv.rolling(1440).mean() / max(sv.mean(), 1e-12)).values
    i = 250
    while i < T - HOLD:
        if ts[i] >= cut_t:
            f = feats_at(C, V, i, vabs)
            if f is None or not np.isfinite(f).all():
                i += 1; continue
            p = clf.predict_proba(np.array([f], dtype=np.float32))[0, 1]
            r = C[i + HOLD] / C[i] - 1.0 - COST
            for q in res_mod:
                if p >= q:
                    res_mod[q].append(r)
            if C[i] / C[i - 5] - 1.0 <= -DROP_REF:
                res_fix.append(r)
        i += HOLD   # 开了仓就跳过整段（简化：每 HOLD 根最多一笔）

say('')
say('=' * 100)
say('【二】测试段对比')
say('=' * 100)
say(f'  {"策略":<20s} {"笔数":>9s} {"单笔均值":>9s} {"胜率":>7s} {"t值":>7s}')
a = np.array(res_fix)
if len(a) >= 30:
    mu = a.mean(); sd = a.std(ddof=1)
    say(f'  {"硬编码跌5%":<20s} {len(a):>9,} {mu*100:>+8.3f}% {(a>0).mean():>6.1%} {mu/(sd/np.sqrt(len(a))):>+7.2f}')
for q in (0.5, 0.6, 0.7, 0.8):
    a = np.array(res_mod[q])
    if len(a) < 30:
        say(f'  {"模型选 p>="+str(q):<20s} {len(a):>9,}  样本太少'); continue
    mu = a.mean(); sd = a.std(ddof=1)
    say(f'  {"模型选 p>="+str(q):<20s} {len(a):>9,} {mu*100:>+8.3f}% {(a>0).mean():>6.1%} {mu/(sd/np.sqrt(len(a))):>+7.2f}')

say('')
say('=' * 100)
say('【三】四段（模型选 p>=0.6）')
say('=' * 100)
a = np.array(res_mod[0.6])
if len(a) >= 40:
    seg = pd.qcut(np.arange(len(a)), 4, labels=False)
    for s in range(4):
        m = a[seg == s]
        say(f'  段{s+1}  n={len(m):>6,}  单笔 {m.mean()*100:>+7.3f}%  t={m.mean()/(m.std(ddof=1)/np.sqrt(len(m))):>+6.2f}')
    same = all(a[seg == s].mean() > 0 for s in range(4))
    say(f'  {"✅ 四段同正" if same else "❌ 有段非正"}')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 若"模型选"的单笔均值 >= "硬编码跌5%" 且笔数更多 ⇒ 入场确实可以由模型决定')
say('  · 若略差但量级接近 ⇒ 模型学到了接近的规则，可以继续加特征')
say('  · 若明显更差 ⇒ 入场这块还没学出来，需要更多牌')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
