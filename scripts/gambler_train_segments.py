# -*- coding: utf-8 -*-
"""用 zigzag 段造训练集：正段/负段的入场状态 -> 训练分类器

三个问题：
  ① 入场那一刻的状态，能不能分辨"这段会赚"还是"这段会亏"？
  ② 如果能，正段占比能从 42% 抬到多少？
  ③ 抬上去之后，期望还剩多少（扣成本）？

严格无前视：
  · 特征只用 i_entry 及之前
  · 训练/测试按时间切（前 70% / 后 30%）
  · 不引用任何未来信息

输出：evidence/train_segments.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('TS_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\train_segments.txt'
Z = float(os.environ.get('TS_Z', '0.03'))
FEE = 0.0004
SLIP = float(os.environ.get('TS_SLIP', '0.0005'))
NSYM = int(os.environ.get('TS_NSYM', '60'))
COST = FEE + SLIP

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('段训练集 · 正段 vs 负段的入场状态（严格无前视 + 时间切分）')
say(f'  数据 {os.path.basename(DATA)}  Z={Z:.1%}  成本={COST:.3%}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

FEATS = ['r5', 'r15', 'r60', 'r240', 'r1440', 'vol240', 'volr', 'pos240', 'upfrac60', 'atr240']

def build_feats(C, V):
    """每根 bar 的特征，全部只用 t 及之前的数据"""
    T = len(C)
    F = np.full((T, len(FEATS)), np.nan, dtype=np.float32)
    logc = np.log(np.maximum(C, 1e-12))
    for k, w in enumerate([5, 15, 60, 240, 1440]):
        r = np.full(T, np.nan, dtype=np.float32)
        r[w:] = (logc[w:] - logc[:-w]) * 100.0
        F[:, k] = r
    r5 = F[:, 0].copy()
    vol = np.full(T, np.nan, dtype=np.float32)
    s5 = pd.Series(r5)
    vol = s5.rolling(240).std().values.astype(np.float32)
    F[:, 5] = vol
    sv = pd.Series(V)
    v60 = sv.rolling(60).mean().values
    v1440 = sv.rolling(1440).mean().values
    F[:, 6] = (v60 / np.maximum(v1440, 1e-12)).astype(np.float32)
    sc = pd.Series(C)
    mn = sc.rolling(240).min().values
    mx = sc.rolling(240).max().values
    F[:, 7] = ((C - mn) / np.maximum(mx - mn, 1e-12)).astype(np.float32)
    up = (np.diff(C, prepend=C[0]) > 0).astype(np.float32)
    F[:, 8] = pd.Series(up).rolling(60).mean().values.astype(np.float32)
    arb = np.abs(np.diff(logc, prepend=logc[0])) * 100.0
    F[:, 9] = pd.Series(arb).rolling(240).mean().values.astype(np.float32)
    return F

def zigzag_entry(C, z):
    """返回 [(i_lo, i_entry, i_hi)]，i_entry = 确认底那一刻"""
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

X_rows, y_rows, net_rows, t_rows = [], [], [], []
say('  切段 + 提特征 ...')

for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    V = g['volume'].values.astype(float)
    if len(C) < 3000:
        continue
    F = build_feats(C, V)
    ts = g['open_time'].values
    for (a, e, b) in zigzag_entry(C, Z):
        entry = C[e]
        exit_ = C[b] * (1 - Z)
        net = exit_ / entry - 1.0 - COST
        feat = F[e]
        if not np.isfinite(feat).all():
            continue
        X_rows.append(feat)
        y_rows.append(1 if net > 0 else 0)
        net_rows.append(net)
        t_rows.append(ts[e])

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows)
net = np.array(net_rows)
tt = np.array(t_rows)

say(f'  样本 {len(X):,}  正段占比 {y.mean():.1%}  全体均值 {net.mean()*100:+.3f}%  {time.time()-t0:.0f}s')
if len(X) < 5000:
    say('  ❌ 样本不足，停止')
    sys.exit(1)

order = np.argsort(tt)
X, y, net = X[order], y[order], net[order]
cut = int(len(X) * 0.7)
Xtr, ytr, ntr = X[:cut], y[:cut], net[:cut]
Xte, yte, net_te = X[cut:], y[cut:], net[cut:]
say(f'  训练 {len(Xtr):,}（正段 {ytr.mean():.1%}）  测试 {len(Xte):,}（正段 {yte.mean():.1%}）')

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                     min_samples_leaf=50, random_state=11)
clf.fit(Xtr, ytr)
auc_tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
p_te = clf.predict_proba(Xte)[:, 1]
auc_te = roc_auc_score(yte, p_te)

say('')
say('=' * 100)
say('【一】能不能分辨')
say('=' * 100)
say(f'  AUC 训练 {auc_tr:.3f}   测试 {auc_te:.3f}')
say(f'  （0.5 = 完全认不出；0.55 就足够跟 2.5:1 盈亏比配合赚钱）')

say('')
say('=' * 100)
say('【二】按预测概率取前几档，正段占比与期望')
say('=' * 100)
say(f'  {"档位":>10s} {"笔数":>9s} {"正段占比":>9s} {"单笔均值":>9s} {"累计":>10s}')
for q in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1):
    thr = np.quantile(p_te, 1 - q)
    m = p_te >= thr
    if m.sum() < 100:
        continue
    say(f'  top {q*100:>3.0f}%  {m.sum():>9,} {yte[m].mean():>8.1%} {net_te[m].mean()*100:>+8.3f}% {net_te[m].sum()*100:>+9.1f}%')

say('')
say('=' * 100)
say('【三】四段稳定性（top 40% 的预测）')
say('=' * 100)
thr40 = np.quantile(p_te, 0.6)
m40 = p_te >= thr40
seg_arr = pd.qcut(np.arange(len(p_te)), 4, labels=False)
for sg in range(4):
    mm = m40 & (seg_arr == sg)
    if mm.sum() > 0:
        say(f'  段{sg+1}  n={mm.sum():>7,}  正段占比 {yte[mm].mean():>6.1%}  均值 {net_te[mm].mean()*100:>+7.3f}%')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 测试 AUC > 0.55 且 top 40% 正段占比 > 基线 ⇒ 入场状态可识别，训练集可用')
say('  · 否则 ⇒ 这些段的入场时刻状态长得一样，只能事后看')
say('  · 注意：单笔均值是"段本身"的净收益（已扣确认延迟与成本），不是模型收益')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
