# -*- coding: utf-8 -*-
"""把历史里"底买高卖"的成功片段全标出来，再回答一个问题：
   这些片段**入场那一刻**的状态，当时认得出吗？

这是"事后标注"能否当训练集用的唯一检验点：
  · 如果成功入场点的状态和随机时刻**长得一样** ⇒ 只是事后诸葛亮，学不到
  · 如果明显不一样 ⇒ 这个方向成立，可以拿去喂模型

标签定义（不掺任何未来信息进特征）：
  entry = t 时刻收盘价买入
  fwd_max = 未来 H 根内最高价 / entry - 1
  fwd_min = 未来 H 根内最低价 / entry - 1
  成功片段 ⇔ fwd_max >= 5% 且 fwd_min >= -2%
        （买完能涨 5%，且中途没先跌穿 2% —— "底买"的严格版）

特征只用 t 及之前的数据（rolling / shift，天然无前视）。
检验：成功点 vs 随机点，状态能不能被区分（AUC，时间前推）。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CASE_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\case_labels.txt'
BPD = int(os.environ.get('CASE_BPD', '288'))       # 5m: 288 根/天
H = int(os.environ.get('CASE_H', str(BPD)))        # 持有/观察窗口：1 天
NSYM = int(os.environ.get('CASE_NSYM', '60'))
TP = 0.05          # 涨 5%
SL = -0.02         # 中途不许跌破 -2%
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say(f'历史成功片段标注 + 可识别性检验')
say(f'  数据 {os.path.basename(DATA)}  H={H}根  TP={TP:.0%}  SL={SL:.0%}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA, nrows=None)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

FEATS = ['r5', 'r15', 'r60', 'r240', 'vol', 'rp', 'volr', 'vt', 'drift']
FEAT_CN = {'r5': '5根动量', 'r15': '15根动量', 'r60': '60根动量', 'r240': '240根动量',
           'vol': '波动率', 'rp': '区间位置', 'volr': '量比', 'vt': '笔数比', 'drift': '1根涨跌'}

rows = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    Hi = g['high'].values.astype(float) if 'high' in g else C
    Lo = g['low'].values.astype(float) if 'low' in g else C
    V = g['volume'].values.astype(float)
    T = len(C)
    if T < 2000:
        continue
    # ---- 特征（全部只用过去）----
    P = {}
    for k in (5, 15, 60, 240):
        m = np.full(T, np.nan)
        if T > k:
            m[k:] = C[k:] / C[:-k] - 1.0
        P[f'r{k}'] = m
    r1 = np.zeros(T); r1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    P['drift'] = r1
    P['vol'] = pd.Series(r1).rolling(60, min_periods=20).std().values
    hi = pd.Series(C).rolling(240, min_periods=60).max().values
    lo = pd.Series(C).rolling(240, min_periods=60).min().values
    P['rp'] = (C - lo) / np.maximum(hi - lo, 1e-12)
    q = pd.Series(V).rolling(60, min_periods=20).mean().values
    P['volr'] = np.clip(V / np.maximum(q, 1e-12), 0, 10)
    if 'trades' in g:
        Tr = g['trades'].values.astype(float)
        qt = pd.Series(Tr).rolling(60, min_periods=20).mean().values
        P['vt'] = np.clip(Tr / np.maximum(qt, 1e-12), 0, 10)
    else:
        P['vt'] = np.ones(T)

    # ---- 标签（只用未来，绝不进特征）----
    fwd_hi = np.full(T, np.nan); fwd_lo = np.full(T, np.nan)
    for i in range(0, T - H):
        seg_h = Hi[i + 1:i + 1 + H]
        seg_l = Lo[i + 1:i + 1 + H]
        if len(seg_h) == 0:
            continue
        fwd_hi[i] = seg_h.max() / C[i] - 1
        fwd_lo[i] = seg_l.min() / C[i] - 1
    win = (fwd_hi >= TP) & (fwd_lo >= SL)

    ok = np.isfinite(fwd_hi) & np.isfinite(P['r240']) & np.isfinite(P['vol'])
    for j in range(T):
        if not ok[j]:
            continue
        rows.append((sym, j, 1 if win[j] else 0,
                     fwd_hi[j], fwd_lo[j],
                     *[P[f][j] for f in FEATS]))

cols = ['symbol', 'i', 'y', 'fwd_hi', 'fwd_lo'] + FEATS
D = pd.DataFrame(rows, columns=cols)
del rows
say(f'  样本 {len(D):,}  成功片段 {int(D.y.sum()):,} ({D.y.mean()*100:.2f}%)  {time.time()-t0:.0f}s')

# ---- 成功片段的"成功程度"分布 ----
say('')
say('=' * 100)
say('【一】成功片段长什么样')
say('=' * 100)
W = D[D.y == 1]
say(f'  数量 {len(W):,}  每币中位 {int(W.groupby("symbol").size().median())} 段')
say(f'  平均最大涨幅 {W.fwd_hi.mean()*100:+.2f}%   平均最大回撤 {W.fwd_lo.mean()*100:+.2f}%')
say(f'  涨幅分位: p10 {W.fwd_hi.quantile(.1)*100:.1f}%  p50 {W.fwd_hi.quantile(.5)*100:.1f}%  p90 {W.fwd_hi.quantile(.9)*100:.1f}%')

# ---- 关键检验：状态能否区分成功/非成功 ----
say('')
say('=' * 100)
say('【二】入场时刻的状态，当时认得出吗（AUC，时间前推 5 折）')
say('=' * 100)
say('  AUC=0.5 完全认不出；0.55 微弱；0.60 有用；0.65+ 强')
say('')

D = D.sort_values(['symbol', 'i']).reset_index(drop=True)
# 时间前推：按 i（时间序号）切成 5 段，用前段训练、后段测试
D['fold'] = pd.qcut(D.groupby('symbol').cumcount(), 5, labels=False)

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    HAVE_SK = False

res = []
for f in FEATS:
    aucs = []
    for k in range(1, 5):
        tr = D[D.fold < k]; te = D[D.fold == k]
        if len(tr) < 5000 or len(te) < 1000:
            continue
        X = tr[[f]].values; ytr = tr.y.values
        mu, sd = np.nanmean(X), np.nanstd(X)
        X = np.nan_to_num((X - mu) / max(sd, 1e-12))
        Xt = np.nan_to_num((te[[f]].values - mu) / max(sd, 1e-12))
        coef = np.polyfit(X[:, 0], ytr, 1)[0]       # 简单线性打分，不训练分类器
        aucs.append(auc(te.y.values, Xt[:, 0] * coef))
    m = np.nanmean(aucs) if aucs else np.nan
    res.append((f, m, aucs))

if HAVE_SK:
    # 多特征逻辑回归（时间前推）
    aucs = []
    for k in range(1, 5):
        tr = D[D.fold < k]; te = D[D.fold == k]
        if len(tr) < 5000 or len(te) < 1000:
            continue
        X = tr[FEATS].values.astype(float); ytr = tr.y.values
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
        Xn = np.nan_to_num((X - mu) / sd)
        Xt = np.nan_to_num((te[FEATS].values.astype(float) - mu) / sd)
        clf = LogisticRegression(max_iter=500, C=0.5)
        clf.fit(Xn, ytr)
        ps = clf.predict_proba(Xt)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(Xt))
        aucs.append(auc(te.y.values, ps))
    SK = (np.nanmean(aucs) if aucs else np.nan, aucs)
else:
    SK = (np.nan, [])

for f, m, a in res:
    star = '✅' if m > 0.58 else ('⚠' if m > 0.53 else '❌')
    say(f'  {FEAT_CN[f]:10s} AUC={m:.3f}  {star}   分段: ' + ' '.join(f'{x:.3f}' for x in a))
say(f'\n  九特征联合 AUC={SK[0]:.3f}' + ('  ✅' if SK[0] > 0.58 else '  ❌') +
    '   分段: ' + ' '.join(f'{x:.3f}' for x in SK[1]))

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · AUC 明显 >0.5（且 5 折都同向）⇒ "底买"那一刻的状态是可识别的，标注可用')
say('  · AUC ≈0.5 ⇒ 这些片段事后看得见、当时认不出，喂给模型只会学到噪声')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
