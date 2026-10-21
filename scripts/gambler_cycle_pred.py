# -*- coding: utf-8 -*-
"""gambler_cycle_pred · 在 +20% 时刻，用当时可观测的特征预测「后面还剩多少」"""
import numpy as np
import pandas as pd

SRC = r'D:\中信证券\coinlab\evidence\cycle\cycle20_rows_R30.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle\cycle_pred_summary.txt'

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

df = pd.read_csv(SRC)
df['y'] = (df.rest_after20 >= 0.30).astype(int)
df = df.sort_values('t_lo').reset_index(drop=True)
log('=' * 100)
log('gambler_cycle_pred · +20% 时刻的可观测特征 → 预测「剩余空间 >= 30%」')
log('=' * 100)
log(f'样本 {len(df):,}  正例率 {df.y.mean():.1%}  (标签=峰值比 +20% 点再高 30% 以上)')

cut = df.t_lo.quantile(0.70)
tr = df[df.t_lo <= cut]
te = df[df.t_lo > cut]
log(f'训练 {len(tr):,}（正例 {tr.y.mean():.1%}）  测试 {len(te):,}（正例 {te.y.mean():.1%}）')

feats = ['volr', 'reach_h']
X_tr = tr[feats].fillna(tr[feats].median())
X_te = te[feats].fillna(tr[feats].median())

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

for name, model in [
    ('逻辑回归', LogisticRegression(max_iter=1000)),
    ('GBDT', GradientBoostingClassifier(n_estimators=120, max_depth=2, random_state=7)),
]:
    model.fit(X_tr, tr.y)
    p = model.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(te.y, p)
    except Exception:
        auc = float('nan')
    log(f'\n【{name}】测试段 AUC = {auc:.3f}')
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        sub = te[m]
        log(f'  P∈[{lo:.1f},{hi:.1f})  n={m.sum():>4,}  实际命中率 {sub.y.mean():>5.1%}  '
            f'剩余空间中位 {sub.rest_after20.median():>6.1%}  均值 {sub.rest_after20.mean():>6.1%}')

# 单特征对照
log('\n【单特征对照（测试段）】')
for f in feats:
    s = te[[f, 'y']].dropna()
    if len(s) < 50:
        continue
    try:
        a = roc_auc_score(s.y, s[f])
        log(f'  {f:10s} AUC={a:.3f}')
    except Exception:
        pass

# 量比分档（最朴素的可观测规则）
log('\n【量比分档（测试段）】')
te2 = te.copy()
te2['q'] = pd.qcut(te2.volr, 4, duplicates='drop')
for q, g in te2.groupby('q', observed=True):
    log(f'  {str(q):24s} n={len(g):>4,}  命中率 {g.y.mean():>5.1%}  剩余中位 {g.rest_after20.median():>6.1%}')

log('\n【判读】')
log('  · AUC 若 < 0.60 ⇒ 在 +20% 时刻靠这些特征预测剩余空间，几乎无效')
log('  · 量比单独分档若单调 ⇒ 它是唯一可用的一维门控')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
print('saved', OUT)
