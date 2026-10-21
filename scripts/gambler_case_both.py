# -*- coding: utf-8 -*-
"""正例 vs 反例：把历史上所有"底→顶"和"顶→底"的片段都标出来，
   看模型能不能从**入场那一刻的状态**把两类分开。

上一版只标了正例（底→顶），对"全部时刻"做二分类，AUC≈0.50 ⇒ 认不出。
本版改成**正例 vs 反例对称标注**——这才是用户说的"低买高卖 + 高买低卖都要"：
  正例 win  : 未来 H 根内先涨到 +5%，且中途没跌破 -2%
  反例 lose : 未来 H 根内先跌到 -5%，且中途没涨破 +2%
  两类其余样本丢弃（只比较"该买"和"该卖"的极端时刻）

检验两件事：
  ① 每类各自长什么样（特征均值差异）—— 看方向
  ② 正例 vs 反例 能否分开（AUC，时间前推 5 折）—— 看可分性

AUC 高且有方向 ⇒ 状态确实分方向，标注可用
AUC≈0.5 ⇒ 这两类时刻的"历史"长得一样，事后标的标签喂不进去
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CASE_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\case_both.txt'
BPD = int(os.environ.get('CASE_BPD', '288'))
H = int(os.environ.get('CASE_H', str(BPD)))
NSYM = int(os.environ.get('CASE_NSYM', '20'))
TP = 0.05
SL = 0.02
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('正例 vs 反例 · 底买高卖 与 高买低卖 对称标注')
say(f'  数据 {os.path.basename(DATA)}  H={H}根  ±{TP:.0%}  不许先反向 {SL:.0%}  币={NSYM}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

FEATS = ['r5', 'r15', 'r60', 'r240', 'vol', 'rp', 'volr', 'vt']
CN = {'r5': '5根动量', 'r15': '15根动量', 'r60': '60根动量', 'r240': '240根动量',
      'vol': '波动率', 'rp': '区间位置', 'volr': '量比', 'vt': '笔数比'}

out = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    Hi = g['high'].values.astype(float)
    Lo = g['low'].values.astype(float)
    V = g['volume'].values.astype(float)
    T = len(C)
    if T < 2000:
        continue
    cs = pd.Series(C)
    # 未来 H 根内的最大/最小（向量化；窗口含 t+1..t+H）
    fhi = pd.Series(Hi).rolling(H).max().shift(-H) / cs - 1.0
    flo = pd.Series(Lo).rolling(H).min().shift(-H) / cs - 1.0
    # 特征（只用过去）
    r = {}
    for k in (5, 15, 60, 240):
        m = np.full(T, np.nan)
        if T > k:
            m[k:] = C[k:] / C[:-k] - 1.0
        r[f'r{k}'] = m
    rr = np.zeros(T); rr[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    r['vol'] = pd.Series(rr).rolling(60, min_periods=20).std().values
    hh = cs.rolling(240, min_periods=60).max().values
    ll = cs.rolling(240, min_periods=60).min().values
    r['rp'] = (C - ll) / np.maximum(hh - ll, 1e-12)
    q = pd.Series(V).rolling(60, min_periods=20).mean().values
    r['volr'] = np.clip(V / np.maximum(q, 1e-12), 0, 10)
    if 'trades' in g.columns:
        Tr = g['trades'].values.astype(float)
        qt = pd.Series(Tr).rolling(60, min_periods=20).mean().values
        r['vt'] = np.clip(Tr / np.maximum(qt, 1e-12), 0, 10)
    else:
        r['vt'] = np.ones(T)

    y = np.full(T, -1)                      # -1 = 丢弃
    y[(fhi >= TP) & (flo >= -SL)] = 1       # 正例：底→顶
    y[(flo <= -TP) & (fhi <= SL)] = 0       # 反例：顶→底
    ok = (y >= 0) & np.isfinite(r['r240']) & np.isfinite(r['vol'])
    idx = np.where(ok)[0]
    if len(idx) == 0:
        continue
    block = np.column_stack([np.full(len(idx), len(out) + j) for j in (0,)])  # 占位
    for j in idx:
        out.append((sym, j, int(y[j]), *[r[f][j] for f in FEATS]))

D = pd.DataFrame(out, columns=['symbol', 'i', 'y'] + FEATS)
del out
say(f'  样本 {len(D):,}  正例 {int((D.y==1).sum()):,}  反例 {int((D.y==0).sum()):,}  {time.time()-t0:.0f}s')

say('')
say('=' * 100
     if False else '=' * 100)
say('【一】两类时刻各自长什么样（特征均值）')
say('=' * 100)
say(f'  {"特征":10s} {"正例(底→顶)":>14s} {"反例(顶→底)":>14s} {"差":>10s}')
for f in FEATS:
    a = D.loc[D.y == 1, f].mean()
    b = D.loc[D.y == 0, f].mean()
    say(f'  {CN[f]:10s} {a:>14.4f} {b:>14.4f} {a-b:>+10.4f}')

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

D = D.sort_values(['symbol', 'i']).reset_index(drop=True)
D['fold'] = pd.qcut(D.groupby('symbol').cumcount(), 5, labels=False)

say('')
say('=' * 100)
say('【二】正例 vs 反例 能否分开（AUC，时间前推）')
say('=' * 100)
say('  AUC>0.6 说明"该买"和"该卖"那一刻的历史状态明显不同')
say('')
res = []
for f in FEATS:
    aucs = []
    for k in range(1, 5):
        tr = D[D.fold < k]; te = D[D.fold == k]
        if len(tr) < 3000 or len(te) < 500:
            continue
        coef = np.polyfit(tr[f].fillna(0).values, tr.y.values, 1)[0]
        aucs.append(auc(te.y.values, te[f].fillna(0).values * coef))
    m = np.nanmean(aucs) if aucs else np.nan
    res.append((f, m, aucs))

from sklearn.linear_model import LogisticRegression
Xall = D[FEATS].values.astype(float)
Xall = np.nan_to_num(Xall)
aucs = []
for k in range(1, 5):
    tr = D.fold < k; te = D.fold == k
    if tr.sum() < 3000 or te.sum() < 500:
        continue
    mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-12
    clf = LogisticRegression(max_iter=800, C=0.3)
    clf.fit((Xall[tr] - mu) / sd, D.y.values[tr])
    ps = clf.predict_proba((Xall[te] - mu) / sd)[:, 1]
    aucs.append(auc(D.y.values[te], ps))
JOINT = (np.nanmean(aucs) if aucs else np.nan, aucs)

for f, m, a in res:
    star = '✅' if m > 0.60 else ('⚠' if m > 0.55 else '❌')
    say(f'  {CN[f]:10s} AUC={m:.3f}  {star}   分段: ' + ' '.join(f'{x:.3f}' for x in a))
say(f'\n  八特征联合 AUC={JOINT[0]:.3f}' + ('  ✅' if JOINT[0] > 0.60 else '  ❌') +
    '   分段: ' + ' '.join(f'{x:.3f}' for x in JOINT[1]))

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 联合 AUC>0.60 且 4 折同向 ⇒ 标注可用：模型能从历史状态区分"该买"和"该卖"')
say('  · AUC≈0.5 ⇒ 这两类时刻的历史状态长得一样，"事后标注"喂不进去')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
