# -*- coding: utf-8 -*-
"""gambler · 降噪 v2（修正版）—— 上一版的 4 个错误已修

上一版错在哪（用户指出 + 自查）：
  ① **剔除了 46% 的币**（304→164）—— 但 820 个"局"遍布 277/304 个币，**小币才是操盘主战场**
     ⇒ 那一刀等于把答案扔进垃圾桶（方向性错误）
  ② "剔除异常跳变"只剔了 **1 个点** —— 形同虚设
  ③ **一律去市场因子** —— 若操盘手的痕迹正是"脱离大盘独立走强"，扣除就把信号扣掉了；
     且 beta 用历史估计，小币 beta 极不稳定
  ④ **置换重要性用训练子集算** —— 有偏，导致"重要性全为负"的假象

修正版做法：
  A **保留全部 304 币**；只在**流动性分层**（小/中/大）上分别训练与评估
  B 异常处理只做"极端成交量"（量比 >200 视为假数据 → 置 NaN）
  C 去市场因子做 **A/B 对照**（去 vs 不去，谁高用谁）
  D 重要性用**测试段**算
标签：崩(2h≤−20%) / 涨(2h≥+20%) / 8h翻倍（早期识别）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD, HOR8 = 12, 24, 96
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 降噪 v2（修正版）：保全部币 + 分层 + 去因子 A/B + 测试段算重要性')
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
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d（**全部币，不剔除**）  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
med60 = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
med60 = np.where(med60 > 0, med60, np.nan)
volr60 = QV / med60
bad = np.isfinite(volr60) & (volr60 > 200)
ret1[bad] = np.nan
say('剔除极端成交量点（量比>200）：%s 个（%.5f%%）  %.0fs'
    % (format(int(np.nansum(bad)), ','), 100 * np.nanmean(bad), time.time() - t0))

mkt = np.nanmedian(ret1, axis=1)
mkt_s = pd.Series(mkt)
var_m = mkt_s.rolling(288, min_periods=60).var().values


def defactor(r):
    out = np.full(r.shape, np.nan, np.float32)
    for j in range(r.shape[1]):
        cov = pd.Series(r[:, j]).rolling(288, min_periods=60).cov(mkt_s).values
        beta = np.where(var_m > 0, cov / var_m, 1.0)
        out[:, j] = (r[:, j] - beta * mkt).astype('float32')
    return out


def build(r):
    F = {}
    for w in (12, 48, 288):
        a = np.full((T, N), np.nan, np.float32)
        cs = np.nancumsum(np.nan_to_num(r), axis=0)
        a[w:] = cs[w:] - cs[:-w]
        F['cum%d' % w] = a
    for w in (12, 48):
        m = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
        F['volr%d' % w] = (QV / np.where(m > 0, m, np.nan)).astype('float32')
    for w in (48, 192):
        F['vol%d' % w] = pd.DataFrame(r).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
    for w in (48, 288):
        hi = pd.DataFrame(HV).rolling(w, min_periods=w // 4).max().values
        lo = pd.DataFrame(LV).rolling(w, min_periods=w // 4).min().values
        F['dhi%d' % w] = (CV / hi - 1.0).astype('float32')
        F['dlo%d' % w] = (CV / lo - 1.0).astype('float32')
    sgn = np.sign(np.nan_to_num(r)); idxr = np.arange(T)
    d1 = np.zeros((T, N), np.float32)
    for j in range(N):
        col = sgn[:, j]
        newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
        seg_id = np.cumsum(newseg) - 1
        start = np.where(newseg)[0][seg_id]
        run = (idxr - start + 1).astype(np.float32)
        d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
    F['D1'] = d1
    sd = pd.DataFrame(r).rolling(240, min_periods=40).std().values
    F['D4'] = (np.abs(r) > 3.0 * sd).astype('float32')
    F['D8'] = np.repeat(pd.Series((np.nan_to_num(ret1) == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N, axis=1).astype('float32')
    F['mkt48'] = np.repeat(pd.Series(mkt).rolling(48, min_periods=8).sum().values[:, None], N, axis=1).astype('float32')
    return F


fmin = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fmax = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
fmax8 = pd.DataFrame(HV).iloc[::-1].rolling(HOR8, min_periods=1).max().iloc[::-1].shift(-1).values
rows = np.arange(300, T - HOR8 - 1, STEP)
rr = np.repeat(rows, N); cc = np.tile(np.arange(N), len(rows))
cut = int(T * 0.6)

qv_sum = np.nansum(QV, axis=0)
q33, q66 = np.percentile(qv_sum, [33, 66])
tier = np.where(qv_sum >= q66, 2, np.where(qv_sum >= q33, 1, 0))
say('流动性分层：大 %d 币 / 中 %d 币 / 小 %d 币'
    % (int((tier == 2).sum()), int((tier == 1).sum()), int((tier == 0).sum())))

LAB = {
    '崩(2h≤−20%)': (fmin[rr, cc] / CV[rr, cc] - 1.0 <= DN).astype(np.int8),
    '涨(2h≥+20%)': (fmax[rr, cc] / CV[rr, cc] - 1.0 >= UP).astype(np.int8),
    '8h翻倍(早期识别)': (fmax8[rr, cc] / CV[rr, cc] - 1.0 >= 1.0).astype(np.int8),
}

say('\n' + '=' * 100)
say('【C 去市场因子 A/B 对照】（全币、同切分、同标签）')
say('=' * 100)
RES = {}
for tag, r in (('不去因子', ret1), ('去市场因子', defactor(ret1))):
    F = build(r)
    NAME = list(F.keys())
    X = np.stack([F[k][rr, cc] for k in NAME], axis=1)
    ok = np.isfinite(X).all(axis=1)
    del F; gc.collect()
    tr = rr[ok] < cut; te = ~tr
    say('  [%s] 特征 %d 个，样本 %s' % (tag, len(NAME), format(int(ok.sum()), ',')))
    for lname, y in LAB.items():
        yy = y[ok]
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                             min_samples_leaf=200, l2_regularization=1.0,
                                             random_state=42, early_stopping=False).fit(X[ok][tr], yy[tr])
        p = clf.predict_proba(X[ok][te])[:, 1]
        auc = roc_auc_score(yy[te], p)
        RES[(tag, lname)] = (auc, X[ok], yy, tr, te, p, NAME, ok)
        say('    %-20s AUC=%.4f  基准=%.2f%%' % (lname, auc, 100 * yy[te].mean()))

best_tag = '不去因子' if RES[('不去因子', '8h翻倍(早期识别)')][0] >= RES[('去市场因子', '8h翻倍(早期识别)')][0] else '去市场因子'
say('\n  → 以"8h翻倍"为准，较优的是【%s】，用它做分层与重要性' % best_tag)

say('\n' + '=' * 100)
say('【A 流动性分层】（每层单独训练+测试，全部币参与）')
say('=' * 100)
Xb = RES[(best_tag, '崩(2h≤−20%)')][1]
tr0, te0 = RES[(best_tag, '崩(2h≤−20%)')][3], RES[(best_tag, '崩(2h≤−20%)')][4]
okb = RES[(best_tag, '崩(2h≤−20%)')][7]
cco = cc[okb]
for t_ in (2, 1, 0):
    lay = ['小币', '中币', '大币'][t_]
    mask_c = (tier[cco] == t_)
    for lname, y in LAB.items():
        yok = y[okb]
        tr = tr0 & mask_c; te = te0 & mask_c
        if tr.sum() < 5000 or te.sum() < 2000 or len(np.unique(yok[tr])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                             min_samples_leaf=200, random_state=42,
                                             early_stopping=False).fit(Xb[tr], yok[tr])
        p = clf.predict_proba(Xb[te])[:, 1]
        say('  %-4s %-20s n=%9s AUC=%.4f  基准=%.2f%%'
            % (lay, lname, format(int(te.sum()), ','), roc_auc_score(yok[te], p), 100 * yok[te].mean()))

say('\n' + '=' * 100)
say('【D 置换重要性（用**测试段**算 —— 上一版用训练子集算，全为负是偏差）】')
say('=' * 100)
auc0, Xb2, yb, tr, te, p, names, _okb2 = RES[(best_tag, '8h翻倍(早期识别)')]
clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=5,
                                     min_samples_leaf=200, random_state=42,
                                     early_stopping=False).fit(Xb2[tr], yb[tr])
rng = np.random.default_rng(0)
sub = rng.choice(int(te.sum()), min(150000, int(te.sum())), replace=False)
Xte = Xb2[te][sub]; yte = yb[te][sub]
base = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
imp = []
for k in range(Xte.shape[1]):
    X2 = Xte.copy(); X2[:, k] = rng.permutation(X2[:, k])
    imp.append(base - roc_auc_score(yte, clf.predict_proba(X2)[:, 1]))
order = np.argsort(-np.array(imp))
say('  base AUC(测试子集) = %.4f' % base)
say('  ' + '  '.join('%s=%+.4f' % (names[i], imp[i]) for i in order[:12]))

open(os.path.join(EV, 'gambler_denoise2.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
