# -*- coding: utf-8 -*-
"""gambler · 同伴动力学实验 —— 回答用户："那价格关联的币，震动的方向也考虑了吗？"

诚实回答：**没有**。前面那版只算了"全市场整体共振度"（所有币一起的平均），
既没区分"哪些币与它价格关联"，也没算"这些关联币的震动方向"。

本实验补的就是这一块：
  ① 先定"价格关联"：用**训练段**（前 60% 时间）的 1 根收益算 304×304 相关矩阵，
     每个币取相关性最高的 K=10 个邻居，权重 = 归一化相关系数（实时可得，无前视）
  ② 再算"关联币的震动方向"：
       peer_ret12   关联币过去 1h 收益的加权平均（同伴动量，带符号）
       peer_ret48   关联币过去 4h 收益的加权平均
       peer_dir     关联币中"过去 1h 上涨"的占比（方向一致性）
       peer_lead    peer_ret12 − 自身 ret12 ← **同伴已经动了、我还没动**（滞后传导）
       peer_lead48  peer_ret48 − 自身 ret48
       peer_vol     关联币过去 1h 波动率均值（同伴的躁动程度）
  ③ 评估：同样标签下，加"同伴特征"能否提升 AUC（与仅自身特征对照）
     重点看 **8h 翻倍（早期识别）** 能不能被抬起来。
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD, HOR8, K = 12, 24, 96, 10
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 同伴动力学：价格关联币的"震动方向"能不能帮我们早期识别')
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
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
ret12 = np.full((T, N), np.nan, np.float32); ret12[12:] = CV[12:] / CV[:-12] - 1.0
ret48 = np.full((T, N), np.nan, np.float32); ret48[48:] = CV[48:] / CV[:-48] - 1.0
cut = int(T * 0.6)

# ---------- ① 关联币（用训练段算相关，实时可得） ----------
tr_ret = ret1[:cut]
tr_ret = np.where(np.isfinite(tr_ret), tr_ret, np.nan)
mu = np.nanmean(tr_ret, axis=0)
sd = np.nanstd(tr_ret, axis=0)
sd = np.where(sd > 0, sd, np.nan)
Z = (tr_ret - mu) / sd
Zf = np.nan_to_num(Z)
C = (Zf.T @ Zf) / max(1, Z.shape[0])          # 相关系数矩阵（近似，缺失置 0）
np.fill_diagonal(C, -np.inf)
W = np.zeros((N, N), np.float32)
for j in range(N):
    idx = np.argpartition(-C[j], K)[:K]
    w = np.clip(C[j, idx], 0, None)
    s = w.sum()
    if s > 0:
        W[j, idx] = (w / s).astype(np.float32)
say('关联矩阵算完（每币 top-%d 邻居）  %.0fs' % (K, time.time() - t0))

# ---------- ② 同伴的震动方向 ----------
PEER = {}
PEER['peer_ret12'] = (np.nan_to_num(ret12) @ W.T).astype('float32')
PEER['peer_ret48'] = (np.nan_to_num(ret48) @ W.T).astype('float32')
up12 = np.nan_to_num((ret12 > 0).astype('float32'))
PEER['peer_dir'] = (up12 @ W.T).astype('float32')
PEER['peer_lead'] = (PEER['peer_ret12'] - np.nan_to_num(ret12)).astype('float32')
PEER['peer_lead48'] = (PEER['peer_ret48'] - np.nan_to_num(ret48)).astype('float32')
vol12 = pd.DataFrame(ret1).rolling(12, min_periods=4).std().values
PEER['peer_vol'] = (np.nan_to_num(vol12) @ W.T).astype('float32')
NAME_P = list(PEER.keys())
say('同伴特征 %d 个：%s  %.0fs' % (len(NAME_P), NAME_P, time.time() - t0))

# ---------- 自身基线特征 ----------
F = {}
for w in (12, 48, 288):
    a = np.full((T, N), np.nan, np.float32); a[w:] = CV[w:] / CV[:-w] - 1.0
    F['ret%d' % w] = a
for w in (12, 48):
    med = pd.DataFrame(QV).rolling(w, min_periods=max(4, w // 4)).median().values
    F['volr%d' % w] = (QV / np.where(med > 0, med, np.nan)).astype('float32')
for w in (48, 192):
    F['vol%d' % w] = pd.DataFrame(ret1).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
for w in (48, 288):
    hi = pd.DataFrame(HV).rolling(w, min_periods=w // 4).max().values
    lo = pd.DataFrame(LV).rolling(w, min_periods=w // 4).min().values
    F['dhi%d' % w] = (CV / hi - 1.0).astype('float32')
    F['dlo%d' % w] = (CV / lo - 1.0).astype('float32')
sgn = np.sign(np.nan_to_num(ret1)); idxr = np.arange(T)
d1 = np.zeros((T, N), np.float32)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (idxr - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
F['D1'] = d1
sdv = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
F['D4'] = (np.abs(ret1) > 3.0 * sdv).astype('float32')
NAME_B = list(F.keys())

fmin = pd.DataFrame(LV).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fmax = pd.DataFrame(HV).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
fmax8 = pd.DataFrame(HV).iloc[::-1].rolling(HOR8, min_periods=1).max().iloc[::-1].shift(-1).values

rows = np.arange(300, T - HOR8 - 1, STEP)
rr = np.repeat(rows, N); cc = np.tile(np.arange(N), len(rows))
XB = np.stack([F[k][rr, cc] for k in NAME_B], axis=1)
XP = np.stack([PEER[k][rr, cc] for k in NAME_P], axis=1)
ydn = (fmin[rr, cc] / CV[rr, cc] - 1.0 <= DN).astype(np.int8)
yup = (fmax[rr, cc] / CV[rr, cc] - 1.0 >= UP).astype(np.int8)
y8 = (fmax8[rr, cc] / CV[rr, cc] - 1.0 >= 1.0).astype(np.int8)
ok = np.isfinite(XB).all(axis=1) & np.isfinite(XP).all(axis=1)
XB, XP, rr, cc, ydn, yup, y8 = XB[ok], XP[ok], rr[ok], cc[ok], ydn[ok], yup[ok], y8[ok]
tr = rr < cut; te = ~tr
say('样本 %s  崩 %.2f%%  涨 %.2f%%  8h翻倍 %.3f%%  %.0fs'
    % (format(len(XB), ','), 100 * ydn.mean(), 100 * yup.mean(), 100 * y8.mean(), time.time() - t0))


def run(Xx, yy, tag):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=42, early_stopping=False).fit(Xx[tr], yy[tr])
    p = clf.predict_proba(Xx[te])[:, 1]
    auc = roc_auc_score(yy[te], p)
    say('  %-32s AUC=%.4f  基准=%.2f%%' % (tag, auc, 100 * yy[te].mean()))
    return auc


say('\n' + '=' * 100)
say('【对照】仅自身特征 vs 自身+同伴特征（价格关联币的震动方向）')
say('=' * 100)
for tag, yy in (('崩(2h≤−20%)', ydn), ('涨(2h≥+20%)', yup), ('8h翻倍(早期识别)', y8)):
    a1 = run(XB, yy, tag + ' · 自身')
    a2 = run(np.hstack([XB, XP]), yy, tag + ' · +同伴')
    say('      → 提升 %+.4f' % (a2 - a1))

open(os.path.join(EV, 'gambler_peer.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
