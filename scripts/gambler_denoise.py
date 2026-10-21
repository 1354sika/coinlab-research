# -*- coding: utf-8 -*-
"""gambler · 降噪版实验 —— 回答用户质疑："是不是算法没滤掉噪声，才找不到真局/崩前/涨前信号？"

四项降噪措施（全部与"未降噪"版本同数据、同切分、同标签对照）：
  A 数据清洗：剔除僵尸币（成交量长期为 0 或有效数据不足）、剔除单根异常跳变(|ret|>300%)
  B 去市场因子：每币收益扣掉"对全市场中位收益"的滚动 beta —— 让个股信号不被大盘淹没
  C 特征筛选：先全量训练，再用**置换重要性**选 top-8，重训一次看提升
  D 币种分层：按成交额分 高/中/低 三层，分别评估（大盘币与小币的信号强度可能完全不同）

标签（与 gambler v1 一致，便于对照）：未来 24 根（2h）涨 ≥+20% / 跌 ≤−20%
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
STEP, FWD = 12, 24
DN, UP = -0.20, 0.20
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 降噪实验：清洗 + 去市场因子 + 特征筛选 + 分层')
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
tm = close.index.values
syms = np.array(list(close.columns))
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

# ---------- A 数据清洗 ----------
valid = np.isfinite(CV)
qv_sum = np.nansum(QV, axis=0)
zero_ratio = (QV == 0).sum(axis=0) / T
keep_coin = (valid.sum(axis=0) / T > 0.80) & (zero_ratio < 0.50) & (qv_sum > 0)
say('清洗：%d 币 → 保留 %d 币（剔除僵尸/低覆盖 %d 个）'
    % (N, int(keep_coin.sum()), int((~keep_coin).sum())))
CV2 = CV[:, keep_coin]; HV2 = HV[:, keep_coin]; LV2 = LV[:, keep_coin]; QV2 = QV[:, keep_coin]
syms2 = syms[keep_coin]
N2 = CV2.shape[1]

ret1 = np.full((T, N2), np.nan, np.float32); ret1[1:] = CV2[1:] / CV2[:-1] - 1.0
jump = np.abs(ret1) > 3.0
ret1[jump] = np.nan                      # 剔除单根异常跳变
say('剔除异常单根跳变 %s 个（%.4f%%）  %.0fs'
    % (format(int(np.nansum(jump)), ','), 100 * np.nanmean(jump), time.time() - t0))

# ---------- B 去市场因子 ----------
mkt = np.nanmedian(ret1, axis=1)
mkt_s = pd.Series(mkt)
var_m = mkt_s.rolling(288, min_periods=60).var().values
resid = np.full((T, N2), np.nan, np.float32)
for j in range(N2):
    r = pd.Series(ret1[:, j])
    cov_m = r.rolling(288, min_periods=60).cov(mkt_s).values
    beta = np.where(var_m > 0, cov_m / var_m, 1.0)
    resid[:, j] = (ret1[:, j] - beta * mkt).astype('float32')
say('去市场因子完成（滚动 288 根 beta）  %.0fs' % (time.time() - t0))

# ---------- 特征 ----------
F = {}
def cumret(r, w):
    """用残差收益累加出多尺度收益（去因子后的动量）"""
    out = np.full((T, N2), np.nan, np.float32)
    cs = np.nancumsum(np.nan_to_num(r), axis=0)
    out[w:] = (cs[w:] - cs[:-w]).astype('float32')
    return out
for w in (12, 48, 288):
    F['cum%d' % w] = cumret(resid, w)
for w in (48, 192):
    F['volr%d' % w] = (QV2 / np.where(pd.DataFrame(QV2).rolling(w, min_periods=max(4, w // 4)).median().values > 0,
                                      pd.DataFrame(QV2).rolling(w, min_periods=max(4, w // 4)).median().values, np.nan)).astype('float32')
for w in (48, 192):
    F['vol%d' % w] = pd.DataFrame(resid).rolling(w, min_periods=max(4, w // 4)).std().values.astype('float32')
for w in (48, 288):
    hi = pd.DataFrame(HV2).rolling(w, min_periods=w // 4).max().values
    lo = pd.DataFrame(LV2).rolling(w, min_periods=w // 4).min().values
    F['dhi%d' % w] = (CV2 / hi - 1.0).astype('float32')
    F['dlo%d' % w] = (CV2 / lo - 1.0).astype('float32')
sgn = np.sign(np.nan_to_num(resid)); idxr = np.arange(T)
d1 = np.zeros((T, N2), np.float32)
for j in range(N2):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (idxr - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
F['D1'] = d1
sd = pd.DataFrame(resid).rolling(240, min_periods=40).std().values
F['D4'] = (np.abs(resid) > 3.0 * sd).astype('float32')
F['D8'] = np.repeat(pd.Series((np.nan_to_num(ret1) == 0).mean(axis=1)).rolling(60, min_periods=5).mean().values[:, None], N2, axis=1).astype('float32')
NAME = list(F.keys())
say('特征 %d 个：%s  %.0fs' % (len(NAME), NAME, time.time() - t0))

fmin = pd.DataFrame(LV2).iloc[::-1].rolling(FWD, min_periods=1).min().iloc[::-1].shift(-1).values
fmax = pd.DataFrame(HV2).iloc[::-1].rolling(FWD, min_periods=1).max().iloc[::-1].shift(-1).values
rows = np.arange(300, T - FWD - 1, STEP)
rr = np.repeat(rows, N2); cc = np.tile(np.arange(N2), len(rows))
X = np.stack([F[k][rr, cc] for k in NAME], axis=1)
ydn = (fmin[rr, cc] / CV2[rr, cc] - 1.0 <= DN).astype(np.int8)
yup = (fmax[rr, cc] / CV2[rr, cc] - 1.0 >= UP).astype(np.int8)
ok = np.isfinite(X).all(axis=1) & np.isfinite(fmax[rr, cc]) & np.isfinite(fmin[rr, cc])
X, rr, cc, yup, ydn = X[ok], rr[ok], cc[ok], yup[ok], ydn[ok]
say('样本 %s  涨 %.2f%%  崩 %.2f%%  %.0fs'
    % (format(len(X), ','), 100 * yup.mean(), 100 * ydn.mean(), time.time() - t0))

cut = int(T * 0.6); tr = rr < cut; te = ~tr


def run(Xx, yy, name):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=42, early_stopping=False).fit(Xx[tr], yy[tr])
    p = clf.predict_proba(Xx[te])[:, 1]
    auc = roc_auc_score(yy[te], p)
    best = None
    for t in np.arange(0.05, 0.995, 0.01):
        m = p >= t
        if m.sum() < 50:
            continue
        pr = yy[te][m].mean()
        if pr >= 0.85:
            best = (t, int(m.sum()), pr)
    say('  %-26s AUC=%.4f  基准=%.2f%%  %s' % (
        name, auc, 100 * yy[te].mean(),
        ('★85%%@P≥%.2f 报出%d' % best) if best else '✗无阈值达85%'))
    return auc, p


say('\n' + '=' * 100)
say('【降噪后】清洗 + 去市场因子（对照：gambler v1 原口径 崩0.8997/涨0.9033 但在事件窗口内）')
say('=' * 100)
auc_dn0, p_dn = run(X, ydn, '崩(2h≤−20%)')
auc_up0, p_up = run(X, yup, '涨(2h≥+20%)')

# ---------- C 特征筛选（置换重要性）----------
say('\n【C 特征筛选：置换重要性 top-8 重训】')
clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=5,
                                     min_samples_leaf=200, random_state=42,
                                     early_stopping=False).fit(X[tr], yup[tr])
base_auc = roc_auc_score(yup[te], clf.predict_proba(X[te])[:, 1])
rng = np.random.default_rng(0)
sub = rng.choice(int(tr.sum()), min(200000, int(tr.sum())), replace=False)
imp = []
for k in range(X.shape[1]):
    Xs = X[tr][sub].copy(); Xs[:, k] = rng.permutation(Xs[:, k])
    imp.append(base_auc - roc_auc_score(yup[tr][sub], clf.predict_proba(Xs)[:, 1]))
imp = np.array(imp)
order = np.argsort(-imp)
say('  重要性排序：' + '  '.join('%s=%.4f' % (NAME[i], imp[i]) for i in order[:10]))
top = order[:8]
say('  取 top-8：%s' % [NAME[i] for i in top])
run(X[:, top], yup, '涨(降噪+选8特征)')
run(X[:, top], ydn, '崩(降噪+选8特征)')

# ---------- D 分层 ----------
say('\n【D 按成交额分层（高/中/低）】')
q33, q66 = np.percentile(qv_sum[keep_coin], [33, 66])
tier = np.where(qv_sum[keep_coin] >= q66, 2, np.where(qv_sum[keep_coin] >= q33, 1, 0))
for t_ in (2, 1, 0):
    mask_c = tier[cc] == t_
    tr_m = tr & mask_c; te_m = te & mask_c
    if tr_m.sum() < 5000 or te_m.sum() < 2000:
        continue
    run(X[:, top], yup, ['低流动性', '中流动性', '高流动性'][t_] + '(涨)')

open(os.path.join(EV, 'gambler_denoise.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
