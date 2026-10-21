# -*- coding: utf-8 -*-
"""gambler_multiclass.py — 多分类：抓出来的，要能分清是哪一种局
用户主张（2026-10-16）：
  "不应该是抓出来的就包含，而是抓出来能分的清是哪一种局"

★ 此前是二分类（是/不是局起点）：报出的 307 个点里 62% 落在"无"档，其余混着弱局/微局
  ⇒ 拿到 1 个报出点，根本不知道该按什么仓位、什么持有期、什么撤出规则处理。

本脚本改成 4 类分类（同一批特征、同样三重留出）：
    0 无        不在任何局里
    1 微局      未来 24h 涨 10~20%
    2 弱局      未来 24h 涨 20~100%
    3 真局      未来 24h 涨 >=100%  且  峰后回撤 >=50%      ← 真正想要的那一类
并额外给一套"放宽真局"标签（涨>=50% 且 崩>=30%），看样本量够不够支撑训练。
输出：混淆矩阵 + 每类精确率/召回 + "报出真局的精确率"（模型说是真局，有多少真是）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_multiclass.txt')

WMAX = 288
EARLY = 12
CLASSES = ['无', '微局(10~20%)', '弱局(20~100%)', '真局(>=100%&崩>=50%)']

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_multiclass · 抓出来的要能分清是哪一种局（4 分类 + 三重留出）')
log('=' * 100)

FEAT = []
for k in (6, 12, 24, 48, 96, 288):
    FEAT += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k]
FEAT += ['hh6','hh12','hh24','hh96',
         'll_d1','ll_d3','ll_d6','ll_d12','ll_d24','ll_d48','ll_d96','ll_d288',
         'll_rank','near_ll_run','ll_vol',
         'seg_ret','evt_rate','n_trans','dur_state','run_len','accel',
         'btc_r6','btc_r12','btc_r24','btc_lead','mkt_up','mkt_dn',
         'hour_sin','hour_cos']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume','trades'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','quote_volume':'float32','trades':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
Q = pd.DataFrame({r[0]: pd.Series(r[5], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0t:.0f}s')

R1all = C.pct_change(12).values
Rnall = np.nan_to_num(R1all, nan=0.0)
up_cnt = (Rnall >= 0.05).sum(axis=1)
dn_cnt = (Rnall <= -0.05).sum(axis=1)
mkt_up_g = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
mkt_dn_g = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r6 = np.zeros(T);  btc_r6[6:]  = btc[6:]  / btc[:-6]  - 1.0
btc_r12 = np.zeros(T); btc_r12[12:] = btc[12:] / btc[:-12] - 1.0
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / btc[:-24] - 1.0

def feats(sym, ot, c, h, l, q, tr):
    n = len(c)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F = {}
    for k in (6,12,24,48,96,288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
    for k in (6,12,24,96):
        F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2,k//4)).max().values / c - 1.0
    for k in (1,3,6,12,24,48,96,288):
        lo = pd.Series(l).rolling(k, min_periods=1).min().values
        F['ll_d%d' % k] = c / np.maximum(lo, 1e-12) - 1.0
    lo288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WMAX, min_periods=72).max().values
    F['ll_rank'] = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    near = (c <= lo288 * 1.20).astype('float32')
    chg2 = np.zeros(n, dtype='float32'); chg2[1:] = (near[1:] != near[:-1]).astype('float32')
    ix = np.arange(n)
    F['near_ll_run'] = (ix - np.maximum.accumulate(np.where(chg2 > 0, ix, 0)) + 1).astype('float64')
    mv288 = rmean(q, WMAX)
    F['ll_vol'] = np.where(mv288 > 0, q / np.maximum(mv288, 1e-12), np.nan)
    F['seg_ret'] = c / np.maximum(lo288, 1e-12) - 1.0
    av = pd.Series(np.abs(r1)).rolling(48, min_periods=12).mean().values
    F['evt_rate'] = pd.Series((np.abs(r1) > 3 * av).astype('float32')).rolling(48, min_periods=12).mean().values
    sg = np.sign(np.nan_to_num(r1)); chg = np.zeros(n); chg[1:] = (sg[1:] != sg[:-1]).astype(float)
    F['n_trans'] = pd.Series(chg).rolling(48, min_periods=12).sum().values
    st = np.where(r1 >= 0.02, 2, np.where(r1 <= -0.02, 0, 1))
    dur = np.ones(n, dtype='float64')
    for i in range(1, n):
        dur[i] = dur[i-1] + 1 if st[i] == st[i-1] else 1
    F['dur_state'] = dur
    run = np.ones(n, dtype='float64')
    for i in range(1, n):
        run[i] = run[i-1] + 1 if sg[i] == sg[i-1] and sg[i] != 0 else 1
    F['run_len'] = run
    F['accel'] = F['ret6'] - F['ret24'] / 4.0
    idx = pd.Index(ot)
    b6 = pd.Series(btc_r6, index=ts).reindex(idx).values
    F['btc_r6'] = b6
    F['btc_r12'] = pd.Series(btc_r12, index=ts).reindex(idx).values
    F['btc_r24'] = pd.Series(btc_r24, index=ts).reindex(idx).values
    F['btc_lead'] = b6 - F['ret6']
    F['mkt_up'] = pd.Series(mkt_up_g, index=ts).reindex(idx).values
    F['mkt_dn'] = pd.Series(mkt_dn_g, index=ts).reindex(idx).values
    hh = (ot // 3600000) % 24
    F['hour_sin'] = np.sin(2 * np.pi * hh / 24.0)
    F['hour_cos'] = np.cos(2 * np.pi * hh / 24.0)
    return F

def starts_of(c, h, l):
    n = len(c)
    W = np.lib.stride_tricks.sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= W.min(axis=1) + 1e-12
    hmax = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    cand = np.flatnonzero(is_low & np.isfinite(hmax))
    last = -10 ** 9; out = []
    for t0 in cand:
        if t0 - last < WMAX: continue
        Hh = hmax[t0]; base = max(l[t0], 1e-12)
        gain = Hh / base - 1.0
        if gain < 0.10: continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 96)
        if not (k2 > i1 and l[i1:k2 + 1].min() <= base + (Hh - base) * 0.5):
            continue
        dd24 = 1.0 - l[i1:min(n, i1 + 97)].min() / max(h[i1], 1e-12) if i1 + 3 < n else np.nan
        out.append((t0, gain, dd24)); last = i1
    return out

X, Y4s, Y4w, TT, SS = [], [], [], [], []
for (sym, ot, c, h, l, q, tr) in RAW:
    st = starts_of(c, h, l)
    if not st: continue
    F = feats(sym, ot, c, h, l, q, tr)
    n = len(c)
    y_s = np.zeros(n, dtype='int8')     # 严格档
    y_w = np.zeros(n, dtype='int8')     # 放宽档
    for (t0, gain, dd24) in st:
        lo = t0; hi = min(n, t0 + EARLY + 1)
        if gain >= 1.0 and np.isfinite(dd24) and dd24 >= 0.50:
            y_s[lo:hi] = 3
        elif gain >= 0.20:
            y_s[lo:hi] = 2
        else:
            y_s[lo:hi] = 1
        if gain >= 0.50 and np.isfinite(dd24) and dd24 >= 0.30:
            y_w[lo:hi] = 3
        elif gain >= 0.20:
            y_w[lo:hi] = 2
        else:
            y_w[lo:hi] = 1
    m = np.isfinite(F['ret288']) & np.isfinite(F['ll_d288'])
    idx = np.flatnonzero(m)
    if len(idx) < 500: continue
    keep = np.zeros(len(idx), dtype=bool); keep[::4] = True
    keep |= (y_s[idx] > 0) | (y_w[idx] == 3)
    idx = idx[keep]
    X.append(np.stack([F[k][idx] for k in FEAT], axis=1).astype('float32'))
    Y4s.append(y_s[idx]); Y4w.append(y_w[idx])
    TT.append(ot[idx]); SS.append(np.full(len(idx), sym, dtype=object))
    del F; gc.collect()

X = np.nan_to_num(np.concatenate(X), nan=0.0, posinf=0.0, neginf=0.0)
Y4s = np.concatenate(Y4s); Y4w = np.concatenate(Y4w)
TT = np.concatenate(TT)
log(f'总样本 {len(Y4s):,}')
for tag, Yv in (('严格档', Y4s), ('放宽档', Y4w)):
    cnt = [(int((Yv == i).sum()), 100.0 * (Yv == i).mean()) for i in range(4)]
    log(f'  {tag} 类分布：' + '  '.join(f'{CLASSES[i]}={cnt[i][0]:,}({cnt[i][1]:.2f}%)' for i in range(4)))

def evaluate(Yv, tag):
    log('')
    log('=' * 100)
    log(f'【{tag}】4 分类 · 时间留出（前 70% → 后 30%）')
    log('=' * 100)
    cut = np.quantile(TT, 0.70)
    tr = TT <= cut; te = ~tr
    te_idx = np.flatnonzero(te)
    if len(np.unique(Yv[tr])) < 4:
        log('  训练段类别不足 4，跳过'); return
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    clf.fit(X[tr], Yv[tr])
    p = clf.predict_proba(X[te])
    pred = clf.classes_[np.argmax(p, axis=1)]
    truth = Yv[te]
    log(f'  训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}')
    log('')
    log('  —— 混淆矩阵（行=真实，列=预测）——')
    labs = sorted(set(list(np.unique(truth)) + list(np.unique(pred))))
    cm = confusion_matrix(truth, pred, labels=labs)
    hdr = '  真实\\预测   ' + ''.join(f'{CLASSES[l][:14]:>16}' for l in labs)
    log(hdr)
    for ri, l in enumerate(labs):
        row = ''.join(f'{cm[ri][ci]:>16,}' for ci in range(len(labs)))
        log(f'  {CLASSES[l][:12]:<12}' + row)
    log('')
    log('  —— 各类指标——')
    for l in labs:
        row = labs.index(l)
        tp = cm[row][row]; fp = sum(cm[r][row] for r in range(len(labs))) - tp
        fn = sum(cm[row]) - tp
        prec = 100.0 * tp / (tp + fp) if (tp + fp) else float('nan')
        rec = 100.0 * tp / (tp + fn) if (tp + fn) else float('nan')
        log(f'    {CLASSES[l]:<24} 样本 {int(sum(cm[row])):>8,}  精确率 {prec:>6.1f}%  召回 {rec:>6.1f}%')
    acc = 100.0 * (pred == truth).mean()
    log(f'  总体准确率 {acc:.1f}%')
    # 关键：模型说是"真局"，有多少真是真局
    m3 = (pred == 3)
    if m3.sum():
        tp3 = int((truth[m3] == 3).sum())
        log(f'  ★ 模型报"真局" {int(m3.sum()):,} 个，其中真真局 {tp3:,}（精确率 {100.0*tp3/m3.sum():.1f}%）')
    else:
        log('  ★ 模型从未报"真局"')
    # 真局召回
    m3t = (truth == 3)
    if m3t.sum():
        log(f'  ★ 真实"真局" {int(m3t.sum()):,} 个，被抓到 {int((pred[m3t]==3).sum()):,}（召回 {100.0*(pred[m3t]==3).mean():.1f}%）')

evaluate(Y4s, '严格真局(>=100%&崩>=50%)')
evaluate(Y4w, '放宽真局(>=50%&崩>=30%)')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
