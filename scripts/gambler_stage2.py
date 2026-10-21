# -*- coding: utf-8 -*-
"""gambler_stage2.py — 两阶段识别：先规则粗筛候选局，再模型精判
为什么需要这一步（train2 的诊断结论）：
  全时间点采样时正例率只有 涨0.40% / 崩0.036% / 局0.006%，
  模型输出概率被压扁（中位 0.0005），任何阈值都够不到 85% 精确率。
  正确口径：先用无前视规则把"可能正在做局"的时刻筛出来（候选集），
  再在候选集内判别 —— 正例率抬高一个数量级，85% 才可能在阈值上达到。
规则（全部只用过去数据，无前视）：
  v1 吸筹：笔数比 >= 5（相对自身 48 根均值）且 成交额比 >= 3
  v2 假跌破：近 48 根最低 <= 近 288 根最低的 0.995（创 24h 新低）
  v3 未拉完：过去 288 根涨幅 <= 30%
标签：1h 涨 >=5%  /  4h 涨 >=50%
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

t0 = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
os.makedirs(MD, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_stage2.txt')
N_COIN_LIMIT = 150

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_stage2 · 两阶段识别（规则粗筛候选 → 模型精判）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'volume', 'quote_volume', 'trades', 'taker_quote']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32', 'low': 'float32',
      'volume': 'float32', 'quote_volume': 'float32', 'trades': 'float32', 'taker_quote': 'float32'}

parts, seen = [], []
for ch in pd.read_csv(SRC, usecols=cols, dtype=dt, chunksize=1_000_000):
    if not seen:
        seen = list(pd.unique(ch['symbol']))[:N_COIN_LIMIT]
    else:
        new = [s for s in pd.unique(ch['symbol']) if s not in seen]
        if new:
            if len(seen) >= N_COIN_LIMIT:
                break
            seen = seen + new[:N_COIN_LIMIT - len(seen)]
    parts.append(ch[ch['symbol'].isin(seen)])
df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
df['symbol'] = df['symbol'].astype('category')
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0:.0f}s')

FNAMES = ['ret3', 'ret6', 'ret12', 'ret24', 'ret48', 'ret96', 'ret192', 'ret288',
          'volr12', 'volr48', 'volr192', 'vol12', 'vol48', 'vol192',
          'hh24', 'll24', 'hh96', 'll96', 'hh288', 'll288',
          'trr', 'trr_dev', 'up12', 'dn12',
          'sig_tr', 'sig_vq', 'sig_newlow']
STEP = 3
Xs, y1, y2, tsv, sv, kinds = [], [], [], [], [], []
SYMS = []

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

def rmin(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).min().values

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 2000:
        continue
    si = len(SYMS); SYMS.append(str(sym))
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    v = g['volume'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64')
    tr = g['trades'].values.astype('float64')
    tq = g['taker_quote'].values.astype('float64')
    ot = g['open_time'].values

    F = {}
    for k in (3, 6, 12, 24, 48, 96, 192, 288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
    mtr = rmean(tr, 48); mq = rmean(q, 48)
    trr = np.where(mtr > 0, tr / np.maximum(mtr, 1e-12), np.nan)
    volr = np.where(mq > 0, q / np.maximum(mq, 1e-12), np.nan)
    for k in (12, 48, 192):
        m = rmean(v, k); F['volr%d' % k] = np.where(m > 0, v / np.maximum(m, 1e-12), np.nan)
        mqk = rmean(q, k); F['vol%d' % k] = np.where(mqk > 0, q / np.maximum(mqk, 1e-12), np.nan)
    for k in (24, 96, 288):
        F['hh%d' % k] = rmax(h, k) / c - 1.0
        F['ll%d' % k] = rmin(l, k) / c - 1.0
    ra = tq / np.maximum(q, 1e-12)
    F['trr'] = ra; F['trr_dev'] = ra - rmean(ra, 48)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F['up12'] = rmean((r1 > 0).astype('float64'), 12)
    F['dn12'] = rmean((r1 < 0).astype('float64'), 12)
    F['sig_tr'] = np.nan_to_num(trr, nan=0.0)
    F['sig_vq'] = np.nan_to_num(volr, nan=0.0)
    l48 = rmin(l, 48); l288 = rmin(l, 288)
    F['sig_newlow'] = (l48 <= l288 * 0.995).astype('float64')

    # 候选（无前视）
    cand1 = (trr >= 5) & (volr >= 3) & (F['ret288'] <= 0.30)          # 吸筹 + 未拉完
    cand2 = cand1 & (F['sig_newlow'] > 0)                              # 再加假跌破
    fu12 = np.full(n, np.nan); fu12[:-12] = c[12:] / c[:-12] - 1.0
    fu48 = np.full(n, np.nan); fu48[:-48] = c[48:] / c[:-48] - 1.0
    ok = ~np.isnan(fu48) & ~np.isnan(F['ret288'])
    for kk, cm in ((1, cand1), (2, cand2)):
        idx = np.flatnonzero(cm & ok)[::STEP]
        if len(idx) == 0:
            continue
        Xs.append(np.stack([F[k][idx] for k in FNAMES], axis=1).astype('float32'))
        y1.append((fu12[idx] >= 0.05).astype('int8'))
        y2.append((fu48[idx] >= 0.50).astype('int8'))
        tsv.append(ot[idx]); sv.append(np.full(len(idx), si, dtype='int32'))
        kinds.append(np.full(len(idx), kk, dtype='int8'))

del df; gc.collect()
X = np.concatenate(Xs); Y1 = np.concatenate(y1); Y2 = np.concatenate(y2)
TS = np.concatenate(tsv); SI = np.concatenate(sv); KD = np.concatenate(kinds)
log(f'候选样本 {len(X):,}  正例率 涨{100*Y1.mean():.3f}% 局{100*Y2.mean():.3f}%  耗时 {time.time()-t0:.0f}s')

cut = np.quantile(TS, 0.70)
tr = TS < cut; te = ~tr
log(f'训练 {tr.sum():,}  测试 {te.sum():,}  切分 {pd.to_datetime(int(cut), unit="ms")}')

def report(y, p, ts, si, name):
    log(f'--- {name}  测试候选 {len(y):,}  基准 {100*y.mean():.2f}%  AUC={roc_auc_score(y, p):.4f}')
    log(f'    概率 max={p.max():.3f}  p99={np.quantile(p,0.99):.3f}  p90={np.quantile(p,0.90):.3f}  中位={np.median(p):.3f}')
    best = None
    for thr in np.arange(0.30, 1.0, 0.01):
        sel = p >= thr; n = int(sel.sum())
        if n == 0:
            continue
        prec = y[sel].mean()
        if prec >= 0.85 and (best is None or n > best[3]):
            best = (thr, prec, y[sel].sum() / max(y.sum(), 1), n)
    if best:
        log(f'  ★ 逐点 85%精确率最低阈值={best[0]:.2f}  报出 {best[3]:,}（覆盖候选 {100*best[3]/len(y):.1f}%）  精确率 {100*best[1]:.1f}%  召回 {100*best[2]:.1f}%')
    else:
        log('  ✗ 逐点：无阈值达到 85% 精确率')
    for thr in (0.3, 0.5, 0.7, 0.85, 0.95):
        ev = []
        for c in np.unique(si):
            m = np.flatnonzero((p >= thr) & (si == c))
            if len(m) == 0:
                continue
            o = np.argsort(ts[m]); m = m[o]; t = ts[m]
            brk = np.flatnonzero(np.diff(t) > 12 * 300000)
            st = np.concatenate([[m[0]], m[brk + 1]]) if len(brk) else m[:1]
            ev.extend(y[st].tolist())
        ev = np.array(ev, dtype='int8')
        if len(ev) == 0:
            log(f'      事件级 thr={thr:.2f} 无报出')
        else:
            log(f'      事件级 thr={thr:.2f}  局 {len(ev):,}  精确率 {100*ev.mean():.1f}%')

for tag, y in (('涨 +5%/1h', Y1), ('局 +50%/4h', Y2)):
    for kk, kname in ((1, '候选=吸筹&未拉完'), (2, '候选=吸筹&未拉完&假跌破')):
        m_sel = KD == kk
        if m_sel.sum() < 1000:
            continue
        Xk, yk, tk, sk = X[m_sel], y[m_sel], TS[m_sel], SI[m_sel]
        trk = tk < cut; tek = ~trk
        if yk[trk].sum() < 30:
            log(f'【{tag}｜{kname}】训练正例过少 {int(yk[trk].sum())} → 跳过')
            continue
        log('=' * 100)
        log(f'【{tag}｜{kname}】训练 {int(trk.sum()):,}（正例 {int(yk[trk].sum()):,}）  测试 {int(tek.sum()):,}')
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, min_samples_leaf=30,
                                           l2_regularization=1.0, random_state=42)
        m.fit(Xk[trk], yk[trk])
        p = m.predict_proba(Xk[tek])[:, 1]
        report(yk[tek], p, tk[tek], sk[tek], f'{tag}｜{kname}')
        fn = os.path.join(MD, 'g3_%s_%s.joblib' % (tag.split()[0], kname[:6]))
        joblib.dump({'model': m, 'features': FNAMES}, fn)
        log('  模型已存', fn)

log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
