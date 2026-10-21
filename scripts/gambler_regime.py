# -*- coding: utf-8 -*-
"""gambler_regime.py — 直接训练"正在被操盘"分类器（回应用户核心诉求）

用户给的事实（2026-10-16）：
  操盘者每个月不特定地针对一个或多个币操盘，本金翻 1.5 倍或好几倍之后马上退出，
  让币崩盘。

此前所有实验的标签都是"未来 24h 涨 X%"——那是结果，不是"局"。
本脚本改口径：先用事后规则**定义局**，再训练模型识别"局已启动、但还没拉完"的时刻。

局的定义（事后，仅用于打标签）：
  t0 = 局部低点（[t0-96, t0+96] 窗口内最低）
  H  = t0 之后 288 根（24h）内的最高价
  条件① H / low[t0] - 1 >= 1.5      （翻 1.5 倍以上）
  条件② H 之后 48 根内最低 <= H*0.6  （拉完马上崩）
  满足①② ⇒ t0 是一次"局"的起点

正例（模型要识别的）：局启动后、涨幅在 +5% ~ +100% 之间的时刻（"已在局中但没拉完"）
负例：其余时刻（按时间抽样）

特征：45 个（多尺度 + 段特征，与 gambler_window 相同口径）
切分：时间留出（前 70% 训练 / 后 30% 测试），无随机切分
评估：逐点精确率 + 事件级（每个局只算一次：局启动后首次被报出即命中）
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

t0t = time.time()
D  = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
os.makedirs(MD, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_502.csv')
OUT = os.path.join(EV, 'gambler_regime.txt')
N_COIN_LIMIT = 150
WMAX = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_regime · "正在被操盘"分类器（先定义局，再识别局）')
log('=' * 100)

cols = ['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades']
dt = {'symbol': 'object', 'open_time': 'int64', 'close': 'float32', 'high': 'float32',
      'low': 'float32', 'quote_volume': 'float32', 'trades': 'float32'}
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
log(f'行数 {len(df):,}  币 {df["symbol"].nunique()}  耗时 {time.time()-t0t:.0f}s')

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

SCALES = (6, 12, 24, 48, 96, 144, 288)
FNAMES = []
for k in SCALES:
    FNAMES += ['ret%d' % k, 'volr%d' % k, 'trr%d' % k, 'hh%d' % k, 'll%d' % k]
FNAMES += ['since_hi24', 'since_lo24', 'run_up', 'run_dn', 'seg_ret', 'seg_volr', 'seg_bars']
STEP = 3

Xs, YS, TS, SS = [], [], [], []
SYMS = []
regimes = []          # 只记统计：每个局 (symbol, t0, gain_final)

for sym, g in df.groupby('symbol', observed=True, sort=False):
    n = len(g)
    if n < 2000:
        continue
    si = len(SYMS); SYMS.append(str(sym))
    c = g['close'].values.astype('float64')
    h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64')
    tr = g['trades'].values.astype('float64')
    ot = g['open_time'].values

    # ---- 1) 找局（事后）----
    from numpy.lib.stride_tricks import sliding_window_view
    # 每个 i 的相对 24h 内最高价（后向窗，含未来 ⇒ 仅用于打标签）
    fh = pd.Series(h).rolling(WMAX, min_periods=1).max().shift(-WMAX).values
    fl = pd.Series(l).rolling(WMAX, min_periods=1).min().shift(-WMAX).values
    is_low = np.zeros(n, dtype=bool)
    Wl = sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
    is_low = l <= Wl.min(axis=1) + 1e-12          # 局部低点（±96 根）
    cand = np.flatnonzero(is_low & np.isfinite(fh))
    last_t1 = -10 ** 9
    lab = np.zeros(n, dtype='int8')               # 1 = 局内早期
    for t0 in cand:
        if t0 - last_t1 < WMAX:                    # 同一局去重
            continue
        H = fh[t0]
        if not np.isfinite(H) or H <= 0:
            continue
        if H / max(l[t0], 1e-12) - 1.0 < 0.50:     # 条件① 涨 50% 以上（对应"本金翻 1.5 倍"）
            continue
        i1 = t0 + int(np.argmax(h[t0:t0 + WMAX]))
        k2 = min(n - 1, i1 + 48)
        if k2 <= i1:
            continue
        after_min = l[i1:k2 + 1].min()
        if after_min > H * 0.70:                   # 条件② 拉完崩（回撤 >=30%）
            continue
        regimes.append((os.path.basename(sym), int(ot[t0]), float(H / max(l[t0], 1e-12) - 1.0)))
        # 正例 = 从 t0 到"涨幅达 100%"（或 96 根）为止，且涨幅在 +5%~+100%
        seg = slice(t0, min(n, t0 + 96))
        cseg = c[seg]
        gain = cseg / max(l[t0], 1e-12) - 1.0
        lab[seg] = ((gain >= 0.03) & (gain <= 0.50)).astype('int8')
        last_t1 = i1

    # ---- 2) 特征（只用当期及过去，无前视）----
    F = {}
    for k in SCALES:
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
        mv = rmean(q, k); F['volr%d' % k] = np.where(mv > 0, q / np.maximum(mv, 1e-12), np.nan)
        mt = rmean(tr, k); F['trr%d' % k] = np.where(mt > 0, tr / np.maximum(mt, 1e-12), np.nan)
        F['hh%d' % k] = pd.Series(h).rolling(k, min_periods=max(2, k // 4)).max().values / c - 1.0
        F['ll%d' % k] = pd.Series(l).rolling(k, min_periods=max(2, k // 4)).min().values / c - 1.0
    Wl2 = sliding_window_view(l, WMAX); Wh2 = sliding_window_view(h, WMAX)
    s_lo = np.full(n, np.nan); s_hi = np.full(n, np.nan)
    s_lo[WMAX - 1:] = WMAX - 1 - Wl2.argmin(axis=1)
    s_hi[WMAX - 1:] = WMAX - 1 - Wh2.argmax(axis=1)
    F['since_lo24'] = s_lo; F['since_hi24'] = s_hi
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    sg = np.sign(np.nan_to_num(r1, nan=0.0))
    ru = np.zeros(n, dtype='float64'); rd = np.zeros(n, dtype='float64')
    cu = cd = 0
    for i in range(n):
        cu = cu + 1 if sg[i] > 0 else 0
        cd = cd + 1 if sg[i] < 0 else 0
        ru[i] = cu; rd[i] = cd
    F['run_up'] = ru; F['run_dn'] = rd
    l288 = pd.Series(l).rolling(WMAX, min_periods=72).min().values
    F['seg_ret'] = c / np.maximum(l288, 1e-12) - 1.0
    qm = rmean(q, WMAX)
    F['seg_volr'] = np.where(qm > 0, q / np.maximum(qm, 1e-12), np.nan)
    F['seg_bars'] = s_lo

    idx = np.arange(0, n - WMAX, STEP)
    if len(idx) == 0:
        continue
    X = np.stack([F[k][idx] for k in FNAMES], axis=1).astype('float32')
    m = ~np.isnan(X).any(axis=1)
    idx = idx[m]; X = X[m]
    if len(idx) == 0:
        continue
    Xs.append(X); YS.append(lab[idx]); TS.append(ot[idx]); SS.append(np.full(len(idx), si, dtype='int32'))

del df; gc.collect()
X = np.concatenate(Xs); Y = np.concatenate(YS); TS = np.concatenate(TS); SS = np.concatenate(SS)
log(f'样本 {len(X):,}  特征 {X.shape[1]}  正例（局内早期）{int(Y.sum()):,}（{100*Y.mean():.3f}%）')

RG = pd.DataFrame(regimes, columns=['symbol', 't0', 'gain'])
log(f'')
log('=' * 100)
log('【A】局普查（涨 50% 以上 + 拉完崩 30%）')
log('=' * 100)
if len(RG):
    RG['year'] = pd.to_datetime(RG['t0'], unit='ms').dt.year
    log(f'  局总数 {len(RG):,}  涉及币 {RG["symbol"].nunique()}  翻倍幅度中位 {100*RG["gain"].median():.0f}%')
    log('  按年：' + '  '.join(f'{y}:{v}' for y, v in RG.groupby('year').size().items()))
    log('  币出现次数 top10：' + '  '.join(f'{s}:{v}' for s, v in RG['symbol'].value_counts().head(10).items()))
    days = pd.to_datetime(RG['t0'], unit='ms').dt.date.nunique()
    span_days = (RG['t0'].max() - RG['t0'].min()) / 86400000
    log(f'  有局的天数 {days} / 跨度 {span_days:.0f} 天 = {100*days/max(span_days,1):.0f}% 的天数有局')
else:
    log('  ✗ 未找到符合条件的局（阈值需调整）')

cut = np.quantile(TS, 0.60)
tr = TS < cut; te = ~tr
log('')
log(f'训练 {tr.sum():,}（正例 {int(Y[tr].sum()):,}）  测试 {te.sum():,}（正例 {int(Y[te].sum()):,}）  切分 {pd.to_datetime(int(cut), unit="ms")}')

if Y[tr].sum() >= 50:
    log('=' * 100)
    log('【B】训练 + 时间留出评估')
    log('=' * 100)
    m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=30,
                                        l2_regularization=1.0, random_state=42)
    m_.fit(X[tr], Y[tr])
    p = m_.predict_proba(X[te])[:, 1]
    log(f'  AUC={roc_auc_score(Y[te], p):.4f}  测试基准 {100*Y[te].mean():.3f}%')
    log(f'  概率 max={p.max():.3f} p99={np.quantile(p,0.99):.3f} p90={np.quantile(p,0.90):.3f} 中位={np.median(p):.4f}')
    best = None
    for thr in np.arange(0.10, 1.0, 0.01):
        sel = p >= thr; nn = int(sel.sum())
        if nn == 0:
            continue
        pr = Y[te][sel].mean()
        if pr >= 0.85 and (best is None or nn > best[3]):
            best = (thr, pr, Y[te][sel].sum() / max(Y[te].sum(), 1), nn)
    if best:
        log(f'  ★ 85%精确率最低阈值={best[0]:.2f}  报出 {best[3]:,}（覆盖 {100*best[3]/len(p):.2f}%）  精确率 {100*best[1]:.1f}%  召回 {100*best[2]:.1f}%')
    else:
        log('  ✗ 逐点：无阈值达到 85% 精确率')
    log('  阈值扫描（逐点）：')
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        sel = p >= thr; nn = int(sel.sum())
        if nn == 0:
            log(f'    thr={thr:.2f} 无报出'); continue
        log(f'    thr={thr:.2f}  报出 {nn:,}  精确率 {100*Y[te][sel].mean():.1f}%  召回 {100*Y[te][sel].sum()/max(Y[te].sum(),1):.1f}%')
    joblib.dump({'model': m_, 'features': FNAMES, 'wmax': WMAX}, os.path.join(MD, 'gr_regime.joblib'))
    log('  模型已存', os.path.join(MD, 'gr_regime.joblib'))
else:
    log('训练正例过少，未训练')

log(f'TOTAL {time.time()-t0t:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
