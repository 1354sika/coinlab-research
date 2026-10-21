# -*- coding: utf-8 -*-
"""analyze_lowfreq.py — 低频「赌手模式」：每 3 天一笔、持有 3 天、+50% 止盈

用户描述的赌手：总共几十笔，每 3 天不到一次摸到 +50%，只看 K 线。
前面所有实验都是高频（每天建仓 / 每小时一笔），没测过这个低频口径。

本脚本：
  【一】样本：每 3 天一个决策点，未来 3 天 MFE >= +50% 且 MAE >= -30% 为正样本
  【二】训练：10 个 K 线特征，时间切分 70/30
  【三】回测：每 3 天取 p 最高的币，TP=+50% / 止损 -30% / 持满 3 天，
              逐笔复利，统计单笔期望、多少笔到 1.5x

诚实边界：零滑点、成本 0.09%、MFE/MAE 同窗极值先后未知（按先触止损保守）、
          样本低频但同币相邻点仍部分重叠、只回测验证段。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

log('=' * 100)
log(f'lowfreq · 每 3 天一笔 / 持有 3 天 / +50% 止盈   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 100)

STEP = 864          # 3 天
H = 864             # 持有 3 天
W_L = 288 * 30      # 30 天回看
FEE = 0.0009
TP, STOP = 0.50, 0.30

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'high': 'float32',
                             'low': 'float32', 'close': 'float32', 'volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 25000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['high'].values.astype('float64'), g['low'].values.astype('float64'),
                    g['close'].values.astype('float64'), g['volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)} 载入 {time.time()-t0:.0f}s')

FEAT = ['r6', 'r24', 'r96', 'r288', 'volr', 'sd96', 'pos96', 'swr', 'amp_ratio', 'ntr']
rows = []
for (sym, ts, h, l, c, v) in RAW:
    T = len(c)
    if T < W_L + H + STEP + 10:
        continue
    idx = np.arange(W_L, T - H - 1, STEP)
    if len(idx) == 0:
        continue
    r1 = np.zeros(T); r1[1:] = c[1:] / c[:-1] - 1.0
    mfe = np.empty(len(idx)); mae = np.empty(len(idx)); reth = np.empty(len(idx))
    F = np.empty((len(idx), len(FEAT)))
    for k, j in enumerate(idx):
        cj = c[j]
        F[k, 0] = c[j] / c[j - 6] - 1.0
        F[k, 1] = c[j] / c[j - 24] - 1.0
        F[k, 2] = c[j] / c[j - 96] - 1.0
        F[k, 3] = c[j] / c[j - 288] - 1.0
        v24 = v[j - 24:j].mean(); v288 = v[j - 288:j].mean()
        F[k, 4] = v24 / v288 if v288 > 0 else 1.0
        F[k, 5] = r1[j - 96:j].std()
        hh = h[j - 96:j].max(); ll = l[j - 96:j].min()
        F[k, 6] = (cj - ll) / (hh - ll) if hh > ll else 0.5
        sr = r1[j - 96:j].sum()
        F[k, 7] = np.abs(r1[j - 96:j]).sum() / abs(sr) if abs(sr) > 1e-9 else 1.0
        amp_s = ((h[j - 96:j] - l[j - 96:j]) / c[j - 96:j]).mean()
        amp_l = ((h[j - W_L:j] - l[j - W_L:j]) / c[j - W_L:j]).mean()
        F[k, 8] = amp_s / amp_l if amp_l > 0 else 1.0
        F[k, 9] = (r1[j - 96:j] > 0).mean()
        mfe[k] = h[j + 1:j + H + 1].max() / cj - 1.0
        mae[k] = l[j + 1:j + H + 1].min() / cj - 1.0
        reth[k] = c[j + H] / cj - 1.0
    df = pd.DataFrame(F, columns=FEAT)
    df.insert(0, 't', ts[idx]); df.insert(1, 'sym', sym)
    df['mfe'] = mfe; df['mae'] = mae; df['reth'] = reth
    rows.append(df)
    del df

P = pd.concat(rows, ignore_index=True).sort_values('t').reset_index(drop=True)
del rows, RAW; gc.collect()
log(f'样本 {len(P)}  币 {P.sym.nunique()}  耗时 {time.time()-t0:.0f}s')
log('')

# 一、普查
log('=' * 100)
log('【一】未来 3 天内 MFE >= +50%（且 MAE >= -30%）的比例')
log('=' * 100)
for lvl in [0.20, 0.30, 0.50, 0.80]:
    ok = (P.mfe >= lvl) & (P.mae >= -STOP)
    log(f'  MFE>={lvl:>4.0%}: {int(ok.sum()):>6} 笔  {ok.mean():>6.2%}  '
        f'MFE中位 {P.mfe[ok].median() if ok.sum() else float("nan"):>7.2%}')
log('')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = np.nan_to_num(P[FEAT].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
cut = int(len(P) * 0.70)
mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
yt = ((P.mfe >= TP) & (P.mae >= -STOP)).astype(np.int8).values
log('=' * 100)
log(f'【二】模型（逻辑回归，标签 MFE>={TP:.0%} 且 MAE>=-{STOP:.0%}）')
log('=' * 100)
log(f'训练 {cut} 正 {yt[:cut].mean():.2%}  验证 {len(P)-cut} 正 {yt[cut:].mean():.2%}')
if yt[:cut].sum() < 50:
    log('正样本 < 50，无法训练。')
else:
    m = LogisticRegression(max_iter=800, C=1.0, class_weight='balanced')
    m.fit((X[:cut] - mu) / sd, yt[:cut])
    pv = m.predict_proba((X[cut:] - mu) / sd)[:, 1]
    auc = roc_auc_score(yt[cut:], pv)
    base = yt[cut:].mean()
    log(f'验证集 AUC = {auc:.4f}  基准 {base:.2%}')
    order = np.argsort(-pv)
    for q in [0.10, 0.05, 0.01]:
        k = max(1, int(len(pv) * q)); sel = order[:k]
        log(f'  取 p 前 {q:>4.0%}: 命中 {yt[cut:][sel].mean():>7.2%}  '
            f'提升 {yt[cut:][sel].mean()/base:>5.1f}x  覆盖正样本 {yt[cut:][sel].sum()/yt[cut:].sum():>6.1%}')
    log('')
    log('单特征 AUC：')
    for i, f in enumerate(FEAT):
        try:
            log(f'  {f:<10} {roc_auc_score(yt[cut:], X[cut:, i]):.4f}')
        except Exception:
            log(f'  {f:<10} --')
    log('')

    # 三、低频回测：每 3 天取 p 最高的 1 个，持有 3 天
    log('=' * 100)
    log('【三】低频回测：每 3 天取 p 最高的 1 币，TP=+50% / 止损 -30% / 持满 3 天')
    log('=' * 100)
    V = P.iloc[cut:].copy().reset_index(drop=True)
    V['p'] = pv
    # 每 3 天一个时间桶，桶内取 p 最高的币
    V['bucket'] = (V['t'] // (STEP * 5 * 60 * 1000)).astype(np.int64)
    B = V.sort_values('p', ascending=False).drop_duplicates('bucket').sort_values('bucket')
    log(f'交易日桶 {len(B)} 个')
    log('')
    log(f'{"p阈值":>8} {"入场笔数":>8} {"单笔均值":>10} {"单笔中位":>10} {"胜率":>8} {"首次达1.5x":>10} {"跑满终值":>10} {"最大回撤":>9}')
    for pu_min in [0.0, 0.3, 0.5, 0.7]:
        sub = B[B.p >= pu_min]
        if len(sub) == 0:
            log(f'{pu_min:>8.1f} {0:>8} {"--":>10}')
            continue
        rr = np.where(sub.mae.values <= -STOP, -STOP,
             np.where(sub.mfe.values >= TP, TP, sub.reth.values)) - FEE
        eq = 1.0; hit = None; peak = 1.0; maxdd = 0.0
        for k, r in enumerate(rr):
            eq *= (1 + r)
            peak = max(peak, eq)
            maxdd = min(maxdd, eq / peak - 1.0)
            if eq >= 1.5 and hit is None:
                hit = k + 1
        log(f'{pu_min:>8.1f} {len(sub):>8} {rr.mean():>9.2%} {np.median(rr):>9.2%} '
            f'{(rr>0).mean():>7.2%} {(str(hit)+"笔") if hit else "未达":>10} {eq:>9.3f}x {maxdd:>8.2%}')
    log('')
    log('=' * 100)
    log('【四】滚动起点：从任一笔开始，向前做 K 笔，中途「摸到过」1.5x 的比例')
    log('=' * 100)
    log('（这是回答「赌徒有没有实现过」的直接量化：摸到过 ≠ 保住）')
    rr0 = B[B.p >= 0.5]
    rr_all = np.where(rr0.mae.values <= -STOP, -STOP,
              np.where(rr0.mfe.values >= TP, TP, rr0.reth.values)) - FEE
    log(f'{"K笔":>6} {"起点数":>8} {"曾在途中摸到1.5x":>16} {"占比":>8} {"跑满K笔>=1.5x":>14} {"K笔终值中位":>12}')
    for K in [5, 10, 20, 30]:
        if len(rr_all) < K:
            continue
        n_hit = 0; n_end = 0; ends = []
        for i in range(len(rr_all) - K + 1):
            path = np.cumprod(1 + rr_all[i:i + K])
            if path.max() >= 1.5:
                n_hit += 1
            ends.append(path[-1])
            if path[-1] >= 1.5:
                n_end += 1
        tot = len(rr_all) - K + 1
        log(f'{K:>6} {tot:>8} {n_hit:>16} {n_hit/tot:>7.1%} {n_end:>13} {np.median(ends):>11.3f}x')
    log('')
    log('诚实边界：')
    log('  1. 零滑点，成本 0.09%；MFE/MAE 先后未知，按先触止损保守。')
    log('  2. 每 3 天一个决策点，同币相邻点仍部分重叠。')
    log('  3. 「首次达 1.5x」存在截断偏差；「跑满终值」是不截断的真实结果。')
    log('  4. 只回测验证段（后 30%），特征严格只用决策点及之前。')

with open(os.path.join(OUT, 'lowfreq_REAL.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {os.path.join(OUT, "lowfreq_REAL.txt")}  TOTAL {time.time()-t0:.0f}s')
