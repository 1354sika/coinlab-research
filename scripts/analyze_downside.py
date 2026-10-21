# -*- coding: utf-8 -*-
"""analyze_downside.py — 模型看得出「会先跌穿」吗？

上一轮结论：可交易段 + 强制 +50% 撤出能到 1.5x，但途中回撤 -96%。
问题指向入场端：能不能提前认出「会先打穿止损」的样本？

本脚本：
  【一】训练「会跌」分类器：标签 MAE <= -5% / -10% / -15%（未来 1 天内最大浮亏）
  【二】分层：按 p_down 分 5 档，看各档的 MAE 中位、MFE 中位、打穿比例
  【三】联合回测：入场 = p_up 高 且 p_down 低；看回撤能压到多少

诚实边界：零滑点、成本 0.09%、样本按 1h 重叠未去重、MFE/MAE 先后顺序未知。
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
log(f'downside · 模型看得出会跌吗   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 100)

STEP, H, W_L, FEE = 12, 288, 288 * 30, 0.0009

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'high': 'float32',
                             'low': 'float32', 'close': 'float32', 'volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = np.nan_to_num(P[FEAT].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
cut = int(len(P) * 0.70)
Trim, Tval = slice(0, cut), slice(cut, len(P))
mu, sd = X[Trim].mean(0), X[Trim].std(0) + 1e-9
Xt = (X[Trim] - mu) / sd
Xv = (X[Tval] - mu) / sd
V = P.iloc[cut:].reset_index(drop=True)

log('=' * 100)
log('【一】「会跌」分类器：预测未来 1 天内 MAE 是否击穿阈值')
log('=' * 100)
log(f'{"阈值":>8} {"训练正率":>9} {"验证正率":>9} {"AUC":>8} {"前1%命中":>9} {"前5%命中":>9} {"前10%命中":>10}')
probs = {}
for lvl in [0.05, 0.10, 0.15, 0.20]:
    yd = (P.mae <= -lvl).astype(np.int8).values
    m = LogisticRegression(max_iter=800, C=1.0, class_weight='balanced')
    m.fit(Xt, yd[Trim])
    pv = m.predict_proba(Xv)[:, 1]
    probs[lvl] = pv
    auc = roc_auc_score(yd[Tval], pv)
    order = np.argsort(-pv); base = yd[Tval].mean()
    h1 = yd[Tval][order[:max(1, len(pv)//100)]].mean()
    h5 = yd[Tval][order[:max(1, len(pv)//20)]].mean()
    h10 = yd[Tval][order[:max(1, len(pv)//10)]].mean()
    log(f'{-lvl:>7.0%} {yd[Trim].mean():>8.2%} {base:>8.2%} {auc:>7.4f} '
        f'{h1:>8.2%} {h5:>8.2%} {h10:>9.2%}')
log('')

# 「会涨」模型（对照）
yu = ((P.mfe >= 0.20) & (P.mae >= -0.15)).astype(np.int8).values
mu2 = LogisticRegression(max_iter=800, C=1.0, class_weight='balanced')
mu2.fit(Xt, yu[Trim])
p_up = mu2.predict_proba(Xv)[:, 1]
log(f'对照「会涨」模型 AUC = {roc_auc_score(yu[Tval], p_up):.4f}（基准 {yu[Tval].mean():.2%}）')
log('')

# ---------------- 二、分层 ----------------
log('=' * 100)
log('【二】按 p_down(MAE<=-10%) 分 5 档：跌的风险真的被分开了吗')
log('=' * 100)
V['p_down'] = probs[0.10]
V['p_up'] = p_up
V['f_down'] = pd.qcut(V.p_down, 5, labels=False, duplicates='drop')
log(f'{"档":>4} {"样本":>8} {"p_down均值":>11} {"MAE中位":>9} {"MAE p10":>9} {"MFE中位":>9} {"击穿10%比例":>11}')
for g, sub in V.groupby('f_down'):
    log(f'{int(g):>4} {len(sub):>8} {sub.p_down.mean():>10.3f} {sub.mae.median():>8.2%} '
        f'{sub.mae.quantile(0.10):>8.2%} {sub.mfe.median():>8.2%} {(sub.mae <= -0.10).mean():>10.2%}')
log('')

# ---------------- 三、联合过滤回测 ----------------
log('=' * 100)
log('【三】联合过滤：p_up 高 且 p_down 低 —— 回撤能压到多少')
log('=' * 100)
V = V.sort_values('t').reset_index(drop=True)

def bt(pu_thr, pd_thr, tp, stop, label):
    eq, peak, maxdd, n, hit_n, hit_day = 1.0, 1.0, 0.0, 0, None, None
    t_start = V['t'].iloc[0]; taken = -1
    for i in range(len(V)):
        if eq >= 1.5:
            hit_n = n; hit_day = (V['t'].iloc[i] - t_start) / 86400000.0; break
        r = V.iloc[i]
        if r['t'] <= taken: continue
        if r['p_up'] < pu_thr or r['p_down'] > pd_thr: continue
        if r['mae'] <= -stop: rr = -stop
        elif r['mfe'] >= tp: rr = tp
        else: rr = r['reth']
        rr = rr - FEE
        eq *= (1 + rr); peak = max(peak, eq); maxdd = min(maxdd, eq / peak - 1.0)
        n += 1; taken = r['t'] + STEP * 5 * 60 * 1000
        if eq <= 0.01: break
    hit = f'{hit_n}笔/{hit_day:.0f}天' if hit_n else '未达'
    log(f'{label:<40} {eq:>8.3f}x {n:>6} {maxdd:>9.2%} {hit:>12}')

log(f'{"规则":<40} {"终值":>8} {"笔数":>6} {"最大回撤":>9} {"达1.5x":>12}')
bt(0.50, 1.01, 0.20, 0.15, '基线：仅 p_up>0.5（不滤跌）')
bt(0.50, 0.50, 0.20, 0.15, 'p_up>0.5 且 p_down<0.5')
bt(0.50, 0.30, 0.20, 0.15, 'p_up>0.5 且 p_down<0.3')
bt(0.60, 0.30, 0.20, 0.15, 'p_up>0.6 且 p_down<0.3')
bt(0.70, 0.20, 0.20, 0.15, 'p_up>0.7 且 p_down<0.2')
bt(0.70, 0.20, 0.30, 0.15, 'p_up>0.7 且 p_down<0.2 / TP30')
log('')
log('诚实边界：')
log('  1. 全部回测，零滑点，成本 0.09%。')
log('  2. MFE/MAE 为同窗极值，先后顺序未知；回测按「先触止损」保守处理。')
log('  3. 样本按 STEP=1h 重叠，未做事件去重。')
log('  4. 回测只用验证段（后 30%）。')
log('  5. 未触 TP 也未触止损的笔按「持满 1 天的实际收益 reth」结算。')

with open(os.path.join(OUT, 'downside_REAL.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {os.path.join(OUT, "downside_REAL.txt")}  TOTAL {time.time()-t0:.0f}s')
