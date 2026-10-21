# -*- coding: utf-8 -*-
"""analyze_portfolio.py — 分散能不能把单币 -96% 摊成组合 -20%？

上一轮结论：单序列（一次只押一个币）满仓复利，任何过滤都救不了，回撤 -65%~-96%。
假设：同时押 N 个币、等权，把单币波动摊平，回撤应显著下降。

本脚本：
  每天开一批，选 p_up 最高的 N 个币，等权持有 1 天；
  每币出场 TP=20% / 止损 15% / 持满 1 天；
  组合累计 >= 1.5x 立即全部撤出，统计天数与最大回撤。
  对照 N = 1 / 5 / 10 / 20 / 40。

诚实边界：零滑点、成本 0.09%、MFE/MAE 先后顺序未知（按先触止损保守）、
          样本按 1h 重叠、只回测验证段（后 30%）、未模拟逐仓保证金。
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
log(f'portfolio · 分散 N 币等权 vs 单币满仓   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 100)

STEP, H, W_L, FEE = 12, 288, 288 * 30, 0.0009
TP, STOP = 0.20, 0.15

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

from sklearn.linear_model import LogisticRegression
X = np.nan_to_num(P[FEAT].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
cut = int(len(P) * 0.70)
mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
yt = ((P.mfe >= 0.20) & (P.mae >= -0.15)).astype(np.int8).values
m = LogisticRegression(max_iter=800, C=1.0, class_weight='balanced')
m.fit((X[:cut] - mu) / sd, yt[:cut])
V = P.iloc[cut:].copy().reset_index(drop=True)
V['p_up'] = m.predict_proba((X[cut:] - mu) / sd)[:, 1]
log(f'验证段 {len(V)} 样本，p_up 均值 {V.p_up.mean():.3f}')
log('')

# 每天每币只保留 p_up 最高的一条
V['day'] = V['t'] // 86400000
G = V.sort_values('p_up', ascending=False).drop_duplicates(['day', 'sym'])
G = G.sort_values(['day', 'p_up'], ascending=[True, False]).reset_index(drop=True)
days = np.sort(G['day'].unique())
log(f'验证段交易日 {len(days)} 天，每日可用币中位 {G.groupby("day").size().median():.0f}')
log('')

def bt(N, pu_min, tag):
    eq, peak, maxdd, nd = 1.0, 1.0, 0.0, 0
    hit_day = None
    d0 = days[0]
    for d in days:
        sub = G[G['day'] == d]
        sub = sub[sub.p_up >= pu_min].head(N)
        if len(sub) == 0:
            nd += 1; continue
        rr = np.where(sub.mae.values <= -STOP, -STOP,
             np.where(sub.mfe.values >= TP, TP, sub.reth.values)) - FEE
        eq *= (1.0 + float(rr.mean()))
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
        nd += 1
        if eq >= 1.5 and hit_day is None:
            hit_day = int(d - d0)
            break
    return dict(tag=tag, eq=eq, days=nd, maxdd=maxdd, hit_day=hit_day)

log('=' * 100)
log('【一】分散 N 币等权（每日建仓，持有 1 天，TP20%/止损15%，累计 1.5x 停）')
log('=' * 100)
log(f'{"方案":<34} {"终值":>9} {"交易日":>8} {"最大回撤":>10} {"达1.5x":>12}')
res = []
for N in [1, 5, 10, 20, 40]:
    for pu in [0.5, 0.7]:
        r = bt(N, pu, f'N={N:<3} p_up>={pu}')
        res.append(r)
        hit = f'第{r["hit_day"]}天' if r['hit_day'] is not None else '未达'
        log(f'{r["tag"]:<34} {r["eq"]:>8.3f}x {r["days"]:>8} {r["maxdd"]:>9.2%} {hit:>12}')
log('')

# 对照：不开模型，随机选 N 币
log('=' * 100)
log('【二】对照：每天随机选 N 币（不开模型）')
log('=' * 100)
rng = np.random.default_rng(42)
log(f'{"方案":<34} {"终值":>9} {"交易日":>8} {"最大回撤":>10} {"达1.5x":>12}')
for N in [1, 5, 20]:
    eq, peak, maxdd, nd, hit_day = 1.0, 1.0, 0.0, 0, None
    d0 = days[0]
    for d in days:
        sub = G[G['day'] == d]
        if len(sub) == 0:
            nd += 1; continue
        take = sub.sample(min(N, len(sub)), random_state=int(rng.integers(1e9)))
        rr = np.where(take.mae.values <= -STOP, -STOP,
             np.where(take.mfe.values >= TP, TP, take.reth.values)) - FEE
        eq *= (1.0 + float(rr.mean())); peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0); nd += 1
        if eq >= 1.5 and hit_day is None:
            hit_day = int(d - d0); break
    hit = f'第{hit_day}天' if hit_day is not None else '未达'
    log(f'{"随机 N=" + str(N):<34} {eq:>8.3f}x {nd:>8} {maxdd:>9.2%} {hit:>12}')
log('')
log('诚实边界：')
log('  1. 全部回测，零滑点，成本 0.09%。')
log('  2. MFE/MAE 同窗极值，先后顺序未知；回测按「先触止损」保守处理。')
log('  3. 每日一批、等权、满仓复利；未模拟逐仓保证金与资金费率。')
log('  4. 样本按 1h 重叠，同波行情跨样本相关；未做事件去重。')
log('  5. 只回测验证段（后 30%），特征严格只用决策点及之前。')
log('  6. 「同时持仓 N 个」= 每日再平衡一次，非连续调仓。')

with open(os.path.join(OUT, 'portfolio_REAL.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {os.path.join(OUT, "portfolio_REAL.txt")}  TOTAL {time.time()-t0:.0f}s')
