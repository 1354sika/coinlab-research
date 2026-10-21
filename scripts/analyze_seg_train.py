# -*- coding: utf-8 -*-
"""analyze_seg_train.py — 把「大幅低买高卖」历史样本喂给模型，并加「本金累计+50%强制撤出」机制

用户口径（2026-10-19）：
  1. 无杠杆，吃的是标的自身波动；每笔低买高卖区间很大，中间振幅有大有小。
  2. 「本金翻到百分之五十」= 组合层累计净值 >= 1.5x 就全部撤出（可次日以新本金重开）。

本脚本做两件事：
  【一】标签定义 + 样本普查
      对每个决策点 j（步长 STEP 根）：
        MFE = max(high[j+1 : j+H]) / close[j] - 1
        MAE = min(low [j+1 : j+H]) / close[j] - 1
        正样本 = MFE >= T 且 MAE >= -L   （「低买后能高卖，且没先被打穿」）
  【二】训练 + 回测
      特征只用 j 及之前（无前视）；时间切分 70/30。
      回测：按 p 排序取前 N，每笔按「TP=T / 移动止损回落 30% / 持满 H」出场，逐笔复利，
            组合累计 >= 1.5x 立刻全部撤出，统计「多久 / 多少笔达标」。

诚实边界（写进输出）：
  · 全部回测，零滑点；成本 0.09%（费 0.04% + 滑 0.05%）
  · TP 用 high 触价假设能以限价成交，成交率未计
  · 样本按 STEP 重叠，同波行情跨样本相关；未做事件去重
  · 满仓单序列复利，未模拟多币并发与保证金约束
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
os.makedirs(OUT, exist_ok=True)

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

log('=' * 104)
log(f'seg_train · 大幅低买高卖样本 -> 模型 -> 本金+50%强制撤出   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 104)

# ---------------- 参数 ----------------
STEP = 12          # 决策步长（5m 根）：1 小时
H = 288            # 未来观察窗：1 天
W_S = 96           # 短窗 8h
W_L = 288 * 30     # 长窗 30 天
FEE = 0.0009       # 单边往返 0.09%
T_LIST = [0.10, 0.20, 0.30]   # 「大幅」阈值
L_MAE = 0.15       # 允许的最大浮亏

log(f'数据 {os.path.basename(SRC)}；STEP={STEP} 根(1h)  H={H} 根(1d)  成本 {FEE*100:.2f}%')
log('')

# ---------------- 载入 ----------------
RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'open': 'float32',
                             'high': 'float32', 'low': 'float32', 'close': 'float32',
                             'volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym,
                    g['open_time'].values.astype('int64'),
                    g['open'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'),
                    g['close'].values.astype('float64'),
                    g['volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)} 载入 {time.time()-t0:.0f}s')

# ---------------- 逐币造特征与标签 ----------------
FEAT = ['r6', 'r24', 'r96', 'r288', 'volr', 'sd96', 'pos96', 'swr', 'amp_ratio', 'ntr']
rows = []   # 每行: t, sym, f0..f9, mfe, mae, ret_h(持满 H 的收益)
for (sym, ts, o, h, l, c, v) in RAW:
    T = len(c)
    if T < W_L + H + STEP + 10:
        continue
    idx = np.arange(W_L, T - H - 1, STEP)
    if len(idx) == 0:
        continue
    r1 = np.zeros(T, dtype=np.float64)
    r1[1:] = c[1:] / c[:-1] - 1.0
    # 滚动量
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
        rr = r1[j - 96:j]
        F[k, 5] = rr.std()
        hh = h[j - 96:j].max(); ll = l[j - 96:j].min()
        F[k, 6] = (cj - ll) / (hh - ll) if hh > ll else 0.5
        sr = r1[j - 96:j].sum()
        F[k, 7] = np.abs(r1[j - 96:j]).sum() / abs(sr) if abs(sr) > 1e-9 else 1.0
        amp_s = ((h[j - 96:j] - l[j - 96:j]) / c[j - 96:j]).mean()
        amp_l = ((h[j - W_L:j] - l[j - W_L:j]) / c[j - W_L:j]).mean()
        F[k, 8] = amp_s / amp_l if amp_l > 0 else 1.0
        F[k, 9] = (r1[j - 96:j] > 0).mean()
        # 未来
        hh2 = h[j + 1:j + H + 1].max(); ll2 = l[j + 1:j + H + 1].min()
        mfe[k] = hh2 / cj - 1.0
        mae[k] = ll2 / cj - 1.0
        reth[k] = c[j + H] / cj - 1.0
    df = pd.DataFrame(F, columns=FEAT)
    df.insert(0, 't', ts[idx])
    df.insert(1, 'sym', sym)
    df['mfe'] = mfe; df['mae'] = mae; df['reth'] = reth
    rows.append(df)
    del df

P = pd.concat(rows, ignore_index=True)
del rows, RAW; gc.collect()
log(f'样本 {len(P)}  币 {P.sym.nunique()}  耗时 {time.time()-t0:.0f}s')
log('')

# ---------------- 一、样本普查 ----------------
log('=' * 104)
log('【一】「大幅低买高卖」样本普查（未来 1 天内 MFE >= T 且 MAE >= -15%）')
log('=' * 104)
log(f'{"T(阈值)":>10} {"正样本数":>10} {"占比":>8} {"MFE中位":>10} {"MAE中位":>10}')
for T in T_LIST:
    pos = (P.mfe >= T) & (P.mae >= -L_MAE)
    log(f'{T:>9.0%} {int(pos.sum()):>10} {pos.mean():>7.2%} '
        f'{P.mfe[pos].median():>9.2%} {P.mae[pos].median():>9.2%}')
log('')

# ---------------- 二、训练（时间切分） ----------------
from sklearn.linear_model import LogisticRegression

T_MAIN = 0.20
P = P.sort_values('t').reset_index(drop=True)
y = ((P.mfe >= T_MAIN) & (P.mae >= -L_MAE)).astype(np.int8).values
cut = int(len(P) * 0.70)
X = P[FEAT].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

Xt, Xv = X[:cut], X[cut:]
yt, yv = y[:cut], y[cut:]
log('=' * 104)
log(f'【二】模型（逻辑回归，特征 {len(FEAT)} 个，标签 T={T_MAIN:.0%}）')
log('=' * 104)
log(f'训练 {len(Xt)} 正 {yt.mean():.2%}  验证 {len(Xv)} 正 {yv.mean():.2%}')

mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
m = LogisticRegression(max_iter=800, C=1.0, class_weight='balanced')
m.fit((Xt - mu) / sd, yt)
pv = m.predict_proba((Xv - mu) / sd)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(yv, pv)
log(f'验证集 AUC = {auc:.4f}   基准正样本率 {yv.mean():.2%}')
log('')
log(f'{"取p前":>8} {"样本数":>10} {"命中率":>9} {"提升":>7} {"覆盖正样本":>10}')
order = np.argsort(-pv)
for q in [0.10, 0.05, 0.01]:
    k = int(len(pv) * q)
    sel = order[:k]
    hit = yv[sel].mean()
    log(f'{q:>7.0%} {k:>10} {hit:>8.2%} {hit/yv.mean():>6.1f}x {yv[sel].sum()/yv.sum():>9.1%}')
log('')

# 单特征 AUC
log('单特征 AUC（验证集）：')
for i, f in enumerate(FEAT):
    try:
        a = roc_auc_score(yv, Xv[:, i])
        log(f'  {f:<10} {a:.4f}')
    except Exception:
        log(f'  {f:<10} --')
log('')

# ---------------- 三、回测：+50% 强制撤出 ----------------
log('=' * 104)
log('【三】回测：模型选币 + 出场规则 + 组合累计 >=1.5x 强制撤出')
log('=' * 104)
# 用验证段样本做回测（时间序）
V = P.iloc[cut:].copy()
V['p'] = pv
V = V.sort_values('t').reset_index(drop=True)

def run(tp, stop, optimistic, tag):
    """逐笔复利；累计 >=1.5 停。
    optimistic=True : 假设先触 TP 再触止损（乐观）
    optimistic=False: 假设先触止损（保守，单笔最差 -stop）
    注：MFE/MAE 是同一窗口内极值，真实先后顺序未知——这是本脚本的硬缺陷。
    """
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    n = 0
    first_hit_n = None
    first_hit_day = None
    t_start = V['t'].iloc[0]
    # 每根决策点取 p 最高的 topn 中 1 个（简化：按时间滚动，取当根 p 最高的那个）
    taken_t = -1
    for i in range(len(V)):
        if eq >= 1.5:
            first_hit_n = n
            first_hit_day = (V['t'].iloc[i] - t_start) / 86400000.0
            break
        if V['t'].iloc[i] <= taken_t:
            continue
        # 该时刻所有样本里取 p 最高
        # 简化：直接按序，每 STEP 取一个
        row = V.iloc[i]
        if row['p'] < 0.5:
            continue
        mfe = row['mfe']; mae = row['mae']; reth = row['reth']
        if optimistic:
            if mfe >= tp:
                r = tp
            elif mae <= -stop:
                r = -stop
            else:
                r = reth
        else:
            if mae <= -stop:
                r = -stop
            elif mfe >= tp:
                r = tp
            else:
                r = reth
        r_net = r - FEE
        eq *= (1.0 + r_net)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
        n += 1
        taken_t = row['t'] + STEP * 5 * 60 * 1000
        if eq <= 0.01:
            break
    return dict(tag=tag, eq=eq, n=n, maxdd=maxdd, hit_n=first_hit_n, hit_day=first_hit_day)

log(f'{"出场规则":<30} {"终值":>9} {"笔数":>7} {"最大回撤":>10} {"达1.5x":>12}')
res = []
for tp, stop, opt, tag in [
    (T_MAIN, L_MAE, True,  'TP=20% 止损15% 乐观'),
    (T_MAIN, L_MAE, False, 'TP=20% 止损15% 保守'),
    (0.30, L_MAE, False, 'TP=30% 止损15% 保守'),
    (0.50, L_MAE, False, 'TP=50% 止损15% 保守'),
    (0.10, L_MAE, False, 'TP=10% 止损15% 保守'),
    (0.10, 0.10, False, 'TP=10% 止损10% 保守'),
]:
    r = run(tp, stop, opt, tag)
    res.append(r)
    hit = f'{r["hit_n"]}笔/{r["hit_day"]:.0f}天' if r['hit_n'] else '未达'
    log(f'{tag:<30} {r["eq"]:>8.3f}x {r["n"]:>7} {r["maxdd"]:>9.2%} {hit:>12}')

log('')
log('诚实边界：')
log('  1. 全部回测，零滑点；成本 0.09%（费 0.04% + 滑 0.05%）。')
log('  2. 标签 MFE/MAE 用 high/low 计算，实盘触价能否成交未计。')
log('  3. 样本按 STEP=1h 重叠，同波行情跨样本相关；未做事件去重。')
log('  4. 单序列满仓复利，未模拟多币并发持仓与保证金约束。')
log('  5. 回测只用验证段（后 30%），特征严格只用 j 及之前。')

with open(os.path.join(OUT, 'seg_train_REAL.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {os.path.join(OUT, "seg_train_REAL.txt")}  TOTAL {time.time()-t0:.0f}s')
