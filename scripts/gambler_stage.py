# -*- coding: utf-8 -*-
"""gambler_stage.py — Q1：局内「还能走多远」+「当前在哪个位置」

用户判定（2026-10-17）：这是最重要的方向。
    「识别局还能走多远」+「当前在局的哪个位置」必须列为重要方向。

为什么单独做：
    已有能力：判断「这个币现在是否在局里」 AUC≈0.98（几乎完美）
    却卡在：分不出「刚开始的小局 / 已涨完的大局」→ 三分类真档召回仅 4%
    ⇒ 缺的不是"识别局"，是"给局定阶段"。

【本脚本问的两个问题】
    Q1a  还在局里（且未崩完）的时候，**未来 48h 从当前价还能涨多少**？ → 回归
    Q1b  同样的点，**未来 48h 是否能再拿到 ≥+30% 的出场机会**？ → 二分类（用户口径）

【样本定义（全程只用 t 时刻可观测信息）】
    局内 = 当前价相对过去 288 根(1天)最低点已涨 >= 15%（已启动）
           且 当前价相对过去 288 根最高点回落 < 50%（还没崩完）

【阶段特征（全部 t 时刻可观测，无前视）】
    over_now      已涨幅度 c/lo288-1
    bars_since_lo 距 288 根最低点多少根（局龄）
    rise_speed    已涨幅度 / 局龄（平均每根涨多少）
    vol_decay     近 6 根均量比 / 近 96 根均量比（量能衰竭度）
    pullback      自 288 根最高点回落幅度
    dist_hi288    距 288 根最高点的距离 c/hi288-1
    new_hi_frac   近 12 根里创新高的比例
    flat_bars     连续未创新高的根数
    vol_z24       近 24 根量比的 z 分（相对 30 天）
    ret_accel     近 6 根涨幅 - 前 6 根涨幅（加速度）
    pos288        (c-lo288)/(hi288-lo288) 位置分位

【标签】
    y_fwd    未来 48h 内最高价 / 当前价 - 1                      （连续）
    y_hit30  y_fwd >= 0.30                                     （二分类，用户口径）

【纪律】
    训练=前 70% 时间，测试=后 30%；只报测试段；
    分档表按 y_fwd 五档给出，看"哪些阶段还能拿到 30%"。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288
T_H = 48                     # 未来多少小时
TARGET = 0.30                # 用户口径：+30%
BAR_PER_H = 12               # 5m
_WIN = int(T_H * BAR_PER_H)  # 576 根

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 110)
log('gambler_stage · Q1 局内「还能走多远」+「当前在哪个位置」')
log('=' * 110)

# ---------- 1. 读数据 ----------
RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32',
                             'quote_volume': 'float32', 'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns)
SYM2CI = {s: i for i, s in enumerate(syms)}
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')
t_cut = int(T * 0.70)
log(f'训练段 <= 第 {t_cut} 根   测试段 > 第 {t_cut} 根')

FEAT_NAMES = ['over_now', 'bars_since_lo', 'rise_speed', 'vol_decay', 'pullback',
              'dist_hi288', 'new_hi_frac', 'flat_bars', 'vol_z24', 'ret_accel', 'pos288']
XA, YA, TA = [], [], []   # 特征 / y_fwd / 时间位置

for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < WDAY * 4:
        continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over_now = c / np.maximum(lo288, 1e-12) - 1.0
        dist_hi288 = c / np.maximum(hi288, 1e-12) - 1.0
        pos288 = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)

    # 局龄：距 288 内最低点的根数
    # 用 argmin 滚动：逐点回看 288 根找最低点索引（O(n*288) 对 315k*60 太慢）
    # 改为：用 pandas rolling apply 也慢。改用近似——记录"最近一次刷新 288 低点"的位置
    # 更准确且 O(n)：用 numpy 的滑动窗口 argmin（用 minimum_filter 不行）
    # 折中：先算 lo288 序列，再用差分检测"新低出现"的位置，之后累计根数
    bars_since_lo = np.full(n, np.nan)
    last_lo_idx = -1
    for i in range(WDAY, n):
        if np.isnan(lo288[i]):
            continue
        # lo288[i] 是 i-287..i 的最小值；若 c[i] 就是该窗口最小值附近（差 < 1e-12 或接近）
        if last_lo_idx < i - WDAY + 1:
            # 需要重找：用 argmin
            w = l[i - WDAY + 1:i + 1]
            if len(w):
                last_lo_idx = i - WDAY + 1 + int(np.argmin(w))
        if l[i] <= lo288[i] * 1.0000001:
            last_lo_idx = i
        bars_since_lo[i] = i - last_lo_idx

    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        vol_decay = volr6 / np.maximum(volr96, 1e-12)

    # 新高统计
    hi_roll12 = hs.rolling(12, min_periods=6).max().values
    is_new_hi = (h >= hi_roll12 * 0.999999).astype('float64')
    new_hi_frac = pd.Series(is_new_hi).rolling(12, min_periods=6).mean().values
    flat = np.zeros(n); k = 0
    for i in range(1, n):
        k = 0 if is_new_hi[i] > 0 else k + 1
        flat[i] = k

    # 量比 z 分（近 24 根量比相对过去 30 天）
    vm = pd.Series(volr).rolling(WDAY * 30, min_periods=WDAY).mean().values
    vs = pd.Series(volr).rolling(WDAY * 30, min_periods=WDAY).std().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        vol_z24 = (volr24 - vm) / np.maximum(vs, 1e-12)

    r6 = np.full(n, np.nan); r6[6:] = c[6:] / c[:-6] - 1.0
    with np.errstate(invalid='ignore'):
        ret_accel = r6 - np.concatenate([np.full(6, np.nan), r6[:-6]])
    with np.errstate(invalid='ignore', divide='ignore'):
        rise_speed = over_now / np.maximum(bars_since_lo, 1.0)

    # 未来 48h 最高 / 当前价 - 1
    futmax = pd.Series(h).rolling(_WIN, min_periods=1).max().shift(-_WIN + 1).values
    with np.errstate(invalid='ignore', divide='ignore'):
        y_fwd = futmax / np.maximum(c, 1e-12) - 1.0

    # 局内样本：已涨 >=15% 且 自峰值回落 <50%
    m = (over_now >= 0.15) & (dist_hi288 > -0.50) & np.isfinite(y_fwd) & (bars_since_lo > 0)

    cols = [over_now, bars_since_lo, rise_speed, vol_decay, (1 + dist_hi288),
            dist_hi288, new_hi_frac, flat, vol_z24, ret_accel, pos288]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])
    Mm = M[m]
    if len(Mm) == 0:
        continue
    posv = np.searchsorted(ts, ot)[m]
    XA.append(Mm.astype('float32'))
    YA.append(y_fwd[m].astype('float32'))
    TA.append(posv.astype('int64'))

del RAW, C; gc.collect()

if not XA:
    log('无样本，退出'); raise SystemExit(1)

X = np.vstack(XA); y = np.concatenate(YA); tidx = np.concatenate(TA)
del XA, YA, TA; gc.collect()
log(f'局内样本 {len(y):,}   基准 y_fwd 中位 {np.median(y)*100:.2f}%   P(y>=30%) {100*(y>=TARGET).mean():.2f}%')

tr = tidx < t_cut; te = ~tr
log(f'训练 {tr.sum():,}   测试 {te.sum():,}   测试段 P(y>=30%) {100*(y[te]>=TARGET).mean():.3f}%')

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, r2_score

# ---------- Q1a 回归：还能涨多少 ----------
reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=6,
                                    min_samples_leaf=100, random_state=7)
reg.fit(X[tr], np.clip(y[tr], -0.2, 3.0))
p = reg.predict(X[te])
log('\n' + '=' * 110)
log('【Q1a】回归：未来 48h 还能涨多少（测试段）')
log('=' * 110)
log(f'  R² = {r2_score(y[te], p):.4f}   预测中位 {np.median(p)*100:.2f}%   实际中位 {np.median(y[te])*100:.2f}%')
# 预测分 10 档，看每档实际还能涨多少
qs = np.percentile(p, np.arange(0, 101, 10))
log(f'  {"预测档":<16}{"样本":>9}{"预测中位%":>11}{"实际中位%":>11}{"实际均值%":>11}{"P(>=30%)":>10}')
for i in range(10):
    m = (p >= qs[i]) & (p <= qs[i+1] if i == 9 else p < qs[i+1])
    if m.sum() == 0:
        continue
    log(f'  P{int(qs[i]*1000)/10:>5.1f}~P{int(qs[i+1]*1000)/10:<6.1f}{m.sum():>9,}'
        f'{np.median(p[m])*100:>11.2f}{np.median(y[te][m])*100:>11.2f}'
        f'{y[te][m].mean()*100:>11.2f}{100*(y[te][m]>=TARGET).mean():>10.2f}')

# ---------- Q1b 分类：还能不能拿到 30% ----------
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                     min_samples_leaf=100, random_state=7)
clf.fit(X[tr], (y[tr] >= TARGET).astype('int8'))
pc = clf.predict_proba(X[te])[:, 1]
log('\n' + '=' * 110)
log('【Q1b】分类：还能不能再拿到 ≥+30% 出场机会（测试段）')
log('=' * 110)
log(f'  AUC = {roc_auc_score((y[te] >= TARGET).astype("int8"), pc):.4f}   测试基准 {100*(y[te]>=TARGET).mean():.3f}%')
log(f'  {"阈值":>6}{"报出":>10}{"覆盖率":>9}{"精确率":>9}{"召回":>8}')
for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    m = pc >= thr
    if m.sum() == 0:
        log(f'  {thr:>6.2f}{"无报出":>10}'); continue
    prec = (y[te][m] >= TARGET).mean()
    rec = m[(y[te] >= TARGET)].sum() / max((y[te] >= TARGET).sum(), 1)
    log(f'  {thr:>6.2f}{m.sum():>10,}{100*m.mean():>8.2f}%{100*prec:>8.2f}%{100*rec:>7.2f}%')

# ---------- Q1c 阶段特征单变量：当前"位置"与剩余空间 ----------
log('\n' + '=' * 110)
log('【Q1c】阶段特征单变量：当前在哪个位置 ⇒ 还剩多少空间（训练段相关 + 测试段分档）')
log('=' * 110)
log(f'  {"特征":<16}{"训练段corr":>12}')
for j, nm in enumerate(FEAT_NAMES):
    cc = np.corrcoef(X[tr, j], np.clip(y[tr], -0.2, 3.0))[0, 1]
    log(f'  {nm:<16}{cc:>+12.4f}')

log('\n  关键：按 over_now（当前已涨%）分档看剩余空间（测试段）')
log(f'  {"已涨档":<16}{"样本":>9}{"未来48h中位%":>14}{"P(>=30%)":>10}')
ov = X[:, 0]
for a, b in [(0.15, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 1.0), (1.0, 2.0), (2.0, 99)]:
    m = te & (ov >= a) & (ov < b)
    if m.sum() == 0:
        continue
    log(f'  {a*100:>4.0f}~{b*100 if b<99 else 999:>4.0f}%{"":<6}{m.sum():>9,}'
        f'{np.median(y[m])*100:>14.2f}{100*(y[m]>=TARGET).mean():>10.2f}')

log('\n  按 bars_since_lo（局龄，根）分档（测试段，1 根=5m）')
bs = X[:, 1]
for a, b in [(0, 12), (12, 48), (48, 144), (144, 288), (288, 99999)]:
    m = te & (bs >= a) & (bs < b)
    if m.sum() == 0:
        continue
    log(f'  {a:>5}~{b if b<99999 else 99999:>6}根{"":<6}{m.sum():>9,}'
        f'{np.median(y[m])*100:>14.2f}{100*(y[m]>=TARGET).mean():>10.2f}')

out = os.path.join(OUT, 'stage_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
