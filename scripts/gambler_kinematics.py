# -*- coding: utf-8 -*-
"""gambler_kinematics.py — 用「微分」看周期：斜率(速度)、二阶导(加速度)、燃料积分

用户判据（2026-10-17）：
    「我的斜率的意思是，我认为我们需要引入高等数学的角度观察和思考……
      我们之前用了蝴蝶效应、概率行为心理侧写等很多手段，
      但是我们漏了高等数学的角度，也没有用高等数学的角度，把已验证的结论融合。」

本脚本只做一件事，但它是"认定前"能不能识别的关键检验：

    对每一个周期，取**认定点(+30%)之前**的那段价格轨迹，算：
        v(t) = 每根 5m 的对数收益            ← 一阶导（速度/斜率）
        a(t) = v(t) - v(t-1)                 ← 二阶导（加速度）
        jerk = a(t) - a(t-1)                 ← 三阶导（急动度）
        燃料 = Σ(成交额) / 涨幅              ← 积分：涨 1 个点要烧多少钱
        拐点 = a 由正转负的次数               ← "斜率开始下降"的次数
    然后问：**这些导数特征，能不能在认定之前就把"大周期"和"小周期"分开？**

分组（按周期最终涨幅，事后才知道，用于检验可分辨性）：
    BIG  : gain >= 100%（翻倍以上）
    MID  : 50% <= gain < 100%
    SML  : 30% <= gain <  50%（刚够认定的"小周期"）

判据：BIG vs SML 在**认定前**的 AUC；若显著 >0.5 ⇒ 认定前存在可用的数学信号。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
BAR_MIN = 5
MAX_BARS = int(48 * 60 / BAR_MIN)
TRIG = 0.30
R = 0.10          # 周期结束的回落阈值（四档里居中、认定后见顶占比 16.4%）

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_kinematics · 用导数看周期：认定之前能不能分出大小')
log('=' * 116)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'读入 {len(RAW)} 币   耗时 {time.time()-t0:.0f}s')

rows = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < 600:
        continue
    lg = np.log(np.maximum(c, 1e-12))
    i = 0
    while i < n - 1:
        lo_idx = i; lo = l[i]
        conf = -1; peak = -np.inf; peak_i = -1
        j = i; done = False
        while j < n - 1:
            j += 1
            if conf < 0:
                if l[j] < lo:
                    lo = l[j]; lo_idx = j
                if j - lo_idx > MAX_BARS:
                    i = lo_idx + 1; done = True; break
                if h[j] >= lo * (1 + TRIG):
                    conf = j; peak = h[j]; peak_i = j
            else:
                if h[j] > peak:
                    peak = h[j]; peak_i = j
                if j - conf > MAX_BARS or l[j] <= peak * (1 - R):
                    gain = peak / lo - 1
                    # ---- 认定前的轨迹 [lo_idx, conf] ----
                    if conf - lo_idx >= 12:
                        seg = lg[lo_idx:conf + 1]
                        v = np.diff(seg)                       # 速度
                        a = np.diff(v)                         # 加速度
                        jk = np.diff(a)                        # 急动度
                        vol_seg = q[lo_idx:conf + 1]
                        dp = seg[-1] - seg[0]
                        fuel = float(np.sum(vol_seg) / max(dp, 1e-9)) if dp > 0 else np.nan
                        # 拐点：a 由正转负
                        turn = int(np.sum((a[:-1] > 0) & (a[1:] <= 0)))
                        m = len(v)
                        pos_vmax = float(np.argmax(v) / max(m - 1, 1))
                        rows.append(dict(
                            sym=sym, gain=gain, bars=m,
                            conf_h=(conf - lo_idx) * BAR_MIN / 60,
                            v_mean=float(np.mean(v)), v_std=float(np.std(v)),
                            v_max=float(np.max(v)), v_last=float(np.mean(v[-6:])),
                            v_max_pos=pos_vmax,
                            a_mean=float(np.mean(a)), a_std=float(np.std(a)),
                            a_max=float(np.max(a)), a_min=float(np.min(a)),
                            a_last=float(np.mean(a[-6:])),
                            jk_std=float(np.std(jk)),
                            turn=turn, turn_rate=turn / max(m, 1),
                            fuel=fuel,
                            log_fuel=float(np.log1p(max(fuel, 0))) if np.isfinite(fuel) else np.nan,
                            sum_v=float(np.sum(v)),
                        ))
                    i = j + 1; done = True; break
        if done:
            continue
        break
del RAW; gc.collect()

DF = pd.DataFrame(rows)
DF.to_csv(os.path.join(OUT, 'kinematics.csv'), index=False, encoding='utf-8-sig')
DF = DF[np.isfinite(DF['gain'])].copy()
DF['grp'] = np.where(DF.gain >= 1.0, 'BIG',
              np.where(DF.gain >= 0.5, 'MID', 'SML'))
log(f'周期 {len(DF):,}（R={R:.0%}）   BIG {int((DF.grp=="BIG").sum())} / '
    f'MID {int((DF.grp=="MID").sum())} / SML {int((DF.grp=="SML").sum())}   耗时 {time.time()-t0:.0f}s')

FEATS = ['conf_h', 'bars', 'v_mean', 'v_std', 'v_max', 'v_last', 'v_max_pos',
         'a_mean', 'a_std', 'a_max', 'a_min', 'a_last', 'jk_std',
         'turn', 'turn_rate', 'log_fuel', 'sum_v']

log('\n' + '=' * 116)
log('【A】认定前三组的导数特征中位数（全部在 t=认定点时即可观测）')
log('=' * 116)
log(f'  {"特征":<12}{"BIG(≥100%)":>14}{"MID(50~100%)":>15}{"SML(30~50%)":>14}{"BIG-SML差":>12}')
for f in FEATS:
    b = DF[DF.grp == 'BIG'][f].median(); m = DF[DF.grp == 'MID'][f].median(); s = DF[DF.grp == 'SML'][f].median()
    log(f'  {f:<12}{b:>14.4g}{m:>15.4g}{s:>14.4g}{(b-s):>12.4g}')

from sklearn.metrics import roc_auc_score
log('\n' + '=' * 116)
log('【B】★ 认定前能不能分出「翻倍大周期」和「刚够认定的小周期」（BIG vs SML 单变量 AUC）')
log('=' * 116)
sub = DF[DF.grp.isin(['BIG', 'SML'])].copy()
y = (sub.grp == 'BIG').astype('int8').values
res = []
for f in FEATS:
    x = sub[f].values.astype('float64')
    ok = np.isfinite(x)
    if ok.sum() < 100 or y[ok].sum() < 20:
        continue
    auc = roc_auc_score(y[ok], x[ok])
    res.append((f, auc, ok.sum()))
res.sort(key=lambda t: -abs(t[1] - 0.5))
log(f'  {"特征":<12}{"AUC":>8}{"样本":>9}{"方向":>10}')
for f, auc, nn in res:
    log(f'  {f:<12}{auc:>8.3f}{nn:>9,}{"越大越可能BIG" if auc>0.5 else "越小越可能BIG":>10}')

log('\n' + '=' * 116)
log('【C】组合模型（BIG vs SML）：只用认定前可观测的导数特征')
log('=' * 116)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
X = np.column_stack([np.nan_to_num(sub[f].values.astype('float64'), nan=0.0, posinf=0.0, neginf=0.0) for f in FEATS])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_depth=3,
                                     min_samples_leaf=30, random_state=7)
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
log(f'  测试 AUC = {roc_auc_score(yte, p):.3f}   测试基准 BIG 占比 {100*yte.mean():.1f}%')
log(f'  {"阈值":>6}{"报出":>8}{"精确率%":>10}{"召回%":>8}')
for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
    m = p >= thr
    if m.sum() == 0:
        log(f'  {thr:>6.1f}{"无":>8}'); continue
    log(f'  {thr:>6.1f}{m.sum():>8}{100*yte[m].mean():>10.1f}{100*m[yte==1].sum()/max((yte==1).sum(),1):>8.1f}')

log('\n' + '=' * 116)
log('【D】关键变量分档：认定前最后 6 根的「加速度 a_last」 vs 最终涨幅')
log('=' * 116)
qs = DF['a_last'].quantile([0, .2, .4, .6, .8, 1.0]).values
log(f'  {"a_last 档":<20}{"周期数":>8}{"涨幅中位%":>12}{"≥100%占比":>12}')
for k in range(5):
    m = (DF.a_last >= qs[k]) & (DF.a_last <= qs[k+1] if k == 4 else DF.a_last < qs[k+1])
    d = DF[m]
    if len(d) < 10:
        continue
    log(f'  [{qs[k]:+.5f},{qs[k+1]:+.5f}]{len(d):>8}{100*d.gain.median():>12.0f}'
        f'{100*(d.gain>=1.0).mean():>12.1f}')

out = os.path.join(OUT, 'kinematics_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
