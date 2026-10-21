# -*- coding: utf-8 -*-
"""gambler_annual_diag · 查三个怀疑（1 年数据、按日历切分）

1) 特征重要性：模型到底在用什么特征（是 vstd 波动率，还是真有方向信息）
2) 925 笔的集中度：按币、按天，是不是集中在少数暴涨事件
3) 幸存者偏差：这 60 个币 1 年的涨幅分布
"""
import os, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
START = 50.0
POS = 0.10
MAXSYM = int(os.environ.get('SYMS', '60'))

VW = 240
CONF = 2880
HOLD = 1440
TPM, SLM = 0.30, 0.95
FEATS = ['volr','tbr','v240','pos','lo_age','r1','r15','r60','hr','dw','h3','h10']

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_annual_diag · 查三个怀疑')
log('=' * 100)

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = pd.Series(l).rolling(121, min_periods=1).min().values
    lo_i = []
    i = 120
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 30
        else:
            i += 1
    pairs = []
    for i in lo_i:
        if i + CONF >= n: continue
        for j in range(i + 1, min(i + CONF + 1, n)):
            if c[j] >= l[i] * 1.03:
                pairs.append((i, j)); break
    return pairs


def build(pairs, o, h, l, c, t, v, tb):
    n = len(c)
    feats, entries = [], []
    hist_hits = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]; lo_p = l[li]
        tp_px = lo_p * (1.0 + TPM)
        sl_px = lo_p * SLM
        seg_h = h[j + 1:j + 1 + HOLD]
        seg_l = l[j + 1:j + 1 + HOLD]
        hit_tp = seg_h >= tp_px
        hit_sl = seg_l <= sl_px
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
        if i_tp >= 0 and i_sl >= 0:
            if i_tp < i_sl:
                ret = tp_px / e - 1; is_tp = 1
            else:
                ret = sl_px / e - 1; is_tp = 0
        elif i_tp >= 0:
            ret = tp_px / e - 1; is_tp = 1
        elif i_sl >= 0:
            ret = sl_px / e - 1; is_tp = 0
        else:
            ret = c[min(j + HOLD, n - 1)] / e - 1; is_tp = 0

        vw = v[j - VW:j + 1]
        v240 = float(np.std(np.log1p(np.maximum(vw, 0))))
        vm = float(np.mean(vw))
        volr = float(v[j] / vm) if vm > 0 else 1.0
        vb = np.sum(v[j - 14:j])
        tbr = float(np.sum(tb[j - 14:j]) / vb) if vb > 0 else 0.5
        r1 = e / c[j - 1] - 1 if j >= 1 else 0.0
        r15 = e / c[j - 15] - 1 if j >= 15 else 0.0
        r60 = e / c[j - 60] - 1 if j >= 60 else 0.0
        ts = pd.Timestamp(int(t[j]) // 1000, unit='s', tz='UTC')
        pos = e / lo_p if lo_p > 0 else 1.0
        lo_age = j - li
        h3 = float(np.mean(hist_hits[-3:])) if hist_hits else -1.0
        h10 = float(np.mean(hist_hits[-10:])) if hist_hits else -1.0
        feats.append([volr, tbr, v240, pos, lo_age, r1, r15, r60, ts.hour, ts.dayofweek, h3, h10])
        entries.append((int(t[j]), ret, is_tp, sym_of))
        hist_hits.append(is_tp)
    return np.asarray(feats), entries


ROWS = []
sym_of = None
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    sym_of = sym
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    X, E = build(pairs, o, h, l, c, t, v, tb)
    for (ts, ret, is_tp, sy), feat in zip(E, X):
        ROWS.append((ts, ret, is_tp, sy, feat))
log(f'入场点 {len(ROWS):,}  耗时 {time.time()-t0:.0f}s')

df = pd.DataFrame({'t': [r[0] for r in ROWS], 'ret': [r[1] for r in ROWS],
                   'tp': [r[2] for r in ROWS], 'sym': [r[3] for r in ROWS]})
X = np.array([r[4] for r in ROWS], dtype=float)
df['ym'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
df['ymd'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')
months = sorted(df['ym'].unique())
ntr = max(1, int(len(months) * 0.75))
train_m = months[:ntr]; test_m = months[ntr:]
tr = np.where(df['ym'].isin(train_m).values)[0]
te = np.where(df['ym'].isin(test_m).values)[0]
log(f'训练 {len(tr):,}  测试 {len(te):,}  训练月 {train_m[0]}~{train_m[-1]}  测试月 {test_m[0]}~{test_m[-1]}')

mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Pte = mdl.predict_proba(X[te])[:, 1]
log(f'AUC 样本外 {roc_auc_score(df["tp"].values[te], Pte):.4f}')

# ---- 1) 特征重要性 ----
log('')
log('=' * 100)
log('【一】特征重要性（permutation，测试段，n_repeats=3）')
sub = np.arange(len(te))
rng = np.random.default_rng(7)
if len(sub) > 20000:
    sub = rng.choice(sub, 20000, replace=False)
res = permutation_importance(mdl, X[te][sub], df['tp'].values[te][sub],
                             scoring='roc_auc', n_repeats=3, random_state=7, n_jobs=1)
imp = sorted(zip(FEATS, res.importances_mean, res.importances_std), key=lambda x: -x[1])
log(f"{'feature':<10}{'AUC 降幅':>12}{'std':>10}")
tot = max(sum(max(v, 0) for _, v, _ in imp), 1e-9)
for f, v, s in imp:
    log(f'{f:<10}{v:>12.4f}{s:>10.4f}   {max(v,0)/tot*100:>5.1f}%')

# ---- 2) 925 笔集中度 ----
sel = te[Pte >= 0.5]
d = df.iloc[sel].copy()
log('')
log('=' * 100)
log(f'【二】P>=0.5 的 {len(d)} 笔集中度')
log(f"  按币：top1 {d['sym'].value_counts().iloc[0]} 笔（{d['sym'].value_counts().iloc[0]/len(d)*100:.1f}%）  "
    f"top5 {d['sym'].value_counts().head(5).sum()} 笔（{d['sym'].value_counts().head(5).sum()/len(d)*100:.1f}%）")
log(f"  按天：top1 {d['ymd'].value_counts().iloc[0]} 笔（{d['ymd'].value_counts().iloc[0]/len(d)*100:.1f}%）  "
    f"top5 {d['ymd'].value_counts().head(5).sum()} 笔（{d['ymd'].value_counts().head(5).sum()/len(d)*100:.1f}%）")
log(f"  按币数：{d['sym'].nunique()} / {df['sym'].nunique()} 个币有交易")
log(f"  按天数：{d['ymd'].nunique()} / {df['ymd'].nunique()} 天有交易")
log('')
log('  top10 币：')
for sym, cnt in d['sym'].value_counts().head(10).items():
    sub2 = d[d['sym'] == sym]
    log(f'    {sym:<14}{cnt:>5} 笔  均值 {sub2["ret"].mean()*100:>6.2f}%  TP率 {sub2["tp"].mean()*100:>5.1f}%')
log('')
log('  top10 天：')
for ymd, cnt in d['ymd'].value_counts().head(10).items():
    sub2 = d[d['ymd'] == ymd]
    log(f'    {ymd}  {cnt:>4} 笔  均值 {sub2["ret"].mean()*100:>6.2f}%')

# ---- 3) 幸存者偏差：60 币 1 年涨幅 ----
log('')
log('=' * 100)
log('【三】60 个币的 1 年表现（幸存者偏差检查）')
rows = []
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    if len(c) < 5000: continue
    first = c[0]; last = c[-1]
    mx = np.max(h)
    rows.append((sym, last / first - 1, mx / first - 1, len(c)))
rv = pd.DataFrame(rows, columns=['sym', 'yr_ret', 'max_up', 'bars'])
log(f'  币数 {len(rv)}  数据根数中位 {rv["bars"].median():,.0f}（1 年约 525,600）')
log(f'  1 年收益：中位 {rv["yr_ret"].median()*100:>8.1f}%   均值 {rv["yr_ret"].mean()*100:>8.1f}%   '
    f'最低 {rv["yr_ret"].min()*100:>8.1f}%   最高 {rv["yr_ret"].max()*100:>8.1f}%')
log(f'  正收益币占比 {(rv["yr_ret"] > 0).mean()*100:.1f}%   '
    f'涨幅>100% 的币占比 {(rv["yr_ret"] > 1).mean()*100:.1f}%')
log(f'  期间最大涨幅：中位 {rv["max_up"].median()*100:.0f}%   均值 {rv["max_up"].mean()*100:.0f}%')
log('')
log('  涨幅最高的 10 个币：')
for row in rv.sort_values('yr_ret', ascending=False).head(10).itertuples():
    log(f'    {row.sym:<14} 1年 {row.yr_ret*100:>8.1f}%   期间最大 {row.max_up*100:>8.0f}%')

log('')
log('=' * 100)
log('【判读】')
log('  · 若 vstd/volr 主导且其余 <10% ⇒ 模型学的是"波动大"，不是方向')
log('  · 若 925 笔 >50% 集中在 top5 币或 top5 天 ⇒ 不是普适信号，是少数事件')
log('  · 若 60 币正收益占比高、涨幅中位大 ⇒ 数据本身是牛市样本，存在幸存者偏差')

out = os.path.join(OUT, 'annual_diag_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
