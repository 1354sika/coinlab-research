# -*- coding: utf-8 -*-
"""gambler_retail30.py — Q2：散户博弈（局外）在「+30% 可实现」口径下有没有机会

用户判定（2026-10-17）：
    「散户博弈仍然有巨大的机会，因为机会频繁，数据样本更大，
      也符合期望收益 30% 的标准。」

此前判"局外负期望"用的是旧口径（+10% 止盈 / 2h 超时 / -5% 止损）：
    job-36192-11 gambler-normal：局内 +2.584%/笔、局外 **-0.100%/笔**
    ⇒ 那是 10% 口径，不是用户要的 30% 口径。**必须在同一把尺子下重测。**

【本脚本口径（与 v5 完全一致，可比）】
    样本   ：局外点 = 当前价相对过去 288 根(1天)最低点已涨 < 15%（**没启动**）
    标签   ：未来 48h 内最高价 / 当前价 - 1 >= +30%      （用户口径）
    交易   ：入场后 +30% 止盈 / -15% 止损 / 48h 超时，扣 0.09%（maker 0.04%+taker 0.05%）
             同根内若高低都触，**保守按止损先成交**
    切分   ：前 70% 训练 / 后 30% 测试，**只报测试段**

【散户情绪特征（全部 t 时刻可观测，来自 K 线自带的 taker/笔数）】
    trade_intensity  成交笔数 / 24 根均值            （人多不多）
    avg_trade_size   单笔均额 / 30 天中位             （大单还是散户）
    taker_buy_ratio  主动买额 / 成交额，24 根均值      （买卖谁主动）
    vol_price_div    主动买占比 z 分 - 价格涨幅 z 分   （量价背离）
    run_same_dir     连续同向根数                     （惯性）
    vola_jump        近 12 根波动 / 过去 96 根波动     （波动跳升）
    fomo             r12 高 且 量比高                 （追涨）
    panic            r12 低 且 量比高                 （恐慌）
    volr6/24/96      量比（超 30 日均量倍数）
    mkt_up / mkt_dn  全市场同刻涨/跌 >5% 的币数        （市场氛围）
    btc_r24          BTC 24h 涨幅                    （大盘）
    pos288           在 288 根箱体中的位置
    hour / dow       时段

产物：evidence/butterfly/retail30_summary.txt
"""
import os, gc, time, itertools
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288
T_H = 48
TARGET = 0.30
SL = 0.15
FEE = 0.0009
_WIN = int(T_H * 12)
SPLIT = 0.70
MIN_N = 200
MIN_LIFT1 = 1.30

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 110)
log('gambler_retail30 · Q2 散户博弈（局外）在 +30% 可实现口径下的机会')
log('=' * 110)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low',
                               'quote_volume', 'trades', 'taker_quote'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32', 'taker_quote': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64'), g['taker_quote'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns)
SYM2CI = {s: i for i, s in enumerate(syms)}
ts = C.index.values
T, N = C.shape
log(f'币 {len(syms)}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')
t_cut = int(T * 0.70)

# 市场氛围（全市场同刻）
R12all = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12all, nan=0.0) >= 0.05).sum(axis=1)
dn_cnt = (np.nan_to_num(R12all, nan=0.0) <= -0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
mkt_dn = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r24 = np.zeros(T); btc_r24[24:] = btc[24:] / np.maximum(btc[:-24], 1e-12) - 1.0
del C, R12all; gc.collect()

FN = ['r12', 'r288', 'over_now', 'pos288', 'dist_hi288',
      'volr6', 'volr24', 'volr96', 'trade_intensity', 'avg_trade_size',
      'taker_buy_ratio', 'vol_price_div', 'run_same_dir', 'vola_jump',
      'fomo', 'panic', 'mkt_up', 'mkt_dn', 'btc_r24', 'hour', 'dow']
XS, YS, WS, TS_ = [], [], [], []   # 特征 / 标签 / 交易净收益 / 时间位置
CHK_TP = CHK_SL = CHK_N = 0        # 自检：窗口内触及止盈/止损的比例

for (sym, ot, c, h, l, q, tr, tq) in RAW:
    n = len(c)
    if n < WDAY * 4:
        continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q); trs = pd.Series(tr)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over_now = c / np.maximum(lo288, 1e-12) - 1.0
        dist_hi288 = c / np.maximum(hi288, 1e-12) - 1.0
        pos288 = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    r = {}
    for k in (6, 12, 24, 96, 288):
        v = np.full(n, np.nan); v[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1.0
        r[k] = v
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values
    mt = trs.rolling(24, min_periods=6).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        trade_intensity = tr / np.maximum(mt, 1e-12)
        ats = q / np.maximum(tr, 1e-12)
        ats_m = pd.Series(ats).rolling(WDAY * 30, min_periods=WDAY).median().values
        avg_trade_size = ats / np.maximum(ats_m, 1e-12)
        tbr = tq / np.maximum(q, 1e-12)
        tbr24 = pd.Series(tbr).rolling(24, min_periods=6).mean().values
    tbr_m = pd.Series(tbr24).rolling(WDAY, min_periods=72).mean().values
    tbr_s = pd.Series(tbr24).rolling(WDAY, min_periods=72).std().values
    r12m = pd.Series(r[12]).rolling(WDAY, min_periods=72).mean().values
    r12s = pd.Series(r[12]).rolling(WDAY, min_periods=72).std().values
    with np.errstate(invalid='ignore', divide='ignore'):
        vol_price_div = (tbr24 - tbr_m) / np.maximum(tbr_s, 1e-12) - (r[12] - r12m) / np.maximum(r12s, 1e-12)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
    sg = np.sign(np.nan_to_num(r1))
    run = np.ones(n)
    for i in range(1, n):
        run[i] = run[i-1] + 1 if (sg[i] == sg[i-1] and sg[i] != 0) else 1
    av12 = pd.Series(r1).rolling(12, min_periods=6).std().values
    av96 = pd.Series(r1).rolling(96, min_periods=24).std().values
    with np.errstate(invalid='ignore', divide='ignore'):
        vola_jump = av12 / np.maximum(av96, 1e-12)
    fomo = ((r[12] >= 0.05).astype('float64') * (volr6 >= 3).astype('float64'))
    panic = ((r[12] <= -0.05).astype('float64') * (volr6 >= 3).astype('float64'))

    # ★ 窗口法（与 pos_trade 同法，已验证）：直接取 [t, t+_WIN) 的真实高低价
    m = (over_now < 0.15) & np.isfinite(volr24) & (np.arange(n) < n - _WIN - 1)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        continue
    if len(idx) > 4000:                       # 抽样，控制内存（每币上限 4000 点）
        idx = idx[::max(1, len(idx) // 4000)]
    wh = np.lib.stride_tricks.sliding_window_view(h, _WIN)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, _WIN)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, _WIN)[idx]
    c0 = c[idx]
    hit_tp = wh >= (c0 * (1 + TARGET))[:, None]
    hit_sl = wl <= (c0 * (1 - SL))[:, None]
    it = np.argmax(hit_tp, axis=1); il = np.argmax(hit_sl, axis=1)
    any_tp = hit_tp.any(1); any_sl = hit_sl.any(1)
    settle = wc[:, -1] / c0 - 1.0
    w_ = np.where(any_sl & (~any_tp | (il <= it)), -SL,
                  np.where(any_tp, TARGET, settle)) - FEE
    y_ = ((wh.max(1) / c0 - 1.0) >= TARGET).astype('int8')
    CHK_TP += int(any_tp.sum()); CHK_SL += int(any_sl.sum()); CHK_N += len(idx)

    posv = np.searchsorted(ts, ot)
    hourv = ((ot // 3_600_000) % 24).astype('float64')
    dowv = (((ot // 86_400_000) + 4) % 7).astype('float64')
    cols = [r[12], r[288], over_now, pos288, dist_hi288,
            volr6, volr24, volr96, trade_intensity, avg_trade_size,
            tbr24, vol_price_div, run, vola_jump, fomo, panic,
            mkt_up[posv], mkt_dn[posv], btc_r24[posv], hourv, dowv]
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])[idx]
    XS.append(M.astype('float32'))
    YS.append(y_)
    WS.append(w_.astype('float32'))
    TS_.append(posv[idx].astype('int64'))

del RAW; gc.collect()
if not XS:
    log('无样本，退出'); raise SystemExit(1)
X = np.vstack(XS); Y = np.concatenate(YS); W = np.concatenate(WS); tidx = np.concatenate(TS_)
del XS, YS, WS, TS_; gc.collect()
tr = tidx < t_cut; te = ~tr
log(f'局外样本 {len(Y):,}   训练 {tr.sum():,}   测试 {te.sum():,}')
log(f'窗口自检：触及止盈 {100*CHK_TP/max(CHK_N,1):.2f}%   触及止损 {100*CHK_SL/max(CHK_N,1):.2f}%   （同窗口两者都触 = 保守按止损）')
log(f'★ 基准：测试段 P(未来48h触及+30%) = {100*Y[te].mean():.3f}%   '
    f'对比 v5 全局基准 1.9464%')
log(f'★ 交易基准（全部局外点无脑入场）：期望净 {100*np.nanmean(W[te]):+.3f}%/笔  '
    f'中位 {100*np.nanmedian(W[te]):+.3f}%  胜率 {100*np.nanmean(W[te] > 0):.1f}%')

# ---------- 条件池 ----------
CONDS = []   # (name, train bool vec)
def mk(name, v):
    CONDS.append((name, v))

def quants(v, tag, qs=(10, 25, 50, 75, 90)):
    fin = np.isfinite(v)
    if fin.sum() < 1000:
        return
    qv = np.percentile(v[fin], qs)
    for q, val in zip(qs, qv):
        mk(f'{tag}>={val:.4g}', v >= val)
        mk(f'{tag}<={val:.4g}', v <= val)

for j, nm in enumerate(FN):
    quants(X[:, j], nm)
# 组合型
mk('fomo>0 & volr24>=2', (X[:, FN.index('fomo')] > 0) & (X[:, FN.index('volr24')] >= 2))
mk('panic>0 & volr24>=2', (X[:, FN.index('panic')] > 0) & (X[:, FN.index('volr24')] >= 2))
mk('tbr24>=0.55 & volr24>=2', (X[:, FN.index('taker_buy_ratio')] >= 0.55) & (X[:, FN.index('volr24')] >= 2))
mk('vol_price_div>=1 & volr6>=3', (X[:, FN.index('vol_price_div')] >= 1) & (X[:, FN.index('volr6')] >= 3))
log(f'条件池 {len(CONDS)} 条')

base_tr = Y[tr].mean()
pool = []
for nm, v in CONDS:
    m = v[tr]
    if m.sum() < MIN_N:
        continue
    lift = (Y[tr][m].mean() / base_tr) if base_tr > 0 else 0
    if lift >= MIN_LIFT1:
        pool.append((nm, lift, int(m.sum())))
pool.sort(key=lambda x: -x[1])
log(f'\n高质量条件（训练段 lift>={MIN_LIFT1}）：{len(pool)} / {len(CONDS)}')
log(f'  {"条件":<28}{"训练lift":>10}{"n_train":>10}{"测试lift":>10}{"测试n":>9}')
for nm, lift, nt in pool[:25]:
    v = dict(CONDS)[nm]
    mt = v[te]
    lt = (Y[te][mt].mean() / Y[te].mean()) if (mt.sum() > 0 and Y[te].mean() > 0) else float('nan')
    log(f'  {nm:<28}{lift:>10.2f}{nt:>10,}{lt:>10.2f}{int(mt.sum()):>9,}')

# ---------- 2 元组合真穷举 ----------
log('\n' + '=' * 110)
log('【2 元组合真穷举（高质量池）】训练段搜索 → 测试段复核')
log('=' * 110)
names = [p[0] for p in pool]
D2 = dict(CONDS)
res = []
base_te = Y[te].mean()
for a, b in itertools.combinations(range(len(names)), 2):
    va = D2[names[a]]; vb = D2[names[b]]
    m = va & vb
    nt = int((m & tr).sum())
    if nt < MIN_N:
        continue
    lt_tr = Y[tr][m[tr]].mean() / base_tr
    if lt_tr < 1.5:
        continue
    mt = m & te
    mte_ = mt[te]
    nte = int(mte_.sum())
    if nte < 50:
        continue
    lt_te = Y[te][mte_].mean() / base_te
    wmean = np.nanmean(W[te][mte_])
    res.append((names[a], names[b], lt_tr, nt, lt_te, nte, wmean, Y[te][mte_].mean()))
res.sort(key=lambda x: -x[4])
log(f'  有效组合 {len(res)}  测试段基准 {100*base_te:.3f}%')
log(f'  {"组合":<52}{"训lift":>8}{"测lift":>8}{"测n":>8}{"期望净%":>9}{"精确率%":>9}')
for a, b, ltr, nt, lte, nte, wm, pr in res[:25]:
    log(f'  {a[:24]:<24} & {b[:24]:<24}{ltr:>8.2f}{lte:>8.2f}{nte:>8,}{100*wm:>+9.2f}{100*pr:>9.2f}')
if res:
    best = sorted(res, key=lambda x: -x[6])[:10]
    log('\n  ★ 按测试段期望净排序 top-10（这才是判据）')
    log(f'  {"组合":<52}{"测n":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"P(30%)%":>9}')
    for a, b, ltr, nt, lte, nte, wm, pr in best:
        m = (D2[a] & D2[b])[te]
        wv = W[te][m]
        log(f'  {a[:24]:<24} & {b[:24]:<24}{nte:>8,}{100*wm:>+10.2f}'
            f'{100*np.nanmedian(wv):>9.2f}{100*np.nanmean(wv > 0):>8.1f}{100*pr:>9.2f}')

# ---------- 模型法（对照：全部特征一起用） ----------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                     min_samples_leaf=200, random_state=7)
clf.fit(X[tr], Y[tr])
p = clf.predict_proba(X[te])[:, 1]
log('\n' + '=' * 110)
log('【模型法对照：全部散户情绪特征】')
log('=' * 110)
log(f'  AUC = {roc_auc_score(Y[te], p):.4f}   测试基准 {100*base_te:.3f}%')
log(f'  {"阈值":>6}{"报出":>10}{"覆盖":>9}{"精确率":>9}{"召回":>8}{"期望净%":>10}{"胜率":>8}')
for thr in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    m = p >= thr
    if m.sum() == 0:
        log(f'  {thr:>6.2f}{"无报出":>10}'); continue
    prec = Y[te][m].mean()
    rec = m[Y[te] == 1].sum() / max((Y[te] == 1).sum(), 1)
    wv = W[te][m]
    log(f'  {thr:>6.2f}{m.sum():>10,}{100*m.mean():>8.2f}%{100*prec:>8.2f}%{100*rec:>7.2f}%'
        f'{100*np.nanmean(wv):>+10.2f}{100*np.nanmean(wv > 0):>7.1f}%')

# ---------- 特征重要性（置换，测试段） ----------
rng = np.random.default_rng(7)
base_auc = roc_auc_score(Y[te], p)
imp = []
for j, nm in enumerate(FN):
    Xt = X[te].copy(); perm = rng.permutation(len(Xt))
    Xt[:, j] = Xt[perm, j]
    imp.append((nm, base_auc - roc_auc_score(Y[te], clf.predict_proba(Xt)[:, 1])))
imp.sort(key=lambda x: -x[1])
log('\n  置换重要性 top-8（测试段）')
for nm, v in imp[:8]:
    log(f'    {nm:<20}{v:>+10.4f}')

out = os.path.join(OUT, 'retail30_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
pd.DataFrame(res, columns=['a', 'b', 'lift_train', 'n_train', 'lift_test', 'n_test',
                           'exp_net', 'precision']).to_csv(
    os.path.join(OUT, 'retail30_combos.csv'), index=False, encoding='utf-8-sig')
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
