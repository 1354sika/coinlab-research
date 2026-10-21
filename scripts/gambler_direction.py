# -*- coding: utf-8 -*-
"""gambler_direction.py — 振幅只告诉你「要动」，方向才是钱

已确认的三件事：
    ① 局真实存在（偏度 66×、CUSUM 2.67×、振幅 2.06×，全 p<1e-5）
    ② 振幅能预"要动"：未来 6h 起局 AUC 0.9262
    ③ 但无论怎么退出都亏（固定止损 -0.24%、ATR 自适应 -0.44%、等回调 -1.43%）

⇒ 缺的是方向。"要动"和"往哪动"是两件事，我们只解决了前者。

【本脚本】只在振幅异动的时刻，测方向能否预测：
    样本：amp_ratio ≥ 2（8h 振幅是 30 天均值的 2 倍）
    标签（未来 24h，谁先到）：
        UP   先涨 ≥30%
        DN   先跌 ≥30%
        NONE 都没触及
    特征（全部 t 时刻可观测）：
        taker 买占比（主动买/总成交额）及其 z 分
        上影/下影占比（谁在被顶、谁在被砸）
        连续同向根数、量价背离、加速度
        市场齐涨/齐跌币数、BTC 24h 方向
        振幅、量比、位置
    输出：三分类准确率 + "涨 vs 跌"二分类 AUC + 阈值表 + 交易期望
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; FEE = 0.0009
H = 288 * 12          # 未来？其实 24h = 288 根
FUT = 288

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_direction · 振幅异动时，方向能不能预测')
log('=' * 116)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low',
                               'quote_volume', 'trades', 'taker_quote'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32', 'taker_quote': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64'), g['taker_quote'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns); ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12 = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
dn_cnt = (np.nan_to_num(R12, nan=0.0) <= -0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
mkt_dn = pd.Series(dn_cnt).rolling(24, min_periods=1).max().values
btc = C['BTCUSDT'].values.astype('float64') if 'BTCUSDT' in syms else np.zeros(T)
btc_r24 = np.zeros(T); btc_r24[288:] = btc[288:] / np.maximum(btc[:-288], 1e-12) - 1.0
del C, R12; gc.collect()
log(f'币 {N}   全局 {T}   耗时 {time.time()-t0:.0f}s')

FN = ['amp_ratio', 'upper_sh', 'lower_sh', 'tbr24', 'tbr_z', 'trade_int',
      'run_dir', 'vp_div', 'accel', 'r24', 'r288', 'over', 'volr24', 'mkt_up', 'mkt_dn', 'btc_r24']
XS, YS, RS, TID = [], [], [], []
cnt = {'UP': 0, 'DN': 0, 'NONE': 0}

for (sym, ot, c, h, l, q, tr, tq) in RAW:
    n = len(c)
    if n < WDAY * 4 + FUT + 2:
        continue
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    base = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        ar = amp / np.maximum(base, 1e-12)
    rng_ = np.maximum(h - l, 1e-12)
    upper = (h - np.maximum(c, np.roll(c, 1))) / rng_
    lower = (np.minimum(c, np.roll(c, 1)) - l) / rng_
    with np.errstate(invalid='ignore', divide='ignore'):
        tbr = tq / np.maximum(q, 1e-12)
    tbr24 = pd.Series(tbr).rolling(24, min_periods=6).mean().values
    tbr_m = pd.Series(tbr24).rolling(WDAY, min_periods=72).mean().values
    tbr_s = pd.Series(tbr24).rolling(WDAY, min_periods=72).std().values
    with np.errstate(invalid='ignore', divide='ignore'):
        tbr_z = (tbr24 - tbr_m) / np.maximum(tbr_s, 1e-12)
    mt = pd.Series(tr).rolling(24, min_periods=6).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        trade_int = tr / np.maximum(mt, 1e-12)
    cprev = np.roll(c, 1); cprev[0] = c[0]
    sg = np.sign(c - cprev)
    run = np.ones(n)
    for i in range(1, n):
        run[i] = run[i-1] + 1 if (sg[i] == sg[i-1] and sg[i] != 0) else 1
    r6 = np.full(n, np.nan); r6[6:] = c[6:] / np.maximum(c[:-6], 1e-12) - 1.0
    with np.errstate(invalid='ignore'):
        accel = r6 - np.concatenate([np.full(6, np.nan), r6[:-6]])
    r24 = np.full(n, np.nan); r24[288:] = c[288:] / np.maximum(c[:-288], 1e-12) - 1.0
    r288 = np.full(n, np.nan); r288[288:] = r24[288:]
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
        vp_div = tbr_z - (r24 - np.nanmean(r24)) / max(np.nanstd(r24), 1e-12)
    qs = pd.Series(q)
    bq = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = q / np.maximum(bq, 1e-12)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values

    # 未来 24h：先到 +30% 还是先到 -30%
    up_hit = np.zeros(n, bool); dn_hit = np.zeros(n, bool)
    first_up = np.full(n, FUT); first_dn = np.full(n, FUT)
    for k in range(1, FUT + 1):
        j0 = np.arange(n - FUT - 1)
        hh = h[j0 + k]; ll = l[j0 + k]
        m_up = (hh >= c[j0] * 1.30) & (first_up[j0] == FUT)
        m_dn = (ll <= c[j0] * 0.70) & (first_dn[j0] == FUT)
        first_up[j0[m_up]] = k
        first_dn[j0[m_dn]] = k
    up_hit = first_up < first_dn
    dn_hit = first_dn < first_up

    sig = (np.isfinite(ar) & (ar >= 2.0) & (np.arange(n) < n - FUT - 2))
    idx = np.flatnonzero(sig)
    if len(idx) == 0:
        continue
    if len(idx) > 3000:
        idx = idx[::max(1, len(idx)//3000)]
    cols = [ar, upper, lower, tbr24, tbr_z, trade_int, run, vp_div, accel,
            r24, r288, over, volr24, mkt_up[np.searchsorted(ts, ot)][idx] if False else
            np.zeros(n), np.zeros(n), np.zeros(n)]
    posv = np.searchsorted(ts, ot)
    cols[13] = np.interp(np.arange(n), np.arange(n), mkt_up[posv])
    cols[14] = np.interp(np.arange(n), np.arange(n), mkt_dn[posv])
    cols[15] = np.interp(np.arange(n), np.arange(n), btc_r24[posv])
    M = np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in cols])[idx]
    XS.append(M.astype('float32'))
    TID.append(posv[idx].astype('int64'))
    y = np.where(up_hit[idx], 1, np.where(dn_hit[idx], 0, -1)).astype('int8')
    YS.append(y)
    # 交易结果：做多（先涨30%→+0.30；先跌30%→-0.30；都没→0）
    v = np.where(up_hit[idx], 0.30, np.where(dn_hit[idx], -0.30, 0.0)) - FEE
    RS.append(v.astype('float32'))
    cnt['UP'] += int((y == 1).sum()); cnt['DN'] += int((y == 0).sum()); cnt['NONE'] += int((y == -1).sum())
del RAW; gc.collect()

X = np.vstack(XS); Y = np.concatenate(YS); Rv = np.concatenate(RS)
TID = np.concatenate(TID)
tr = TID < t_cut; te = ~tr
log(f'样本 {len(X):,}（振幅≥2×）  UP {cnt["UP"]:,} / DN {cnt["DN"]:,} / NONE {cnt["NONE"]:,}')
log(f'训练 {int(tr.sum()):,}  测试 {int(te.sum()):,}   耗时 {time.time()-t0:.0f}s')

both = Y >= 0
trb = tr & both; teb = te & both
log(f'先到者样本（UP/DN 二者之一）：训练 {int(trb.sum()):,}  测试 {int(teb.sum()):,}   '
    f'其中 UP 占 {100*(Y[teb]==1).mean():.1f}%')

log('\n' + '=' * 116)
log('【A】方向二分类：先涨30% vs 先跌30%（只看振幅异动的时刻）')
log('=' * 116)
if trb.sum() > 200 and teb.sum() > 100:
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4,
                                         min_samples_leaf=100, random_state=7)
    clf.fit(X[trb], (Y[trb] == 1).astype('int8'))
    p = clf.predict_proba(X[teb])[:, 1]
    log(f'  AUC = {roc_auc_score((Y[teb]==1).astype("int8"), p):.4f}   '
        f'测试基准 UP 占比 {100*(Y[teb]==1).mean():.1f}%')
    log(f'  {"阈值":>6}{"报出":>9}{"精确率(涨)":>12}{"召回":>8}{"做多期望净%":>13}{"做空期望净%":>13}')
    for thr in (0.5, 0.55, 0.6, 0.65, 0.7):
        m = p >= thr
        if m.sum() < 20:
            log(f'  {thr:>6.2f}{int(m.sum()):>9}  样本不足'); continue
        prec = (Y[teb][m] == 1).mean()
        rec = m[(Y[teb] == 1)].sum() / max((Y[teb] == 1).sum(), 1)
        log(f'  {thr:>6.2f}{int(m.sum()):>9,}{100*prec:>11.1f}%{100*rec:>7.1f}%'
            f'{100*np.nanmean(Rv[teb][m]):>+13.2f}{100*np.nanmean(-Rv[teb][m]):>+13.2f}')
    # 特征重要性
    rng = np.random.default_rng(7)
    base = roc_auc_score((Y[teb]==1).astype('int8'), p)
    imp = []
    for j, nm in enumerate(FN):
        Xt = X[teb].copy(); perm = rng.permutation(len(Xt))
        Xt[:, j] = Xt[perm, j]
        imp.append((nm, base - roc_auc_score((Y[teb]==1).astype('int8'),
                                             clf.predict_proba(Xt)[:, 1])))
    imp.sort(key=lambda t: -t[1])
    log('\n  方向特征重要性 top-8：')
    for nm, v in imp[:8]:
        log(f'    {nm:<12}{v:>+10.4f}')
else:
    log('  样本不足')

log('\n' + '=' * 116)
log('【B】三分类（UP/DN/NONE）准确率')
log('=' * 116)
clf2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4,
                                      min_samples_leaf=200, random_state=7)
clf2.fit(X[tr], Y[tr])
pred = clf2.predict(X[te])
log(f'  总体准确率 {100*(pred == Y[te]).mean():.2f}%   '
    f'(全判 NONE 的基准 {100*(Y[te] == -1).mean():.2f}%)')
for c in (-1, 0, 1):
    m = Y[te] == c
    if m.sum() == 0:
        continue
    log(f'  真实 {["DN","NONE","UP"][c+1]:<5} 样本 {int(m.sum()):>7,}  召回 {100*(pred[m]==c).mean():>6.2f}%')

out = os.path.join(OUT, 'direction_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
