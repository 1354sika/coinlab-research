# -*- coding: utf-8 -*-
"""
gambler_alpha_small · 小周期(+5%)到底有没有可预测的 alpha
核心对照：
  A 绝对标签 y_abs = 未来24h最大涨幅 >= 5%
  B 相对标签 y_rel = 未来24h最大涨幅 - BTC同期最大涨幅 >= 5%
同一批入场点、同一批特征、同一个模型、同一切分
若 AUC(B) > AUC(A) + 0.05 ⇒ alpha 比 beta 更可预测，小周期有救
若两者都 ~0.5 ⇒ 这份数据里确实没有方向信息
"""
import os, sys, time, io
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FN = 'klines_1m_1y.csv'
LOW_WIN = 240
HOLD = 1440
CONFIRM = 0.03
TP = 0.05
FEE = 0.005

logbuf = []
def log(s=''):
    print(s)
    logbuf.append(str(s))

def read_all():
    p = os.path.join(DATA, FN)
    lst = []
    for chunk in pd.read_csv(p, chunksize=2_000_000,
                             usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base', 'quote_volume']):
        chunk['symbol'] = chunk['symbol'].astype(str)
        lst.append(chunk)
    df = pd.concat(lst, ignore_index=True)
    return df

def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_alpha_small · 小周期(+5%)是 beta 还是 alpha')
    log('=' * 100)

    df = read_all()
    syms = sorted(df['symbol'].unique().tolist())
    log(f'币 {len(syms)}  行 {len(df):,}  读取 {time.time()-t0:.0f}s')

    # ---- BTC 基准 ----
    g = df[df['symbol'] == 'BTCUSDT'].sort_values('open_time')
    if len(g) == 0:
        log('无 BTCUSDT，退出')
        return
    btc_ot = g['open_time'].values.astype(np.int64)
    btc_h = g['high'].values.astype(np.float64)
    btc_c = g['close'].values.astype(np.float64)
    btc_hs = pd.Series(btc_h)
    btc_futmax = btc_hs.rolling(HOLD, min_periods=1).max().shift(-HOLD).values
    log(f'BTC 基准就绪  行 {len(btc_ot):,}')

    X_all, y_abs, y_rel, t_all = [], [], [], []

    for si, sym in enumerate(syms):
        if sym == 'BTCUSDT':
            continue
        g = df[df['symbol'] == sym].sort_values('open_time')
        ot = g['open_time'].values.astype(np.int64)
        o = g['open'].values.astype(np.float64)
        h = g['high'].values.astype(np.float64)
        l = g['low'].values.astype(np.float64)
        c = g['close'].values.astype(np.float64)
        v = g['volume'].values.astype(np.float64)
        qv = g['quote_volume'].values.astype(np.float64)
        tb = g['taker_base'].values.astype(np.float64)
        n = len(c)
        if n < LOW_WIN + HOLD + 10:
            continue

        lo_win = pd.Series(l).rolling(LOW_WIN, min_periods=60).min().values
        r1 = np.zeros(n); r1[1:] = c[1:] / c[:-1] - 1
        r5 = np.zeros(n); r5[5:] = c[5:] / c[:-5] - 1
        r15 = np.zeros(n); r15[15:] = c[15:] / c[:-15] - 1
        r60 = np.zeros(n); r60[60:] = c[60:] / c[:-60] - 1
        r240 = np.zeros(n); r240[240:] = c[240:] / c[:-240] - 1
        vbase = pd.Series(v).rolling(288, min_periods=60).mean().values
        volr = np.where(vbase > 0, v / np.maximum(vbase, 1e-12), np.nan)
        tbr = np.where(v > 0, tb / np.maximum(v, 1e-12), np.nan)
        rets = r1[1:]
        vstd = pd.Series(rets).rolling(60, min_periods=30).std().values
        vstd = np.concatenate([[np.nan], vstd])
        vstd1440 = pd.Series(rets).rolling(1440, min_periods=240).std().values
        vstd1440 = np.concatenate([[np.nan], vstd1440])
        hi1440 = pd.Series(h).rolling(1440, min_periods=60).max().values
        pos = np.where(hi1440 > 0, c / hi1440, np.nan)
        hour = ((ot // 3_600_000) % 24).astype(np.float64)
        dow = ((ot // 86_400_000) % 7).astype(np.float64)
        lo_age = pd.Series(l).rolling(720, min_periods=60).apply(lambda a: (len(a) - 1 - int(np.argmin(a))), raw=True).values
        # 未来 24h 最大涨幅
        fut_max = pd.Series(h).rolling(HOLD, min_periods=1).max().shift(-HOLD).values

        # BTC 同期
        idx = np.searchsorted(btc_ot, ot)
        idx = np.clip(idx, 0, len(btc_ot) - 1)
        btc_entry_c = btc_c[idx]
        btc_fm = btc_futmax[idx]
        btc_fut_ret = np.where(btc_entry_c > 0, btc_fm / btc_entry_c - 1, np.nan)
        # BTC 前 15/60 分钟涨幅
        btc_r15 = np.zeros(len(btc_c)); btc_r15[15:] = btc_c[15:] / btc_c[:-15] - 1
        btc_r60 = np.zeros(len(btc_c)); btc_r60[60:] = btc_c[60:] / btc_c[:-60] - 1
        btc_r15v = btc_r15[idx]; btc_r60v = btc_r60[idx]
        btc_rets = np.zeros(len(btc_c)); btc_rets[1:] = btc_c[1:] / btc_c[:-1] - 1
        btc_vstd_arr = pd.Series(btc_rets).rolling(60, min_periods=30).std().values
        btc_vstdv = btc_vstd_arr[idx]

        entry = lo_win * (1 + CONFIRM)
        m = (c >= entry) & (o <= entry * 1.03)
        m[0] = False
        m = m & np.isfinite(fut_max) & np.isfinite(lo_win) & np.isfinite(pos)
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue

        feats = np.column_stack([
            r1[idxs], r5[idxs], r15[idxs], r60[idxs], r240[idxs],
            volr[idxs], tbr[idxs], vstd[idxs], vstd1440[idxs],
            hour[idxs], dow[idxs], pos[idxs], lo_age[idxs],
            btc_r15v[idxs], btc_r60v[idxs], btc_vstdv[idxs],
        ])
        fmax = fut_max[idxs]
        fund = entry[idxs]
        ret_fut = fmax / fund - 1
        y_a = (ret_fut >= TP).astype(np.int8)
        y_r = ((ret_fut - btc_fut_ret[idxs]) >= TP).astype(np.int8)

        ok = np.isfinite(feats).all(axis=1)
        X_all.append(feats[ok]); y_abs.append(y_a[ok]); y_rel.append(y_r[ok])
        t_all.append(ot[idxs][ok])

        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(X_all); ya = np.concatenate(y_abs); yr = np.concatenate(y_rel)
    tt = np.concatenate(t_all)
    order = np.argsort(tt)
    X, ya, yr = X[order], ya[order], yr[order]

    n = len(X)
    cut = int(n * 0.7)
    FEATS = ['r1', 'r5', 'r15', 'r60', 'r240', 'volr', 'tbr', 'vstd', 'vstd1440',
             'hour', 'dow', 'pos', 'lo_age', 'btc_r15', 'btc_r60', 'btc_vstd']

    log(f'\n样本 {n:,}  训练 {cut:,}  测试 {n-cut:,}')
    log(f'   基线 y_abs 比例 {ya.mean():.1%}   y_rel 比例 {yr.mean():.1%}')

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    for tag, y in [('A 绝对(币自己的涨幅>=5%)', ya), ('B 相对(跑赢BTC超5%)', yr)]:
        mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                             max_depth=6, min_samples_leaf=200,
                                             random_state=7)
        mdl.fit(X[:cut], y[:cut])
        p = mdl.predict_proba(X[cut:])[:, 1]
        auc = roc_auc_score(y[cut:], p)
        log(f'\n【{tag}】')
        log(f'  测试段 AUC = {auc:.4f}   基线 {y[cut:].mean():.1%}')
        for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
            mk = (p >= lo) & (p < hi)
            if mk.sum() == 0:
                continue
            log(f'  P∈[{lo:.1f},{hi:.2f})  n={mk.sum():>7,}  实际比例 {y[cut:][mk].mean():>6.1%}')
        imp = getattr(mdl, 'feature_importances_', None)
        if imp is not None:
            top = sorted(zip(FEATS, imp), key=lambda x: -x[1])[:8]
            log('  特征重要性 Top8: ' + '  '.join(f'{k}={v:.3f}' for k, v in top))

    log('\n' + '=' * 100)
    log('【判读】')
    log('  · AUC(B) - AUC(A) > 0.05 ⇒ alpha 比 beta 可预测，小周期有救')
    log('  · 两者都 ~0.5 ⇒ 这份数据里没有方向信息（beta 和 alpha 都没有）')
    log('  · AUC(A) 低但 AUC(B) 高 ⇒ 之前的失败是"被市场涨跌淹没"，该换标签重做')

    out = os.path.join(OUT, 'alpha_small_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
