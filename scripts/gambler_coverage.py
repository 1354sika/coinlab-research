# -*- coding: utf-8 -*-
"""
gambler_coverage · 模型覆盖率：总周期有多少、抓到多少、漏掉多少、误抓多少
  规则已锁定为 A1：240min低点×1.03 / 当根收盘价买入 / TP+5% / SL-6% / 24h / 费0.5%
  输出：各概率阈值下的 精确率 / 召回率 / 抓到真周期数 / 漏掉数 / 误抓数
"""
import os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = r'D:\中信证券\coinlab'
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
OUT = os.path.join(ROOT, 'evidence', 'cycle')

LOW_WIN = 240
CONFIRM = 0.03
HOLD = 1440
TP = 0.05
SL = -0.06
FEE = 0.005
CAP = 5000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_coverage · 模型覆盖率统计（A1 规则锁定）')
    log('=' * 100)
    log(f'  入场=240min低点×{1+CONFIRM:.2f}  买入=当根收盘价  TP=+{TP:.0%}  SL={SL:.0%}  HOLD={HOLD}min  费={FEE:.2%}')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    Xs, ys, ts = [], [], []

    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        ot = g['open_time'].values.astype(np.int64)
        o = g['open'].values.astype(np.float64)
        h = g['high'].values.astype(np.float64)
        l = g['low'].values.astype(np.float64)
        c = g['close'].values.astype(np.float64)
        v = g['volume'].values.astype(np.float64)
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

        entry = lo_win * (1 + CONFIRM)
        m = (c >= entry) & (o <= entry * 1.03)
        m[0] = False
        m = m & np.isfinite(lo_win) & np.isfinite(pos) & np.isfinite(vstd1440)
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > CAP:
            idxs = idxs[::max(1, len(idxs) // CAP)][:CAP]

        feats = np.column_stack([
            r1[idxs], r5[idxs], r15[idxs], r60[idxs], r240[idxs],
            volr[idxs], tbr[idxs], vstd[idxs], vstd1440[idxs],
            hour[idxs], dow[idxs], pos[idxs],
        ])
        ok = np.isfinite(feats).all(axis=1)
        F = feats[ok]
        IDX = idxs[ok]
        if len(IDX) == 0:
            continue

        px = c[IDX]
        tp_p = px * (1 + TP)
        sl_p = px * (1 - abs(SL))
        y = np.zeros(len(IDX), dtype=np.int8)
        for k in range(len(IDX)):
            idx = IDX[k]
            end = min(idx + HOLD, n - 1)
            if end <= idx + 1:
                continue
            seg_h = h[idx + 1:end + 1]
            seg_l = l[idx + 1:end + 1]
            hit_tp = np.where(seg_h >= tp_p[k])[0]
            hit_sl = np.where(seg_l <= sl_p[k])[0]
            i_tp = int(hit_tp[0]) if len(hit_tp) else 10 ** 9
            i_sl = int(hit_sl[0]) if len(hit_sl) else 10 ** 9
            if i_tp < 10 ** 9 and i_tp < i_sl:
                y[k] = 1

        Xs.append(F); ys.append(y); ts.append(ot[IDX])

        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(Xs); y = np.concatenate(ys); tt = np.concatenate(ts)
    order = np.argsort(tt)
    X, y = X[order], y[order]
    n = len(X)
    tr = int(n * 0.7)
    Xtr, Xte, ytr, yte = X[:tr], X[tr:], y[:tr], y[tr:]

    mdl = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                         learning_rate=0.08, random_state=0)
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)

    n_te = len(yte)
    n_pos = int(yte.sum())
    n_neg = n_te - n_pos
    log('\n' + '=' * 100)
    log('【一】测试段总体')
    log('=' * 100)
    log(f'  测试段触发点（满足 低点×1.03 且收盘价到位）   {n_te:,}')
    log(f'  其中「真周期」= 24h 内先触 +5% 且未先触 -6%   {n_pos:,}  ({n_pos/n_te:.1%})')
    log(f'  其中「假信号」= 先触 -6% 或超时未到 +5%        {n_neg:,}  ({n_neg/n_te:.1%})')
    log(f'  AUC = {auc:.4f}')

    log('\n' + '=' * 100)
    log('【二】各概率阈值的覆盖')
    log('=' * 100)
    log(f'  {"阈值":>6} {"报出":>8} {"抓到真周期":>10} {"漏掉真周期":>10} {"误抓(假)":>10} {"精确率":>8} {"召回率":>8} {"净期望":>9}')
    log('  ' + '-' * 90)
    for lo in THRESHOLDS:
        mm = p >= lo
        n_rep = int(mm.sum())
        if n_rep == 0:
            continue
        tp = int(yte[mm].sum())
        fp = n_rep - tp
        fn = n_pos - tp
        prec = tp / n_rep
        rec = tp / n_pos if n_pos > 0 else 0.0
        ne = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lo:>6.1f} {n_rep:>8,} {tp:>10,} {fn:>10,} {fp:>10,} {prec:>7.1%} {rec:>7.1%} {ne*100:>+8.2f}%')

    log('\n' + '=' * 100)
    log('【三】判读')
    log('=' * 100)
    log('  · 召回率 = 抓到的真周期 / 全部真周期。低 = 漏掉大量机会')
    log('  · 精确率 = 抓到的真周期 / 报出的总数。低 = 大量误抓')
    log('  · 提高门槛 ⇒ 精确率升、召回率降、每笔更肥但机会更少')
    log('  · 要同时提精确率与召回率，只能改模型或加特征（K线内已榨干）')

    out = os.path.join(OUT, 'coverage_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
