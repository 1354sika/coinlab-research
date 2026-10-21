# -*- coding: utf-8 -*-
"""
gambler_first_touch · 用「先触止盈」作标签，而不是「摸到止盈」
  y = 1  先触 +TP（且不先触 -SL）
  y = 0  先触 -SL 或 超时（24h 内都没到 TP）
对照：alpha_small 的标签是「24h 内摸到 +5%」，不管中间跌多少 —— 73.8% 最高档压线
目标：看模型能否预测真实交易目标「先触」
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
SL = 0.12
CAP = 5000     # 每币采样上限
STEP = 1

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_first_touch · 标签 = 先触止盈（不是摸到）')
    log('=' * 100)
    log(f'  TP=+{TP:.1%}  SL=-{SL:.1%}  HOLD={HOLD}min  入场=240min低点×{1+CONFIRM:.2f}  扣费另计')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    X_all, y_all, t_all = [], [], []

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

        y_ft = np.zeros(len(idxs), dtype=np.int8)
        for k, idx in enumerate(idxs):
            e = entry[idx]
            tp_p = e * (1 + TP)
            sl_p = e * (1 - SL)
            end = min(idx + HOLD, n - 1)
            if end <= idx + 1:
                continue
            seg_h = h[idx + 1:end + 1]
            seg_l = l[idx + 1:end + 1]
            hit_tp = np.where(seg_h >= tp_p)[0]
            hit_sl = np.where(seg_l <= sl_p)[0]
            i_tp = int(hit_tp[0]) if len(hit_tp) else 10 ** 9
            i_sl = int(hit_sl[0]) if len(hit_sl) else 10 ** 9
            # 先触止盈：TP 严格先于 SL（同根算 SL，保守）
            if i_tp < 10 ** 9 and i_tp < i_sl:
                y_ft[k] = 1

        feats = np.column_stack([
            r1[idxs], r5[idxs], r15[idxs], r60[idxs], r240[idxs],
            volr[idxs], tbr[idxs], vstd[idxs], vstd1440[idxs],
            hour[idxs], dow[idxs], pos[idxs],
        ])
        ok = np.isfinite(feats).all(axis=1)
        X_all.append(feats[ok]); y_all.append(y_ft[ok]); t_all.append(ot[idxs][ok])

        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(X_all); y = np.concatenate(y_all); tt = np.concatenate(t_all)
    order = np.argsort(tt)
    X, y = X[order], y[order]

    n = len(X)
    tr = int(n * 0.7)
    Xtr, Xte, ytr, yte = X[:tr], X[tr:], y[:tr], y[tr:]
    log(f'\n样本 {n:,}  训练 {tr:,}  测试 {n-tr:,}')
    log(f'  基线 先触止盈率 {y.mean():.1%}  测试段 {yte.mean():.1%}')

    mdl = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                         learning_rate=0.08, random_state=0)
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    log(f'\n【标签=先触止盈】测试段 AUC = {auc:.4f}   基线 {yte.mean():.1%}')

    for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        log(f'  P∈[{lo:.1f},{hi:.2f})  n={m.sum():>7,}  实际先触率 {yte[m].mean():>6.1%}')

    log('\n' + '=' * 100)
    log('【对照】alpha_small 用「摸到」标签：AUC 0.6211，最高档 73.8% 命中，但盈亏平衡 73.5% —— 压线')
    log('        本实验用「先触」标签：若 AUC 仍 >0.6 且高档命中 > 平衡线 10pp ⇒ 真实可交易')
    log('        盈亏平衡线 p* = (SL+FEE)/(TP+SL) = (12+0.5)/(5+12) = 73.5%')
    log('=' * 100)

    out = os.path.join(OUT, 'first_touch_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
