# -*- coding: utf-8 -*-
"""
gambler_first_touch2 · 一次跑两个变量：
  ① 止损 SL ∈ {-6%, -8%, -10%, -12%}  → 看平衡线变化后是否转正
  ② 阈值 P≥0.7 / 0.8 / 0.9            → 看模型挑得更狠能否更准
标签 = 先触止盈（不先触止损）
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
FEE = 0.005
CAP = 5000
SL_LIST = [-0.06, -0.08, -0.10, -0.12]

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_first_touch2 · 止损档 × 阈值档')
    log('=' * 100)
    log(f'  TP=+{TP:.1%}  HOLD={HOLD}min  入场=240min低点×{1+CONFIRM:.2f}  费={FEE:.2%}')
    log(f'  SL 档: {[f"{x:.0%}" for x in SL_LIST]}')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    buckets = {sl: {'X': [], 'y': [], 't': []} for sl in SL_LIST}

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
        T = ot[IDX]

        for sl in SL_LIST:
            y = np.zeros(len(IDX), dtype=np.int8)
            tp_p = entry[IDX] * (1 + TP)
            sl_p = entry[IDX] * (1 - abs(sl))
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
            buckets[sl]['X'].append(F)
            buckets[sl]['y'].append(y)
            buckets[sl]['t'].append(T)

        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    # 每个 SL 单独训练 + 评估
    log('\n' + '=' * 100)
    log('【结果】每个 SL 档：AUC + 分档命中率 + 平衡线')
    log('=' * 100)

    for sl in SL_LIST:
        b = buckets[sl]
        if not b['X']:
            continue
        X = np.vstack(b['X']); y = np.concatenate(b['y']); tt = np.concatenate(b['t'])
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

        p_star = (abs(sl) + FEE) / (TP + abs(sl))
        log(f'\n--- SL = {sl:.0%}  平衡线 p* = {p_star:.1%} ---')
        log(f'    样本 {n:,}  测试 {len(yte):,}  基线先触率 {yte.mean():.1%}  AUC = {auc:.4f}')

        for lo in [0.5, 0.6, 0.7, 0.8, 0.9]:
            m = p >= lo
            if m.sum() < 30:
                log(f'    P>={lo:.1f}  n={m.sum():>6,}  (样本不足)')
                continue
            hit = yte[m].mean()
            log(f'    P>={lo:.1f}  n={m.sum():>6,}  命中 {hit:>6.1%}  净期望 {hit*TP - (1-hit)*abs(sl) - FEE:>+6.2f}%  判 {"✓" if hit > p_star else "✗"}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('  · 净期望 = 命中×TP − (1−命中)×|SL| − 费')
    log('  · 若某 SL 档的 P>=0.7 净期望 > 0 且样本 > 100 ⇒ 这是第一个真实可交易配置')
    log('  · 若所有 SL 档都负 ⇒ 「先触 +5%」在 K 线层面做不到，需换 TP 或换数据')
    log('=' * 100)

    out = os.path.join(OUT, 'first_touch2_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
