# -*- coding: utf-8 -*-
"""
gambler_confirm_sweep · 扫描「低点+多少%」触发阈值（CONFIRM）
  固定：TP=+5% / SL=-6%（CLOSE 口径下唯一为正的止损）/ HOLD=24h / 费 0.5%
  输出：LOW 与 CLOSE 两口径 × 8 个 CONFIRM 档 × 4 个概率阈值
README:
  CONFIRM=0.03 即「低点+3%」；0.02 更早入场（价格更低，确认度更低）
"""
import os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = r'D:\中信证券\coinlab'
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
OUT = os.path.join(ROOT, 'evidence', 'cycle')

LOW_WIN = 240
HOLD = 1440
TP = 0.05
FEE = 0.005
CAP = 5000
CONFIRM_GRID = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
SL = -0.06
MODES = ['low', 'close']
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_confirm_sweep · 扫描触发阈值 CONFIRM（低点+多少%）')
    log('=' * 100)
    log(f'  TP=+{TP:.1%}  SL={SL:.0%}  HOLD={HOLD}min  费={FEE:.2%}  CAP={CAP}')
    log(f'  CONFIRM 网格 {CONFIRM_GRID}')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    # buckets[ci][em] = {'X':[], 'y':[], 't':[]}
    buckets = {ci: {em: {'X': [], 'y': [], 't': []} for em in MODES} for ci in range(len(CONFIRM_GRID))}

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

        base_ok = np.isfinite(lo_win) & np.isfinite(pos) & np.isfinite(vstd1440)
        base_ok[0] = False

        for ci, cv in enumerate(CONFIRM_GRID):
            entry = lo_win * (1 + cv)
            m = (c >= entry) & (o <= entry * 1.03) & base_ok
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

            for em in MODES:
                px = entry[IDX] if em == 'low' else c[IDX]
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
                buckets[ci][em]['X'].append(F)
                buckets[ci][em]['y'].append(y)
                buckets[ci][em]['t'].append(T)

        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    log('\n' + '=' * 100)
    log('【结果】CONFIRM 扫描 · TP+5%/SL-6%/24h · 费 0.5%')
    log('=' * 100)

    for em in MODES:
        log(f'\n########## 口径 = {em.upper()} ##########')
        for ci, cv in enumerate(CONFIRM_GRID):
            b = buckets[ci][em]
            if not b['X']:
                log(f'  CONFIRM={cv:.3f}  (无样本)')
                continue
            X = np.vstack(b['X']); y = np.concatenate(b['y']); tt = np.concatenate(b['t'])
            order = np.argsort(tt)
            X, y = X[order], y[order]
            n = len(X)
            if n < 200:
                log(f'  CONFIRM={cv:.3f}  n={n}  (样本不足)')
                continue
            tr = int(n * 0.7)
            Xtr, Xte, ytr, yte = X[:tr], X[tr:], y[:tr], y[tr:]

            mdl = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                                 learning_rate=0.08, random_state=0)
            mdl.fit(Xtr, ytr)
            p = mdl.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, p)
            p_star = (abs(SL) + FEE) / (TP + abs(SL))
            log(f'  CONFIRM={cv:.3f}  总 {n:,}  测试 {len(yte):,}  基线 {yte.mean():.1%}  AUC={auc:.4f}  平衡线 {p_star:.1%}')
            for lo in THRESHOLDS:
                mm = p >= lo
                if mm.sum() < 30:
                    log(f'      P>={lo:.1f}  n={mm.sum():>6,}  (不足)')
                    continue
                hit = yte[mm].mean()
                ne = hit * TP - (1 - hit) * abs(SL) - FEE
                tag = '✓' if ne > 0 else '✗'
                log(f'      P>={lo:.1f}  n={mm.sum():>6,}  命中 {hit:>6.1%}  净期望 {ne*100:>+6.2f}%  判{tag}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('  · CONFIRM 越小 = 越早入场（价格越便宜），但假信号越多')
    log('  · 只有 CLOSE 口径下净期望 > 0 才是可执行配置')
    log('=' * 100)

    out = os.path.join(OUT, 'confirm_sweep_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
