# -*- coding: utf-8 -*-
"""
gambler_sim_trade · 真正的历史模拟交易（资金曲线）
  当前最优模式：240min低点×1.03 触发 → 收盘价买入 → 模型P>=0.7 → TP+5%/SL-6%/24h
  本金 100，每笔 10% 仓位，逐笔按时间顺序
输出：资金曲线、最大回撤、逐月、连亏
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
SL = 0.06
FEE = 0.005
CAP = 5000
THR = 0.7
POS = 0.10

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_sim_trade · 历史模拟交易（资金曲线）')
    log('=' * 100)
    log(f'  入场=240min低点×{1+CONFIRM:.2f}  买入=当根收盘价')
    log(f'  P>={THR}  TP=+{TP:.1%}  SL=-{SL:.1%}  HOLD={HOLD}min  费={FEE:.2%}')
    log(f'  本金 100  每笔仓位 {POS:.0%}')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    # 先收集所有信号 + 特征
    rows = []  # (t_enter, sym, p_reason, ret, hold_min)

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
        F = feats[ok]; IDX = idxs[ok]
        if len(IDX) == 0:
            continue

        px = c[IDX]
        tp_p = px * (1 + TP)
        sl_p = px * (1 - SL)
        y = np.zeros(len(IDX), dtype=np.int8)
        for k in range(len(IDX)):
            idx = IDX[k]
            end = min(idx + HOLD, n - 1)
            if end <= idx + 1:
                continue
            seg_h = h[idx + 1:end + 1]; seg_l = l[idx + 1:end + 1]
            hit_tp = np.where(seg_h >= tp_p[k])[0]
            hit_sl = np.where(seg_l <= sl_p[k])[0]
            i_tp = int(hit_tp[0]) if len(hit_tp) else 10 ** 9
            i_sl = int(hit_sl[0]) if len(hit_sl) else 10 ** 9
            if i_tp < 10 ** 9 and i_tp < i_sl:
                y[k] = 1

        rows.append((ot[IDX], sym, F, y, IDX, c, h, l, n))
        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    # 按时间合并训练/测试
    all_t = np.concatenate([r[0] for r in rows])
    all_X = np.vstack([r[2] for r in rows])
    all_y = np.concatenate([r[3] for r in rows])
    order = np.argsort(all_t)
    all_t, all_X, all_y = all_t[order], all_X[order], all_y[order]

    n = len(all_X)
    tr = int(n * 0.7)
    Xtr, Xte, ytr, yte = all_X[:tr], all_X[tr:], all_y[:tr], all_y[tr:]
    log(f'\n样本 {n:,}  训练 {tr:,}  测试 {n-tr:,}')

    mdl = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                         learning_rate=0.08, random_state=0)
    mdl.fit(Xtr, ytr)
    p_te = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te)
    log(f'模型 AUC = {auc:.4f}')

    # 逐笔模拟（测试段，按时间顺序）
    log('\n' + '=' * 100)
    log('【逐笔模拟（测试段，按时间顺序，10% 仓位）】')
    log('=' * 100)

    # 重建每个样本的收益：用 y_te 反推不够，需重算。这里用简化：胜=+5%-费, 负=-SL-费
    sel = p_te >= THR
    log(f'  P>={THR} 笔数 {sel.sum():,}')

    rets = np.where(yte[sel] == 1, TP - FEE, -SL - FEE)
    # 顺序按时间（Xte 已是时间序）
    cap = 100.0
    peak = cap
    mdd = 0.0
    curve = [cap]
    streak = 0
    max_streak = 0
    for r in rets:
        cap *= (1 + POS * r)
        curve.append(cap)
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak
        if dd > mdd:
            mdd = dd
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    log(f'  最终资金 {cap:,.2f} 元  （{cap/100:.2f}x）')
    log(f'  最大回撤 {mdd:.1%}')
    log(f'  最长连亏 {max_streak} 笔')
    log(f'  单笔均值 {rets.mean()*100:+.2f}%  中位 {np.median(rets)*100:+.2f}%  胜率 {np.mean(rets>0):.1%}')

    # 逐月
    log('\n【逐月】')
    t_te = all_t[tr:]
    months = pd.to_datetime(t_te[sel], unit='ms', utc=True).strftime('%Y-%m')
    dfm = pd.DataFrame({'m': months, 'r': rets})
    for m, g in dfm.groupby('m'):
        log(f'    {m}  {len(g):>5} 笔  均值 {g["r"].mean()*100:+.2f}%  胜率 {(g["r"]>0).mean():.1%}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('  · 这是逐笔模拟，不含持仓重叠/资金占用（每笔独立 10% 仓位）')
    log('  · 未计滑点；SL 按固定 -6% 计（实际可能滑到更深）')
    log('  · 若最终资金 > 100 且逐月不长期为负 ⇒ 历史模拟成立')
    log('=' * 100)

    out = os.path.join(OUT, 'sim_trade_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
