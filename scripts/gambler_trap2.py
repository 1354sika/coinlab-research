# -*- coding: utf-8 -*-
"""
gambler_trap2 · 正向利用「诱多概率」：高波动时刻是不是涨跌都更容易触发？
  与 trap 完全同一批触发点、同一模型，只把用法从「排除」改成「分档正向看」
  核心问题：pD 高的那批，真信号率是不是反而更高？
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
CAP = 1500
TRAP_WIN = 60
TRAP_DROP = 0.02

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def build_features(g):
    ot = g['open_time'].values.astype(np.int64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    c = g['close'].values.astype(np.float64)
    op = g['open'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    tb = g['taker_base'].values.astype(np.float64)
    n = len(c)
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
    rebound = np.where(lo_win > 0, c / np.maximum(lo_win, 1e-12) - 1, np.nan)
    upper = (h - np.maximum(c, op)) / np.maximum(h - l, 1e-12)
    upper5 = pd.Series(upper).rolling(5, min_periods=1).mean().values
    v5 = pd.Series(v).rolling(5, min_periods=1).mean().values
    vdecay = np.where(vbase > 0, v5 / np.maximum(vbase, 1e-12), np.nan)
    rebound_z = np.where(vstd > 1e-9, rebound / vstd, np.nan)
    F = np.column_stack([r1, r5, r15, r60, r240, volr, tbr,
                         vstd, vstd1440, hour, dow, pos,
                         rebound, upper5, vdecay, rebound_z])
    return F, lo_win, c, h, l, op, ot


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_trap2 · 正向利用诱多概率')
    log('=' * 100)

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())

    Xs, yA, yD, yB, ts = [], [], [], [], []
    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        if len(g) < LOW_WIN + HOLD + 10:
            continue
        F, lo_win, c, h, l, op, ot = build_features(g)
        n = len(c)
        entry = lo_win * (1 + CONFIRM)
        m = (c >= entry) & (op <= entry * 1.03)
        m[0] = False
        m = m & np.isfinite(lo_win) & np.isfinite(F[:, 11]) & np.isfinite(F[:, 8])
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > CAP:
            idxs = idxs[::max(1, len(idxs) // CAP)][:CAP]
        feats = F[idxs]
        ok = np.isfinite(feats).all(axis=1)
        IDX = idxs[ok]; Fs = feats[ok]
        if len(IDX) == 0:
            continue
        px = c[IDX]
        tp_p = px * (1 + TP); sl_p = px * (1 - abs(SL)); trap_p = px * (1 - TRAP_DROP)
        yA_k = np.zeros(len(IDX), dtype=np.int8)
        yD_k = np.zeros(len(IDX), dtype=np.int8)
        yB_k = np.zeros(len(IDX), dtype=np.int8)
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
            if i_tp < 10 ** 9:
                yB_k[k] = 1
            if i_tp < i_sl and i_tp < 10 ** 9:
                yA_k[k] = 1
            tw = min(TRAP_WIN, len(seg_l))
            if tw > 0 and seg_l[:tw].min() <= trap_p[k]:
                yD_k[k] = 1
        Xs.append(Fs); yA.append(yA_k); yD.append(yD_k); yB.append(yB_k)
        ts.append(ot[IDX])
        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(Xs)
    yA = np.concatenate(yA); yD = np.concatenate(yD); yB = np.concatenate(yB)
    tt = np.concatenate(ts)
    order = np.argsort(tt)
    X = X[order]; yA = yA[order]; yD = yD[order]; yB = yB[order]
    n = len(X); tr = int(n * 0.7)
    log(f'\n总触发点 {n:,}  训练 {tr:,}  测试 {n-tr:,}')
    log(f'  真信号基线 {yA.mean():.1%}  诱多基线 {yD.mean():.1%}  摸到+5%基线 {yB.mean():.1%}')

    p_star = (abs(SL) + FEE) / (TP + abs(SL))
    log(f'  平衡线 p* = {p_star:.1%}')

    # 诱多模型
    mdlD = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                          learning_rate=0.07, random_state=0)
    mdlD.fit(X[:tr], yD[:tr])
    pD = mdlD.predict_proba(X[tr:])[:, 1]
    log(f'\n  AUC(诱多) = {roc_auc_score(yD[tr:], pD):.4f}')

    # ============ 核心：按 pD 分档，看真信号率 ============
    log('\n' + '=' * 100)
    log('【一】核心：按诱多概率 pD 分档，看真信号率（越高的档是不是越好）')
    log('=' * 100)
    log(f'  {"pD档":>12} {"笔数":>8} {"真信号率":>9} {"摸到+5%率":>10} {"净期望":>9} {"vs基线":>8}')
    edges = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.45),
             (0.45, 0.60), (0.60, 1.01)]
    base = yA[tr:].mean()
    for lo, hi in edges:
        mm = (pD >= lo) & (pD < hi)
        if mm.sum() < 30:
            continue
        prec = yA[tr:][mm].mean()
        hitB = yB[tr:][mm].mean()
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  [{lo:>4.2f},{hi:>4.2f}) {int(mm.sum()):>8,} {prec:>8.1%} {hitB:>9.1%} {exp_:>+8.2%} {prec-base:>+7.1%}')

    # ============ 只做高档：pD 越大越买 ============
    log('\n' + '=' * 100)
    log('【二】只做高档：pD >= 阈值（与「排除」完全相反）')
    log('=' * 100)
    log(f'  {"pD>=":>6} {"笔数":>8} {"真信号率":>9} {"召回":>7} {"净期望":>9}')
    for lo in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mm = pD >= lo
        if mm.sum() < 30:
            continue
        prec = yA[tr:][mm].mean()
        rec = yA[tr:][mm].sum() / max(1, yA[tr:].sum())
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lo:>6.1f} {int(mm.sum()):>8,} {prec:>8.1%} {rec:>6.1%} {exp_:>+8.2%}')

    # ============ 双高：pD 高 + pA 高 ============
    log('\n' + '=' * 100)
    log('【三】双高：pD 高 且 pA 高（波动大 + 像真信号）')
    log('=' * 100)
    mdlA = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                          learning_rate=0.07, random_state=0)
    mdlA.fit(X[:tr], yA[:tr])
    pA = mdlA.predict_proba(X[tr:])[:, 1]
    log(f'\n  AUC(真信号) = {roc_auc_score(yA[tr:], pA):.4f}')
    log(f'  {"pA>=":>5} {"pD>=":>5} {"笔数":>8} {"真信号率":>9} {"召回":>7} {"净期望":>9}')
    for la in [0.3, 0.4, 0.5, 0.6]:
        for ld in [0.2, 0.3, 0.4, 0.5]:
            mm = (pA >= la) & (pD >= ld)
            if mm.sum() < 30:
                continue
            prec = yA[tr:][mm].mean()
            rec = yA[tr:][mm].sum() / max(1, yA[tr:].sum())
            exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
            log(f'  {la:>5.1f} {ld:>5.1f} {int(mm.sum()):>8,} {prec:>8.1%} {rec:>6.1%} {exp_:>+8.2%}')

    # ============ 单看 vstd(原特征) 的分档对照 ============
    log('\n' + '=' * 100)
    log('【四】对照：直接用 vstd（波动率）分档，与 pD 分档对比')
    log('=' * 100)
    vs = X[tr:, 7]
    qs = np.nanpercentile(vs, [20, 40, 60, 80])
    log(f'  vstd 分位: {qs[0]:.5f} {qs[1]:.5f} {qs[2]:.5f} {qs[3]:.5f}')
    log(f'  {"vstd档":>14} {"笔数":>8} {"真信号率":>9} {"净期望":>9}')
    bounds = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]),
              (qs[2], qs[3]), (qs[3], np.inf)]
    labels = ['<p20', 'p20~p40', 'p40~p60', 'p60~p80', '>p80']
    for (lo, hi), lb in zip(bounds, labels):
        mm = (vs >= lo) & (vs < hi)
        if mm.sum() < 30:
            continue
        prec = yA[tr:][mm].mean()
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lb:>14} {int(mm.sum()):>8,} {prec:>8.1%} {exp_:>+8.2%}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('=' * 100)
    log('  · 【一】若高档真信号率显著高于基线 ⇒ 诱多概率是正向信号，不是要排除的')
    log('  · 【二】若某档净期望 > 0 ⇒ 这就是可交易配置')
    log('  · 【三】双高若优于单高 ⇒ 两者互补')
    log('  · 【四】若 vstd 分档与 pD 分档走势一致 ⇒ pD 本质就是波动率')

    out = os.path.join(OUT, 'trap2_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
