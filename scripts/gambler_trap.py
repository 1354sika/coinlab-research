# -*- coding: utf-8 -*-
"""
gambler_trap · 专门识别「诱多」：反弹 3% 后 1 小时内又跌 2%
  目的：如果诱多可识别（AUC>0.65），就用「排除诱多」来提纯真信号
  与 gambler_newmodel 完全同一批触发点，仅标签不同 + 加 4 个诱多特征
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
TRAP_WIN = 60        # 诱多观察窗：60 根（1 小时）
TRAP_DROP = 0.02     # 跌回 -2% 即诱多

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

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

    # ---- 诱多专用 4 特征 ----
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


FEATNAMES = ['r1','r5','r15','r60','r240','volr','tbr','vstd','vstd1440',
             'hour','dow','pos','rebound','upper5','vdecay','rebound_z']


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_trap · 专门识别「诱多」')
    log('=' * 100)
    log(f'  触发：240min低点×{1+CONFIRM:.2f}  当根收盘价买入  CAP={CAP}/币')
    log(f'  诱多标签：触发后 {TRAP_WIN} 根内 最低价 <= 入场价×(1-{TRAP_DROP:.0%})')
    log(f'  真信号标签：先触+{TP:.0%} 且未先触{SL:.0%}（对照用）')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    Xs, yA, yD, ts = [], [], [], []

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
        IDX = idxs[ok]
        Fs = feats[ok]
        if len(IDX) == 0:
            continue

        px = c[IDX]
        tp_p = px * (1 + TP)
        sl_p = px * (1 - abs(SL))
        trap_p = px * (1 - TRAP_DROP)
        yA_k = np.zeros(len(IDX), dtype=np.int8)
        yD_k = np.zeros(len(IDX), dtype=np.int8)
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
            if i_tp < i_sl and i_tp < 10 ** 9:
                yA_k[k] = 1
            tw = min(TRAP_WIN, len(seg_l))
            if tw > 0 and seg_l[:tw].min() <= trap_p[k]:
                yD_k[k] = 1

        Xs.append(Fs); yA.append(yA_k); yD.append(yD_k)
        ts.append(ot[IDX])

        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(Xs)
    yA = np.concatenate(yA); yD = np.concatenate(yD)
    tt = np.concatenate(ts)
    order = np.argsort(tt)
    X = X[order]; yA = yA[order]; yD = yD[order]

    n = len(X)
    tr = int(n * 0.7)
    log(f'\n总触发点 {n:,}  训练 {tr:,}  测试 {n-tr:,}')
    log(f'  真信号基线 {yA.mean():.1%}   诱多基线 {yD.mean():.1%}')
    ov = float(((yA == 1) & (yD == 1)).sum()) / max(1, int(yA.sum()))
    log(f'  真信号中同时是诱多的比例 {ov:.1%}（>30% 说明两者高度重叠，反转无意义）')

    # ---------- 诱多模型 ----------
    log('\n' + '=' * 100)
    log('【一】诱多模型（GBDT）')
    log('=' * 100)
    mdlD = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                          learning_rate=0.07, random_state=0)
    mdlD.fit(X[:tr], yD[:tr])
    pD = mdlD.predict_proba(X[tr:])[:, 1]
    aucD = roc_auc_score(yD[tr:], pD)
    log(f'\n  AUC(诱多) = {aucD:.4f}   （0.5=无信息）')
    log(f'  {"阈值":>5} {"报出":>8} {"精确率":>7} {"召回率":>7}')
    for lo in THRESHOLDS:
        mm = pD >= lo
        if mm.sum() < 30:
            continue
        log(f'  {lo:>5.1f} {int(mm.sum()):>8,} {yD[tr:][mm].mean():>6.1%} '
            f'{yD[tr:][mm].sum()/max(1,yD[tr:].sum()):>6.1%}')

    # ---------- 真信号模型（同特征对照） ----------
    log('\n' + '=' * 100)
    log('【二】对照：真信号模型（相同 16 特征）')
    log('=' * 100)
    mdlA = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                          learning_rate=0.07, random_state=0)
    mdlA.fit(X[:tr], yA[:tr])
    pA = mdlA.predict_proba(X[tr:])[:, 1]
    aucA = roc_auc_score(yA[tr:], pA)
    p_star = (abs(SL) + FEE) / (TP + abs(SL))
    log(f'\n  AUC(真信号) = {aucA:.4f}   平衡线 p* = {p_star:.1%}')
    log(f'  {"阈值":>5} {"报出":>8} {"精确率":>7} {"召回率":>7} {"净期望":>9}')
    for lo in THRESHOLDS:
        mm = pA >= lo
        if mm.sum() < 30:
            continue
        prec = yA[tr:][mm].mean()
        rec = yA[tr:][mm].sum() / max(1, yA[tr:].sum())
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lo:>5.1f} {int(mm.sum()):>8,} {prec:>6.1%} {rec:>6.1%} {exp_:>+8.2%}')

    # ---------- 核心：排除诱多 ----------
    log('\n' + '=' * 100)
    log('【三】核心：排除「诱多」后，真信号精确率是否提升')
    log('=' * 100)
    base_prec = yA[tr:].mean()
    log(f'\n  不排除（全部买入）：真信号率 {base_prec:.1%}  '
        f'净期望 {base_prec*TP-(1-base_prec)*abs(SL)-FEE:+.2%}')
    log(f'  {"排除 pD>=":>10} {"剩余笔数":>9} {"真信号率":>9} {"净期望":>9} {"提升":>8}')
    for lo in THRESHOLDS:
        keep = pD < lo
        if keep.sum() < 30:
            continue
        prec = yA[tr:][keep].mean()
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lo:>10.1f} {int(keep.sum()):>9,} {prec:>8.1%} {exp_:>+8.2%} {prec-base_prec:>+7.1%}')

    # ---------- 双筛 ----------
    log('\n' + '=' * 100)
    log('【四】双筛：既像真信号、又不像诱多')
    log('=' * 100)
    log(f'  {"pA>=":>6} {"pD<":>6} {"报出":>8} {"真信号率":>9} {"召回":>7} {"净期望":>9}')
    for lo_a in [0.4, 0.5, 0.6, 0.7]:
        for lo_d in [0.4, 0.5, 0.6]:
            mm = (pA >= lo_a) & (pD < lo_d)
            if mm.sum() < 30:
                continue
            prec = yA[tr:][mm].mean()
            rec = yA[tr:][mm].sum() / max(1, yA[tr:].sum())
            exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
            log(f'  {lo_a:>6.1f} {lo_d:>6.1f} {int(mm.sum()):>8,} {prec:>8.1%} {rec:>6.1%} {exp_:>+8.2%}')

    # ---------- 特征重要性 ----------
    log('\n【五】诱多模型特征重要性（permutation 近似，越大越重要）')
    base = aucD
    imps = []
    rng = np.random.default_rng(0)
    for j, nm in enumerate(FEATNAMES):
        Xp = X[tr:].copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        try:
            auc_j = roc_auc_score(yD[tr:], mdlD.predict_proba(Xp)[:, 1])
        except Exception:
            auc_j = base
        imps.append((nm, base - auc_j))
    for nm, v in sorted(imps, key=lambda x: -abs(x[1]))[:10]:
        log(f'  {nm:12s} {v:+.4f}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('=' * 100)
    log('  · AUC(诱多) > 0.65 ⇒ 诱多可识别，排除它可能提纯')
    log('  · AUC(诱多) ≈ 0.5 ⇒ 1m K 线里没有「诱多」的痕迹')
    log('  · 真信号与诱多重叠率 > 30% ⇒ 两者是同一批波动时刻，反转无意义')
    log('  · 【三】某行「真信号率」显著高于基线 ⇒ 这就是新工具')
    log('  · 【四】双筛净期望 > 单用 pA ⇒ 两者互补，不是同一信息')

    out = os.path.join(OUT, 'trap_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
