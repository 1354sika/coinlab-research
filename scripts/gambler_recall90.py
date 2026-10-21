# -*- coding: utf-8 -*-
"""
gambler_recall90 · 目标：召回 >=90% 时，精确率尽可能高
  与 trap2 完全同一批触发点（240min低点×1.03、当根收盘价买入）
  标签：先触 +5% 且未先触 -6%
  改动：把特征从 16 维扩到 68 维 —— 用 240 根序列的采样路径 + 多尺度 + 结构 + 加速度
  评估：TPR(召回) >= 0.90 时的精确率、以及报出上限 11,428 下能抓到多少真信号
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
SEQ = 240          # 序列窗口
NSAMP = 24         # 路径采样点数

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
    hi1440 = pd.Series(h).rolling(1440, min_periods=60).max().values

    def r(k):
        a = np.zeros(n)
        if k < n:
            a[k:] = c[k:] / c[:-k] - 1
        return a

    r1, r5, r15, r30 = r(1), r(5), r(15), r(30)
    r60, r120, r240 = r(60), r(120), r(240)

    vbase = pd.Series(v).rolling(288, min_periods=60).mean().values
    volr = np.where(vbase > 0, v / np.maximum(vbase, 1e-12), np.nan)
    tbr = np.where(v > 0, tb / np.maximum(v, 1e-12), np.nan)

    rets = r1[1:]
    vstd60 = np.concatenate([[np.nan], pd.Series(rets).rolling(60, min_periods=30).std().values])
    vstd240 = np.concatenate([[np.nan], pd.Series(rets).rolling(240, min_periods=120).std().values])
    vstd1440 = np.concatenate([[np.nan], pd.Series(rets).rolling(1440, min_periods=240).std().values])

    pos = np.where(hi1440 > 0, c / hi1440, np.nan)
    rebound = np.where(lo_win > 0, c / np.maximum(lo_win, 1e-12) - 1, np.nan)
    hour = ((ot // 3_600_000) % 24).astype(np.float64)
    dow = ((ot // 86_400_000) % 7).astype(np.float64)

    # 加速度
    acc1 = r15 - r60 / 4.0
    acc2 = r60 - r240 / 4.0

    # 序列路径：240 根上采样 NSAMP 个点，价格/成交量归一化
    cS = pd.Series(c)
    vS = pd.Series(v)
    step = SEQ // NSAMP  # 10
    roll_c = cS.rolling(step, min_periods=1).mean()
    roll_v = vS.rolling(step, min_periods=1).mean()

    path_c = np.full((n, NSAMP), np.nan)
    path_v = np.full((n, NSAMP), np.nan)
    for k in range(NSAMP):
        # 第 k 个采样点 = 当前往前的 (NSAMP-1-k)*step 根
        shift = (NSAMP - 1 - k) * step
        if shift == 0:
            path_c[:, k] = c
            path_v[:, k] = v
        else:
            path_c[:, k] = roll_c.shift(shift).values
            path_v[:, k] = roll_v.shift(shift).values
    # 归一化：价格相对当前收盘；量相对窗口均值
    path_c = path_c / np.maximum(c[:, None], 1e-12) - 1.0
    vmean_win = np.nanmean(path_v, axis=1, keepdims=True)
    path_v = path_v / np.maximum(vmean_win, 1e-12)

    # 窗口内结构
    hi_seq = pd.Series(h).rolling(SEQ, min_periods=60).max().values
    lo_seq = pd.Series(l).rolling(SEQ, min_periods=60).min().values
    pos_in_seq = np.where(hi_seq > lo_seq,
                          (c - lo_seq) / np.maximum(hi_seq - lo_seq, 1e-12), np.nan)
    dd_seq = np.where(hi_seq > 0, c / hi_seq - 1, np.nan)

    base = np.column_stack([r1, r5, r15, r30, r60, r120, r240,
                            volr, tbr, vstd60, vstd240, vstd1440,
                            pos, rebound, hour, dow, acc1, acc2,
                            pos_in_seq, dd_seq])
    F = np.column_stack([base, path_c, path_v])
    return F, lo_win, c, h, l, op, ot


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_recall90 · 召回>=90% 时最大化精确率')
    log('=' * 100)
    log(f'  触发：240min低点×{1+CONFIRM:.2f} 当根收盘价  CAP={CAP}/币')
    log(f'  标签：先触+{TP:.0%} 且未先触{SL:.0%}')
    log(f'  特征：20 基础 + {NSAMP} 价格路径 + {NSAMP} 量能路径 = {20+2*NSAMP} 维')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    Xs, ys, ts = [], [], []
    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        if len(g) < LOW_WIN + HOLD + 10:
            continue
        F, lo_win, c, h, l, op, ot = build_features(g)
        n = len(c)
        entry = lo_win * (1 + CONFIRM)
        m = (c >= entry) & (op <= entry * 1.03)
        m[0] = False
        m = m & np.isfinite(lo_win) & np.isfinite(F[:, 12]) & np.isfinite(F[:, 11])
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
        tp_p = px * (1 + TP); sl_p = px * (1 - abs(SL))
        y = np.zeros(len(IDX), dtype=np.int8)
        for k in range(len(IDX)):
            idx = IDX[k]
            end = min(idx + HOLD, n - 1)
            if end <= idx + 1:
                continue
            sh = h[idx + 1:end + 1]; sl = l[idx + 1:end + 1]
            ht = np.where(sh >= tp_p[k])[0]
            hs = np.where(sl <= sl_p[k])[0]
            it = int(ht[0]) if len(ht) else 10 ** 9
            isl = int(hs[0]) if len(hs) else 10 ** 9
            if it < isl and it < 10 ** 9:
                y[k] = 1
        Xs.append(Fs); ys.append(y); ts.append(ot[IDX])
        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(Xs); y = np.concatenate(ys); tt = np.concatenate(ts)
    order = np.argsort(tt)
    X = X[order]; y = y[order]
    n = len(X); tr = int(n * 0.7)
    log(f'\n总触发点 {n:,}  训练 {tr:,}  测试 {n-tr:,}')
    log(f'  真信号基线 训练 {y[:tr].mean():.1%}  测试 {y[tr:].mean():.1%}')
    log(f'  特征维数 {X.shape[1]}')

    # 训练（加大模型容量 + 更多迭代）
    mdl = HistGradientBoostingClassifier(
        max_iter=800, max_depth=8, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12, random_state=0)
    mdl.fit(X[:tr], y[:tr])
    p = mdl.predict_proba(X[tr:])[:, 1]
    yt = y[tr:]
    auc = roc_auc_score(yt, p)
    log(f'\n  AUC = {auc:.4f}   （对照：trap2 真信号 0.5586 / LSTM 0.6120）')

    # ---------- 核心：高召回区的精确率 ----------
    log('\n' + '=' * 100)
    log('【一】高召回区：召回 >= 90% 时的精确率（这是硬目标）')
    log('=' * 100)
    log(f'  测试真信号总数 {int(yt.sum()):,}')
    log(f'  {"目标召回":>8} {"阈值":>7} {"报出":>8} {"抓到真":>7} {"精确率":>7} {"净期望":>9} {"错报":>7}')
    npos = int(yt.sum())
    for target in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70]:
        # 找阈值使召回 >= target
        order_p = np.argsort(-p)
        cum = np.cumsum(yt[order_p])
        need = int(np.ceil(target * npos))
        if need > len(cum):
            continue
        k = int(np.searchsorted(cum, need))  # 报前 k+1 个即可达到
        k = min(k + 1, len(p))
        thr = p[order_p][k - 1]
        rep = p >= thr
        caught = int(yt[rep].sum())
        prec = caught / max(1, rep.sum())
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {target:>8.0%} {thr:>7.3f} {int(rep.sum()):>8,} {caught:>7,} '
            f'{prec:>6.1%} {exp_:>+8.2%} {int(rep.sum())-caught:>7,}')

    # ---------- 报出上限 11,428 时能抓多少 ----------
    log('\n' + '=' * 100)
    log('【二】按你的硬约束：最多报 11,428 个（错误率<=30%）')
    log('=' * 100)
    order_p = np.argsort(-p)
    K = min(11428, len(p))
    top = order_p[:K]
    caught = int(yt[top].sum())
    prec = caught / K
    exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
    log(f'  报出 {K:,}  其中真 {caught:,}  假 {K-caught:,}')
    log(f'  精确率 {prec:.1%}  （要求 >=70%）  召回 {caught/npos:.1%}  （要求 >=88.8%）')
    log(f'  单笔净期望 {exp_:+.2%}  （要求 >0）')
    log(f'  判定：{"✓ 达标" if (prec >= 0.70 and caught/npos >= 0.888 and exp_ > 0) else "✗ 未达标"}')

    # ---------- 对照：常规阈值分档 ----------
    log('\n' + '=' * 100)
    log('【三】对照：常规阈值分档（看形状）')
    log('=' * 100)
    log(f'  {"阈值":>6} {"报出":>8} {"真信号率":>9} {"召回":>7} {"净期望":>9}')
    for lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mm = p >= lo
        if mm.sum() < 30:
            continue
        prec = yt[mm].mean()
        rec = yt[mm].sum() / max(1, npos)
        exp_ = prec * TP - (1 - prec) * abs(SL) - FEE
        log(f'  {lo:>6.1f} {int(mm.sum()):>8,} {prec:>8.1%} {rec:>6.1%} {exp_:>+8.2%}')

    # ---------- 特征重要性 ----------
    log('\n【四】特征重要性 Top12（permutation 近似）')
    imps = []
    rng = np.random.default_rng(0)
    sub = np.random.default_rng(1).choice(len(yt), size=min(6000, len(yt)), replace=False)
    Xs_sub = X[tr:][sub]; ys_sub = yt[sub]
    base = roc_auc_score(ys_sub, mdl.predict_proba(Xs_sub)[:, 1])
    names = (['r1','r5','r15','r30','r60','r120','r240','volr','tbr',
              'vstd60','vstd240','vstd1440','pos','rebound','hour','dow',
              'acc1','acc2','pos_in_seq','dd_seq']
             + [f'pc{i}' for i in range(NSAMP)] + [f'pv{i}' for i in range(NSAMP)])
    for j in range(X.shape[1]):
        Xp = Xs_sub.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        try:
            a = roc_auc_score(ys_sub, mdl.predict_proba(Xp)[:, 1])
        except Exception:
            a = base
        imps.append((names[j] if j < len(names) else f'f{j}', base - a))
    for nm, v in sorted(imps, key=lambda x: -abs(x[1]))[:12]:
        log(f'  {nm:12s} {v:+.4f}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('=' * 100)
    log('  · 【一】召回 90% 时精确率若 < 40% ⇒ 现有特征下这个目标不可达')
    log('  · 【二】若判定 ✗ ⇒ 明确差在哪一列（精确率/召回/期望）')
    log('  · 【三】看曲线形状：阈值降一点，精确率掉多少')
    log('  · 【四】若路径特征(pc/pv)进 Top ⇒ 序列形状确实有信息')

    out = os.path.join(OUT, 'recall90_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
