"""
gambler_buy_model2.py · 「该买」模型 v2

标签：A1 触发后，1440 根内先触 entry×1.05 = 1；先触 entry×0.98 或未到 = 0
特征：触发时刻前 60 根 × 7 通道 → 42 维（含 taker_base / taker_quote）
切分：按触发时刻排序，前 67% 训练、后 33% 测试（无前视）
评估：AUC + 分档命中率 + 每档净期望（扣 0.50%/笔）
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

SEQ = 60
A1_WIN = 240
A1_CONFIRM = 1.03
TP = 0.05
SL = 0.02
HOLD = 1440
FEE = 0.005
N_TRIGS = 30000

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

log = lambda s='': print(s, flush=True)


def build_feat(o, h, l, c, v, tb, tq, i, seq=SEQ):
    if i < seq:
        return None
    base = max(float(c[i-1]), 1e-12)
    o64 = o[i-seq:i].astype(np.float64) / base - 1.0
    h64 = h[i-seq:i].astype(np.float64) / base - 1.0
    l64 = l[i-seq:i].astype(np.float64) / base - 1.0
    c64 = c[i-seq:i].astype(np.float64) / base - 1.0
    pv = float(np.median(v[i-seq:i].astype(np.float64))) + 1e-12
    ptb = float(np.median(tb[i-seq:i].astype(np.float64))) + 1e-12
    ptq = float(np.median(tq[i-seq:i].astype(np.float64))) + 1e-12
    v64 = np.log1p(np.maximum(v[i-seq:i].astype(np.float64) / pv, 0.0))
    tb64 = np.log1p(np.maximum(tb[i-seq:i].astype(np.float64) / ptb, 0.0))
    tq64 = np.log1p(np.maximum(tq[i-seq:i].astype(np.float64) / ptq, 0.0))
    s = np.clip(np.stack([o64, h64, l64, c64, v64, tb64, tq64], axis=1), -50.0, 50.0)
    if not np.all(np.isfinite(s)):
        return None
    m = s.mean(axis=0); sd = s.std(axis=0)
    mx = s.max(axis=0); mn = s.min(axis=0)
    last = s[-1]
    slope = s[-30:].mean(axis=0) - s[:30].mean(axis=0)
    out = np.concatenate([m, sd, mx, mn, last, slope]).astype(np.float32)
    return out if np.all(np.isfinite(out)) else None


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_buy_model2 · 「该买」模型 v2（42 维，含 taker）')
    log('=' * 100)
    log(f'  入场：{A1_WIN}min 低点 × {A1_CONFIRM}  当根收盘价')
    log(f'  标签：{HOLD} 根内先触 +{TP:.0%} = 1；先触 -{SL:.0%} 或未到 = 0')
    log(f'  成本 {FEE:.2%}/笔   平衡线 p* = (SL+FEE)/(TP+SL) = {(SL+FEE)/(TP+SL):.1%}')
    log()

    log('读数据...')
    dtypes = {'symbol': 'category', 'open_time': 'int64',
              'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
              'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())
    rng = np.random.default_rng(53)

    log('\n[1/3] 找 A1 触发点 + 构特征 + 打标签...')
    TRIGS = []
    CACHE = {}
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < A1_WIN + SEQ + 300:
            continue
        CACHE[sym] = (g.open.values.astype(np.float32), g.high.values.astype(np.float32),
                      g.low.values.astype(np.float32), g.close.values.astype(np.float32),
                      g.volume.values.astype(np.float32),
                      g.taker_base.values.astype(np.float32), g.taker_quote.values.astype(np.float32),
                      g.open_time.values)
        o, h, l, c, v, tb, tq, t = CACHE[sym]
        cand = []
        for i in range(A1_WIN + SEQ + 1, n - 2):
            if c[i] >= l[i-A1_WIN:i].min() * A1_CONFIRM:
                cand.append(i)
        if len(cand) > 800:
            cand = list(rng.choice(cand, size=800, replace=False))
        for i in cand:
            TRIGS.append((sym, int(i), int(t[i])))
        if (si+1) % 15 == 0:
            log(f'  {si+1}/{len(syms)}  累计 {len(TRIGS):,}  {time.time()-t0:.0f}s')
    log(f'  共 {len(TRIGS):,}')

    if len(TRIGS) > N_TRIGS:
        sel = rng.choice(len(TRIGS), size=N_TRIGS, replace=False)
        TRIGS = [TRIGS[k] for k in sel]
        log(f'  采样 {len(TRIGS):,}')

    TRIGS.sort(key=lambda x: x[2])

    X, y, tarr = [], [], []
    for k, (sym, i, tm) in enumerate(TRIGS):
        o, h, l, c, v, tb, tq, _ = CACHE[sym]
        n = len(c)
        entry = float(c[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        f = build_feat(o, h, l, c, v, tb, tq, i)
        if f is None:
            continue
        tp_px = entry * (1 + TP); sl_px = entry * (1 - SL)
        lab = 0
        for m in range(i+1, min(i+1+HOLD, n)):
            if h[m] >= tp_px:
                lab = 1; break
            if l[m] <= sl_px:
                lab = 0; break
        X.append(f); y.append(lab); tarr.append(tm)
        if (k+1) % 5000 == 0:
            log(f'  {k+1}/{len(TRIGS)}  {time.time()-t0:.0f}s')

    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int8)
    log(f'\n样本 {len(X):,}  正样本率 {y.mean():.1%}')
    log(f'时间 {pd.to_datetime(int(tarr[0]), unit="ms", utc=True)} ~ '
        f'{pd.to_datetime(int(tarr[-1]), unit="ms", utc=True)}')

    tr_end = int(len(X) * 0.67)
    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xte, yte = X[tr_end:], y[tr_end:]
    log(f'训练 {len(Xtr):,}  测试 {len(Xte):,}  测试基线 {yte.mean():.1%}')

    log('\n[2/3] 训练 GBDT...')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=42)
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    log(f'  测试 AUC = {auc:.4f}   基线 {yte.mean():.1%}')

    log('\n[3/3] 分档 + 净期望...')
    log(f'  {"P区间":<14}{"n":>8}{"占比":>8}{"命中率":>9}{"净期望%":>10}{"判定":>5}')
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            hit = yte[m].mean()
            ev = hit * (TP - FEE) - (1 - hit) * (SL + FEE)
            mark = '✓' if ev > 0 else '✗'
            log(f'  [{lo:.1f},{hi:.2f})    {m.sum():>7,}  {m.mean():>7.1%}  {hit:>8.1%}'
                f'  {ev*100:>+9.2f}  {mark:>5}')

    log('\n【累积口径（P>=阈值）】')
    log(f'  {"阈值":<8}{"报出":>8}{"命中率":>9}{"净期望%":>10}{"判定":>5}')
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = p >= th
        if m.sum() > 0:
            hit = yte[m].mean()
            ev = hit * (TP - FEE) - (1 - hit) * (SL + FEE)
            mark = '✓' if ev > 0 else '✗'
            log(f'  >={th:.1f}    {m.sum():>7,}  {hit:>8.1%}  {ev*100:>+9.2f}  {mark:>5}')

    log('\n【判读】')
    log('  · 累积 P>=0.5 若净期望 > 0 ⇒ 该买模型能筛出可交易子集')
    log('  · 若全负 ⇒ 42 维 K 线特征不足以筛出正期望入场')

    log(f'\nTOTAL {time.time()-t0:.0f}s')
    np.save(os.path.join(OUT, 'buy_model2_y.npy'), yte)
    np.save(os.path.join(OUT, 'buy_model2_p.npy'), p)
    log(f'已存 buy_model2_y/p.npy → {OUT}')


if __name__ == '__main__':
    main()
