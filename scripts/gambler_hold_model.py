"""
gambler_hold_model.py · 「该持」模型

标签：从持仓时刻 j 出发，继续持有到 entry×1.05（先触） = 1
      先触 entry×0.98 或 1440 根内未到 = 0
特征：当前 60 根 × 7 通道统计量（42 维）+ 浮盈 + 已持有分钟 = 44 维
切分：按入场时刻排序，前 67% 入场点训练，后 33% 测试
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

SEQ = 60
CH = 7
A1_WIN = 240
A1_CONFIRM = 1.03
TP = 0.05
SL = 0.02
HOLD = 1440
N_ENTRY_TOTAL = 6000
STEPS_PER_ENTRY = 12

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
    log('gambler_hold_model · 「该持」模型')
    log('=' * 100)
    log(f'  入场：{A1_WIN}min 低点 × {A1_CONFIRM}  当根收盘价')
    log(f'  标签：从持仓时刻 j 出发，1440 根内先触 entry×{1+TP:.2f} = 1；'
        f'先触 entry×{1-SL:.2f} 或未到 = 0')
    log(f'  特征：60 根 × 7 通道 → 42 维 + 浮盈 + 已持有分钟 = 44 维')
    log()

    log('读数据...')
    dtypes = {'symbol': 'category', 'open_time': 'int64',
              'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
              'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())
    rng = np.random.default_rng(31)

    log('\n[1/3] 找 A1 触发点...')
    TRIGS = []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < A1_WIN + SEQ + 200:
            continue
        l = g.low.values.astype(np.float32)
        c = g.close.values.astype(np.float32)
        for i in range(A1_WIN + SEQ + 1, n - 1):
            lo = l[i-A1_WIN:i].min()
            if c[i] >= lo * A1_CONFIRM:
                TRIGS.append((sym, i))
        if (si+1) % 15 == 0:
            log(f'    {sym}  {si+1}/{len(syms)}  累计 {len(TRIGS):,}  {time.time()-t0:.0f}s')
    log(f'  共 {len(TRIGS):,} 个触发点')

    if len(TRIGS) > N_ENTRY_TOTAL:
        sel = rng.choice(len(TRIGS), size=N_ENTRY_TOTAL, replace=False)
        TRIGS = [TRIGS[k] for k in sel]
        log(f'  采样 {len(TRIGS):,}')

    t_by_sym = {s: df[df.symbol == s].sort_values('open_time').open_time.values for s in syms}
    TRIGS.sort(key=lambda x: int(t_by_sym[x[0]][x[1]]))
    tr_end = int(len(TRIGS) * 0.67)
    log(f'  训练/测试切分索引 {tr_end}')

    log('\n[2/3] 构样本...')
    Xtr, ytr, Xte, yte = [], [], [], []
    CACHE = {}
    for k, (sym, i) in enumerate(TRIGS):
        if sym not in CACHE:
            g = df[df.symbol == sym].sort_values('open_time')
            CACHE[sym] = (
                g.open.values.astype(np.float32), g.high.values.astype(np.float32),
                g.low.values.astype(np.float32), g.close.values.astype(np.float32),
                g.volume.values.astype(np.float32),
                g.taker_base.values.astype(np.float32), g.taker_quote.values.astype(np.float32),
            )
        o, h, l, c, v, tb, tq = CACHE[sym]
        n = len(c)

        entry = float(c[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp_px = entry * (1 + TP)
        sl_px = entry * (1 - SL)

        end_j = min(i + HOLD, n - 1)
        if end_j - i < 60:
            continue
        js = np.unique(np.linspace(i + 10, end_j - 10, STEPS_PER_ENTRY).astype(int))

        for j in js:
            f = build_feat(o, h, l, c, v, tb, tq, j)
            if f is None:
                continue
            unreal = float(c[j]) / entry - 1.0
            held = float(j - i)
            feat = np.concatenate([f, [unreal, held]]).astype(np.float32)

            lab = 0
            for m in range(j + 1, min(j + HOLD, n)):
                if h[m] >= tp_px:
                    lab = 1
                    break
                if l[m] <= sl_px:
                    lab = 0
                    break

            if k < tr_end:
                Xtr.append(feat); ytr.append(lab)
            else:
                Xte.append(feat); yte.append(lab)

        if (k+1) % 500 == 0:
            log(f'    {k+1}/{len(TRIGS)}  {time.time()-t0:.0f}s')

    Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr, dtype=np.int8)
    Xte = np.asarray(Xte, dtype=np.float32); yte = np.asarray(yte, dtype=np.int8)
    log(f'\n训练 {len(Xtr):,}  正样本 {ytr.mean():.1%}')
    log(f'测试 {len(Xte):,}  正样本 {yte.mean():.1%}')

    log('\n[3/3] 训练 GBDT...')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=6,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=42)
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    log(f'  测试 AUC = {auc:.4f}   基线 {yte.mean():.1%}')

    log('\n【分档】')
    log(f'  {"P区间":<14}{"n":>9}{"继续能到+5%率":>15}{"提升":>8}')
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            log(f'  [{lo:.1f},{hi:.2f})    {m.sum():>8,}  {yte[m].mean():>14.1%}'
                f'  {yte[m].mean()/max(yte.mean(),1e-9):>7.2f}x')

    log('\n【按浮盈分档：看浮盈是否真的带信息】')
    unreal_s = Xte[:, -2]
    for lo, hi in [(-99, -0.02), (-0.02, 0), (0, 0.01), (0.01, 0.03), (0.03, 0.05), (0.05, 99)]:
        m = (unreal_s >= lo) & (unreal_s < hi)
        if m.sum() > 0:
            log(f'  浮盈[{lo:+.0%},{hi:+.0%})  n={m.sum():>7,}  继续能到+5%率 {yte[m].mean():>6.1%}')

    log('\n【判读】')
    log('  · AUC > 0.75 ⇒ 「该持」可作，且比「该买」好做')
    log('  · 浮盈分档若单调 ⇒ 浮盈是有效特征（这正是「该卖」缺的东西）')
    log('  · 若 P 高档的「能到+5%率」显著 > 基线 ⇒ 可交易')

    log(f'\nTOTAL {time.time()-t0:.0f}s')
    np.save(os.path.join(OUT, 'hold_model_y.npy'), yte)
    np.save(os.path.join(OUT, 'hold_model_p.npy'), p)
    log(f'已存 hold_model_y/p.npy → {OUT}')


if __name__ == '__main__':
    main()
