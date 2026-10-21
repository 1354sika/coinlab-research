"""
gambler_sell_model.py · 「该卖」模型：接下来 30 分钟内会不会跌 >=1.5%

标签：future 30min low < close[i-1] * (1 - DROP)
特征：前 SEQ 根的 7 通道（O/H/L/C/V/taker_base/taker_quote），每通道 6 统计量 = 42 维
切分：按时间 67/33
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SEQ = 60
CH = 7
HORIZON = 30        # 30 分钟内
DROP = 0.015        # 跌 1.5%
CAP_PER_SYM = 1200  # 每币最多采样

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle'


def log(s=''):
    print(s, flush=True)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_sell_model · 该卖模型：接下来 30 分钟内会不会跌 >=1.5%')
    log('=' * 100)
    log(f'  标签：未来 {HORIZON} 分钟最低价 < 当前价 × (1 - {DROP:.1%})')
    log(f'  特征：前 {SEQ} 根 × {CH} 通道，每通道 6 统计量 = {CH*6} 维')
    log()

    log('读数据...')
    dtypes = {
        'symbol': 'category', 'open_time': 'int64',
        'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
        'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32',
    }
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())
    rng = np.random.default_rng(11)

    X, y, ts = [], [], []

    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < SEQ + HORIZON + 100:
            continue

        o = g.open.values.astype(np.float32)
        h = g.high.values.astype(np.float32)
        l = g.low.values.astype(np.float32)
        c = g.close.values.astype(np.float32)
        v = g.volume.values.astype(np.float32)
        tb = g.taker_base.values.astype(np.float32)
        tq = g.taker_quote.values.astype(np.float32)
        t = g.open_time.values

        n_pts = min(CAP_PER_SYM, n - HORIZON - SEQ)
        if n_pts <= 0:
            continue
        idxs = rng.choice(np.arange(SEQ, n - HORIZON), size=n_pts, replace=False)

        for i in idxs:
            fut_low = l[i:i + HORIZON].min()
            cur = c[i - 1]
            if not np.isfinite(cur) or cur <= 0:
                continue
            y.append(1 if fut_low < cur * (1 - DROP) else 0)

            base = max(float(cur), 1e-9)
            pv = np.median(v[i - SEQ:i]) + 1e-12
            ptb = np.median(tb[i - SEQ:i]) + 1e-12
            ptq = np.median(tq[i - SEQ:i]) + 1e-12

            seq = np.stack([
                o[i - SEQ:i] / base - 1,
                h[i - SEQ:i] / base - 1,
                l[i - SEQ:i] / base - 1,
                c[i - SEQ:i] / base - 1,
                v[i - SEQ:i] / pv,
                tb[i - SEQ:i] / ptb,
                tq[i - SEQ:i] / ptq,
            ], axis=1)

            m = seq.mean(axis=0)
            s = seq.std(axis=0)
            mx = seq.max(axis=0)
            mn = seq.min(axis=0)
            last = seq[-1]
            slope = seq[-30:].mean(axis=0) - seq[:30].mean(axis=0)
            feat = np.concatenate([m, s, mx, mn, last, slope])
            X.append(feat.astype(np.float32))
            ts.append(t[i])

        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    ts = np.asarray(ts, dtype=np.int64)

    log(f'\n样本 {len(X):,}  正样本率 {y.mean():.1%}')
    log(f'时间 {pd.to_datetime(int(ts.min()), unit="ms", utc=True)} ~ '
        f'{pd.to_datetime(int(ts.max()), unit="ms", utc=True)}')

    order = np.argsort(ts)
    X, y, ts = X[order], y[order], ts[order]
    n = len(X)
    tr_end = int(n * 0.67)
    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xte, yte = X[tr_end:], y[tr_end:]

    log(f'训练 {len(Xtr):,}  测试 {len(Xte):,}')

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    log('\n训练 GBDT...')
    mdl = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=7,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    log(f'  测试 AUC = {auc:.4f}   基线 {yte.mean():.1%}')

    log('\n【分档】')
    log(f'  {"P区间":<14}{"n":>9}{"实际跌率":>10}{"提升":>8}')
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            log(f'  [{lo:.1f},{hi:.2f})    {m.sum():>8,}  {yte[m].mean():>9.1%}  {yte[m].mean()/yte.mean():>7.2f}x')

    log('\n【时间两半（测试段内）】')
    half = len(p) // 2
    for nm, sl in [('前半', slice(0, half)), ('后半', slice(half, len(p)))]:
        pp, yy = p[sl], yte[sl]
        log(f'  {nm}  n={len(pp):,}  基线 {yy.mean():.1%}')
        for th in [0.5, 0.6, 0.7]:
            m = pp >= th
            if m.sum() > 0:
                log(f'    P>={th:.1f}  n={m.sum():>7,}  跌率 {yy[m].mean():>6.1%}  报出占比 {m.mean():>5.1%}')

    log('\n【特征重要性 Top12】')
    try:
        from sklearn.inspection import permutation_importance
        names = ([f'{ch}_mean' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']] +
                 [f'{ch}_std' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']] +
                 [f'{ch}_max' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']] +
                 [f'{ch}_min' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']] +
                 [f'{ch}_last' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']] +
                 [f'{ch}_slope' for ch in ['o', 'h', 'l', 'c', 'v', 'tb', 'tq']])
        # 用子样本算置换重要性（快）
        sub = np.random.default_rng(3).choice(len(Xte), size=min(8000, len(Xte)), replace=False)
        imp = permutation_importance(mdl, Xte[sub], yte[sub], n_repeats=3, random_state=42, scoring='roc_auc')
        for i in np.argsort(-imp.importances_mean)[:12]:
            log(f'  {names[i]:12s}  {imp.importances_mean[i]:+.4f}')
    except Exception as e:
        log(f'  （置换重要性失败：{e}）')

    log(f'\nTOTAL {time.time()-t0:.0f}s')

    np.save(os.path.join(OUT, 'sell_model_y.npy'), yte)
    np.save(os.path.join(OUT, 'sell_model_p.npy'), p)
    np.save(os.path.join(OUT, 'sell_model_ts.npy'), ts[tr_end:])
    log(f'已存 sell_model_y/p/ts.npy → {OUT}')


if __name__ == '__main__':
    main()
