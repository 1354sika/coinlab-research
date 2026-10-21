"""
gambler_hold_ablation.py · 「该持」模型消融实验

目的：判断 AUC 0.9176 里有多少来自「浮盈」（距离计），多少来自 K 线特征（真预测）。

三组对比：
  A 只用 [浮盈, 已持有分钟]        （2 维）
  B 只用 K 线 42 维                （42 维）
  C 全部 44 维                      （当前 0.9176）

若 A ≈ C ⇒ 该持模型本质是「距离计」，K 线特征无增量
若 B > 0.60 且 A < C ⇒ K 线有独立信息
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
    log('gambler_hold_ablation · 该持模型消融：浮盈 vs K 线特征')
    log('=' * 100)

    XA = os.path.join(OUT, 'hold_ablation_X.npy')
    yA = os.path.join(OUT, 'hold_ablation_y.npy')
    spill = os.path.join(OUT, 'hold_ablation_split.npy')
    gA = os.path.join(OUT, 'hold_ablation_grp.npy')

    if os.path.exists(XA) and os.path.exists(yA) and os.path.exists(spill) and os.path.exists(gA):
        log('读取缓存样本...')
        X = np.load(XA); y = np.load(yA); grp = np.load(gA); tr_end = int(np.load(spill)[0])
        log(f'  X {X.shape}  y {y.shape}  grp {grp.shape}  tr_end(入场点索引) {tr_end}')
    else:
        log('读数据...')
        dtypes = {'symbol': 'category', 'open_time': 'int64',
                  'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
                  'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
        df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
        log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

        syms = sorted(df.symbol.unique())
        rng = np.random.default_rng(31)

        log('\n找 A1 触发点...')
        TRIGS = []
        for si, sym in enumerate(syms):
            g = df[df.symbol == sym].sort_values('open_time')
            n = len(g)
            if n < A1_WIN + SEQ + 200:
                continue
            l = g.low.values.astype(np.float32)
            c = g.close.values.astype(np.float32)
            for i in range(A1_WIN + SEQ + 1, n - 1):
                if c[i] >= l[i-A1_WIN:i].min() * A1_CONFIRM:
                    TRIGS.append((sym, i))
            if (si+1) % 15 == 0:
                log(f'  {si+1}/{len(syms)}  累计 {len(TRIGS):,}  {time.time()-t0:.0f}s')
        log(f'  共 {len(TRIGS):,}')

        if len(TRIGS) > N_ENTRY_TOTAL:
            sel = rng.choice(len(TRIGS), size=N_ENTRY_TOTAL, replace=False)
            TRIGS = [TRIGS[k] for k in sel]

        t_by_sym = {s: df[df.symbol == s].sort_values('open_time').open_time.values for s in syms}
        TRIGS.sort(key=lambda x: int(t_by_sym[x[0]][x[1]]))
        tr_end = int(len(TRIGS) * 0.67)

        log('\n构样本...')
        Xl, yl, gl = [], [], []
        CACHE = {}
        for k, (sym, i) in enumerate(TRIGS):
            if sym not in CACHE:
                g = df[df.symbol == sym].sort_values('open_time')
                CACHE[sym] = (g.open.values.astype(np.float32), g.high.values.astype(np.float32),
                              g.low.values.astype(np.float32), g.close.values.astype(np.float32),
                              g.volume.values.astype(np.float32),
                              g.taker_base.values.astype(np.float32), g.taker_quote.values.astype(np.float32))
            o, h, l, c, v, tb, tq = CACHE[sym]
            n = len(c)
            entry = float(c[i])
            if not np.isfinite(entry) or entry <= 0:
                continue
            tp_px = entry * (1 + TP); sl_px = entry * (1 - SL)
            end_j = min(i + HOLD, n - 1)
            if end_j - i < 60:
                continue
            js = np.unique(np.linspace(i + 10, end_j - 10, STEPS_PER_ENTRY).astype(int))
            for j in js:
                f = build_feat(o, h, l, c, v, tb, tq, j)
                if f is None:
                    continue
                feat = np.concatenate([f, [float(c[j]) / entry - 1.0, float(j - i)]]).astype(np.float32)
                lab = 0
                for m in range(j + 1, min(j + HOLD, n)):
                    if h[m] >= tp_px:
                        lab = 1; break
                    if l[m] <= sl_px:
                        lab = 0; break
                Xl.append(feat); yl.append(lab); gl.append(k)
            if (k+1) % 500 == 0:
                log(f'  {k+1}/{len(TRIGS)}  {time.time()-t0:.0f}s')

        X = np.asarray(Xl, dtype=np.float32)
        y = np.asarray(yl, dtype=np.int8)
        grp = np.asarray(gl, dtype=np.int32)
        np.save(XA, X); np.save(yA, y); np.save(gA, grp)
        np.save(spill, np.array([tr_end], dtype=np.int64))
        log(f'  已缓存 {X.shape}  grp {grp.shape}')

    m_tr = grp < tr_end
    Xtr, ytr = X[m_tr], y[m_tr]
    Xte, yte = X[~m_tr], y[~m_tr]
    log(f'\n训练 {len(Xtr):,}  测试 {len(Xte):,}  基线 {yte.mean():.1%}')
    log(f'  训练/测试入场点数 {m_tr.sum()//12} / {(~m_tr).sum()//12}')

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def run(name, cols):
        t = time.time()
        a = Xtr[:, cols]; b = Xte[:, cols]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                           early_stopping=True, validation_fraction=0.1,
                                           random_state=42)
        m.fit(a, ytr)
        p = m.predict_proba(b)[:, 1]
        auc = roc_auc_score(yte, p)
        log(f'  {name:<28}AUC = {auc:.4f}   ({time.time()-t:.0f}s, {len(cols)} 维)')
        return auc, p, m

    log('\n' + '=' * 100)
    log('【消融】')
    log('=' * 100)
    cols_all = list(range(X.shape[1]))
    cols_pnl = [42, 43]
    cols_kline = list(range(42))
    aucA, pA, _ = run('A 只用 浮盈+持有分钟(2维)', cols_pnl)
    aucB, pB, _ = run('B 只用 K线 42维', cols_kline)
    aucC, pC, _ = run('C 全部 44维', cols_all)

    log('\n【判定】')
    gap = aucC - aucA
    log(f'  A(浮盈) {aucA:.4f}  B(K线) {aucB:.4f}  C(全部) {aucC:.4f}')
    log(f'  C − A = {gap:+.4f}')
    if gap < 0.02:
        log('  ⇒ 该持模型几乎等于「距离计」，K 线特征无增量（但距离计本身可交易）')
    elif gap < 0.06:
        log('  ⇒ K 线有少量增量，主体仍是距离计')
    else:
        log('  ⇒ K 线特征带实质增量，该持模型不是纯距离计')

    # 保存三种
    np.save(os.path.join(OUT, 'hold_abl_A.npy'), pA)
    np.save(os.path.join(OUT, 'hold_abl_B.npy'), pB)
    np.save(os.path.join(OUT, 'hold_abl_C.npy'), pC)
    np.save(os.path.join(OUT, 'hold_abl_y.npy'), yte)
    log(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
