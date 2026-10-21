"""
gambler_joint.py · 该卖模型 + 模拟交易（无前视）

标签：未来 30 分钟最低价 < close[i-1] * (1 - 1.5%)
入场：240min 低点 × 1.03，当根收盘价
出场（任一）：
  a. 触及 TP +5%
  b. 触及 SL -2%
  c. 该卖模型 P >= TH（每 STEP 根问一次）
  d. 1440 根超时
对比：原规则 vs 该卖 P>=0.5/0.6/0.7
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SEQ = 60
CH = 7
SELL_HORIZON = 30
SELL_DROP = 0.015
CAP_PER_SYM = 1600
A1_WIN = 240
A1_CONFIRM = 1.03
TP = 0.05
SL = 0.02
HOLD = 1440
FEE = 0.005
STEP = 20
SIMS_N = 3000

np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

log = lambda s='': print(s, flush=True)


def build_feat(o, h, l, c, v, tb, tq, i, seq=SEQ):
    """前 seq 根（i-seq .. i-1）的 42 维特征。float64 计算 + log1p 量能，防溢出。"""
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
    if not np.all(np.isfinite(out)):
        return None
    return out


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_joint · 该卖模型 + 模拟交易（无前视）')
    log('=' * 100)
    log(f'  入场：{A1_WIN}min 低点 × {A1_CONFIRM}  当根收盘价')
    log(f'  出场：TP +{TP:.0%} / SL -{SL:.0%} / 该卖 P>={{TH}} / {HOLD} 根超时')
    log(f'  成本 {FEE:.2%}/笔   该卖模型每 {STEP} 根问一次')
    log()

    log('读数据...')
    dtypes = {'symbol': 'category', 'open_time': 'int64',
              'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
              'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())
    rng = np.random.default_rng(23)

    t_all = df.open_time.values
    t_tr_end = int(np.quantile(t_all, 0.67))
    log(f'  训练截止 {pd.to_datetime(t_tr_end, unit="ms", utc=True)}')

    # ===== 1. 训该卖模型（只用训练段）=====
    log('\n[1/3] 训该卖模型...')
    Xtr, ytr = [], []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < SEQ + SELL_HORIZON + 100:
            continue
        o = g.open.values.astype(np.float32); h = g.high.values.astype(np.float32)
        l = g.low.values.astype(np.float32); c = g.close.values.astype(np.float32)
        v = g.volume.values.astype(np.float32)
        tb = g.taker_base.values.astype(np.float32); tq = g.taker_quote.values.astype(np.float32)
        t = g.open_time.values

        m_tr = (t >= t[SEQ]) & (t < t_tr_end - SELL_HORIZON * 60000)
        idx_tr = np.where(m_tr)[0]
        if len(idx_tr) == 0:
            continue
        n_pts = min(CAP_PER_SYM // 2, len(idx_tr))
        sel = rng.choice(idx_tr, size=n_pts, replace=False)
        for i in sel:
            f = build_feat(o, h, l, c, v, tb, tq, i)
            if f is None:
                continue
            fut_low = l[i:i+SELL_HORIZON].min()
            cur = c[i-1]
            if not np.isfinite(cur) or cur <= 0:
                continue
            ytr.append(1 if fut_low < cur * (1 - SELL_DROP) else 0)
            Xtr.append(f)
        if (si+1) % 15 == 0:
            log(f'    {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int8)
    log(f'  训练样本 {len(Xtr):,}  正样本率 {ytr.mean():.1%}')

    from sklearn.ensemble import HistGradientBoostingClassifier
    mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.10, max_depth=6,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=42)
    mdl.fit(Xtr, ytr)
    log(f'  模型训练完成  ({time.time()-t0:.0f}s)')

    # ===== 2. 找 A1 触发点（仅测试段）=====
    log('\n[2/3] 找 A1 触发点（仅测试段，无前视）...')
    TRIGS = []
    for sym in syms:
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < A1_WIN + SEQ + 200:
            continue
        o = g.open.values.astype(np.float32); h = g.high.values.astype(np.float32)
        l = g.low.values.astype(np.float32); c = g.close.values.astype(np.float32)
        v = g.volume.values.astype(np.float32)
        tb = g.taker_base.values.astype(np.float32); tq = g.taker_quote.values.astype(np.float32)
        t = g.open_time.values

        idx_te = np.where(t >= t_tr_end)[0]
        if len(idx_te) < 100:
            continue
        for i in idx_te:
            if i < A1_WIN + SEQ + 1 or i >= n - 2:
                continue
            lo = l[i-A1_WIN:i].min()
            if c[i] >= lo * A1_CONFIRM:
                TRIGS.append((sym, i, o, h, l, c, v, tb, tq, t))
    log(f'  测试段触发点 {len(TRIGS):,}')

    if len(TRIGS) > SIMS_N:
        sel = rng.choice(len(TRIGS), size=SIMS_N, replace=False)
        TRIGS = [TRIGS[k] for k in sel]
        log(f'  模拟子样本 {len(TRIGS):,}（随机）')

    # ===== 3. 模拟 =====
    log('\n[3/3] 模拟交易...')
    TH_LIST = [None, 0.5, 0.6, 0.7]
    res = {th: {'n': 0, 'tp': 0, 'sl': 0, 'sell': 0, 'to': 0, 'ret': []}
           for th in TH_LIST}

    for tri, (sym, i, o, h, l, c, v, tb, tq, t) in enumerate(TRIGS):
        n = len(c)
        entry = float(c[i])
        if not np.isfinite(entry) or entry <= 0:
            continue

        for th in TH_LIST:
            end = min(i + 1 + HOLD, n)
            ret = None; why = None
            for j in range(i + 1, end):
                if h[j] >= entry * (1 + TP):
                    ret = TP - FEE; why = 'tp'; break
                if l[j] <= entry * (1 - SL):
                    ret = -SL - FEE; why = 'sl'; break
                if th is not None and ((j - i) % STEP == 0):
                    f = build_feat(o, h, l, c, v, tb, tq, j)
                    if f is not None:
                        p = mdl.predict_proba(f.reshape(1, -1))[0, 1]
                        if p >= th:
                            ret = c[j] / entry - 1 - FEE
                            why = 'sell'; break
            if ret is None:
                k = end - 1
                ret = c[k] / entry - 1 - FEE
                why = 'to'

            r = res[th]
            r['n'] += 1
            r[why] += 1
            r['ret'].append(ret)

        if (tri+1) % 2000 == 0:
            log(f'    {tri+1}/{len(TRIGS)}  {time.time()-t0:.0f}s')

    # ===== 结果 =====
    log('\n' + '=' * 100)
    log('【结果】')
    log('=' * 100)
    log(f'  {"方案":<14}{"笔数":>8}{"止盈%":>8}{"止损%":>8}{"主动出%":>9}{"超时%":>8}'
        f'{"均值%":>9}{"中位%":>9}{"胜率%":>8}{"判定":>7}')
    for th in TH_LIST:
        r = res[th]
        n = r['n']
        if n == 0: continue
        rets = np.array(r['ret']) * 100
        mean = rets.mean(); med = np.median(rets); win = (rets > 0).mean() * 100
        nm = '原规则' if th is None else f'该卖 P>={th}'
        mark = '✓' if mean > 0 else '✗'
        log(f'  {nm:<14}{n:>8,}{r["tp"]/n*100:>7.1f}{r["sl"]/n*100:>8.1f}'
            f'{r["sell"]/n*100:>9.1f}{r["to"]/n*100:>8.1f}{mean:>9.2f}{med:>9.2f}{win:>8.1f}{mark:>7}')

    log('\n【判读】')
    log('  · 主动出% 越高 ⇒ 该卖模型开火越多')
    log('  · 止损% 若从 67% 降到 <25% ⇒ 「该卖」达成目标')
    log('  · 均值若转正 ⇒ 这是第一个可交易配置')

    log(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
