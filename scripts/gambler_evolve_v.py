"""
gambler_evolve_v.py · 小的进化可行性验证

只回答一个问题：
  在这份数据上，进化能不能找到「样本外也赚钱」的策略？
  还是只会记住训练段？

规模刻意小：6 币 × 3 月，种群 20，30 代。
防前视：
  - 特征全是过去（rolling 天然如此）
  - 标准化统计量只用训练段
  - 进化只看训练段，测试段只在最后看一次
"""
import sys, io, os, time, random
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')
random.seed(11); np.random.seed(11)

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\cycle'

N_SYM   = 6
MONTHS  = 3
POP     = 20
GENS    = 30
FEE     = 0.005
POS     = 0.10
MIN_TR  = 30

FEATS = ['r5', 'r15', 'r60', 'r240', 'volr', 'vstd', 'tbr']
NF = len(FEATS)

log = lambda s='': print(s, flush=True)


def build(sym, months=MONTHS):
    g = sym.sort_values('open_time')
    o = g.open.values.astype(np.float64); h = g.high.values.astype(np.float64)
    l = g.low.values.astype(np.float64); c = g.close.values.astype(np.float64)
    v = g.volume.values.astype(np.float64)
    tb = g.taker_base.values.astype(np.float64); tq = g.taker_quote.values.astype(np.float64)
    n = len(c)
    if n < 3000:
        return None

    def roll_mean(x, k):
        cs = np.cumsum(np.insert(x, 0, 0))
        out = np.full(len(x), np.nan)
        out[k-1:] = (cs[k:] - cs[:-k]) / k
        return out

    def roll_std(x, k):
        m = roll_mean(x, k)
        m2 = roll_mean(x * x, k)
        return np.sqrt(np.maximum(m2 - m * m, 0))

    r1 = np.zeros(n); r1[1:] = c[1:] / np.maximum(c[:-1], 1e-12) - 1
    X = np.full((n, NF), np.nan)
    X[:, 0] = np.concatenate([np.full(5, np.nan), c[5:] / np.maximum(c[:-5], 1e-12) - 1])
    X[:, 1] = np.concatenate([np.full(15, np.nan), c[15:] / np.maximum(c[:-15], 1e-12) - 1])
    X[:, 2] = np.concatenate([np.full(60, np.nan), c[60:] / np.maximum(c[:-60], 1e-12) - 1])
    X[:, 3] = np.concatenate([np.full(240, np.nan), c[240:] / np.maximum(c[:-240], 1e-12) - 1])
    X[:, 4] = v / np.maximum(roll_mean(v, 60), 1e-12)
    X[:, 5] = roll_std(r1, 60)
    X[:, 6] = tb / np.maximum(tb + tq, 1e-9) - 0.5
    return X, o, h, l, c


def backtest(data, gene, mu, sd):
    w = gene[:NF]; th = gene[NF]
    tp = gene[NF+1]; sl = gene[NF+2]; hold = int(gene[NF+3])
    rets = []
    for (X, o, h, l, c) in data:
        if X is None:
            continue
        Z = (X - mu) / sd
        score = np.nansum(Z * w, axis=1)
        sig = np.isfinite(score) & (score > th)
        n = len(c)
        i = 0
        while i < n - 1:
            if not sig[i]:
                i += 1; continue
            entry = c[i]
            if not np.isfinite(entry) or entry <= 0:
                i += 1; continue
            end = min(i + 1 + hold, n)
            r = None
            j = i + 1
            while j < end:
                if h[j] >= entry * (1 + tp):
                    r = tp - FEE; break
                if l[j] <= entry * (1 - sl):
                    r = -sl - FEE; break
                j += 1
            if r is None:
                r = c[end-1] / entry - 1 - FEE
                j = end - 1
            rets.append(r)
            i = j + 1
    if len(rets) < MIN_TR:
        return None
    r = np.array(rets)
    eq = np.cumprod(1 + r * POS)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return dict(n=len(r), mean=r.mean(), final=eq[-1], dd=dd, eq=eq)


def fitness(res):
    if res is None:
        return -1e6
    return (res['final'] - 1) - 2.0 * (-res['dd'])


def rand_gene():
    return np.array([
        *np.random.uniform(-2, 2, NF),      # 特征权重
        np.random.uniform(-1.5, 1.5),        # 阈值
        np.random.uniform(0.01, 0.10),       # TP
        np.random.uniform(0.005, 0.05),      # SL
        np.random.uniform(60, 1440),         # 持有上限
    ])


def mutate(g, rate=0.25, scale=0.35):
    g2 = g.copy()
    for k in range(len(g2)):
        if np.random.rand() < rate:
            g2[k] += np.random.randn() * scale * (abs(g[k]) + 0.3)
    g2[NF] = np.clip(g2[NF], -3, 3)
    g2[NF+1] = np.clip(g2[NF+1], 0.005, 0.15)
    g2[NF+2] = np.clip(g2[NF+2], 0.003, 0.08)
    g2[NF+3] = np.clip(g2[NF+3], 30, 1440)
    return g2


def main():
    t0 = time.time()
    log('=' * 90)
    log('gambler_evolve_v · 小的进化可行性验证')
    log(f'  {N_SYM} 币 × {MONTHS} 月  种群 {POP}  {GENS} 代  TP/SL/持有/权重全由进化决定')
    log('=' * 90)

    log('\n读数据...')
    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32',
          'low': 'float32', 'close': 'float32', 'volume': 'float32',
          'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:N_SYM]
    log(f'  {len(df):,} 行  用 {syms}')

    train, test = [], []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym]
        b = build(g)
        if b is None:
            continue
        X, o, h, l, c = b
        cut = int(len(c) * 0.7)
        tr = (X[:cut], o[:cut], h[:cut], l[:cut], c[:cut])
        te = (X[cut:], o[cut:], h[cut:], l[cut:], c[cut:])
        train.append(tr); test.append(te)
        log(f'  {sym}: {len(c):,} 根  训练 {cut:,}  测试 {len(c)-cut:,}')

    # 标准化：只用训练段
    allX = np.vstack([t[0] for t in train])
    mu = np.nanmean(allX, axis=0)
    sd = np.nanstd(allX, axis=0) + 1e-9
    log(f'  标准化统计量（仅训练段）: mu={mu.round(3)}')

    log(f'\n[1/2] 进化（{GENS} 代）...')
    pop = [rand_gene() for _ in range(POP)]
    best_hist = []
    for gen in range(GENS):
        fits = []
        for g in pop:
            r = backtest(train, g, mu, sd)
            fits.append(fitness(r))
        order = np.argsort(fits)[::-1]
        pop = [pop[i] for i in order]
        fits = [fits[i] for i in order]
        best_hist.append(fits[0])
        if (gen + 1) % 5 == 0 or gen == 0:
            log(f'  gen {gen+1:>3}  best {fits[0]:>10.1f}  avg {np.mean(fits):>10.1f}  '
                f'第2 {fits[1]:>9.1f}  {time.time()-t0:.0f}s')
        # 保留前 25%，其余由前 50% 变异
        n_keep = max(2, POP // 4)
        newpop = pop[:n_keep]
        while len(newpop) < POP:
            par = pop[np.random.randint(0, max(2, POP // 2))]
            newpop.append(mutate(par))
        pop = newpop

    best = pop[0]
    log(f'\n[2/2] 最好策略')
    log(f'  权重 {dict(zip(FEATS, best[:NF].round(2)))}')
    log(f'  阈值 {best[NF]:.3f}  TP {best[NF+1]:.3f}  SL {best[NF+2]:.3f}  持有 {int(best[NF+3])} 根')

    tr_r = backtest(train, best, mu, sd)
    te_r = backtest(test, best, mu, sd)

    log('\n' + '=' * 90)
    log('【结果】')
    log('=' * 90)
    for name, r in [('训练段', tr_r), ('测试段（样本外）', te_r)]:
        if r is None:
            log(f'  {name}: 交易不足 {MIN_TR} 笔 → 判定为退化解')
        else:
            log(f'  {name}: {r["n"]:>5} 笔  均值 {r["mean"]:>+7.2%}  最终 {r["final"]:>8.2f}x  回撤 {r["dd"]:>+7.1%}')

    log('\n【进化曲线（每代最好）】')
    for k in range(0, len(best_hist), 5):
        log(f'  gen {k+1:>3}  {best_hist[k]:>10.1f}')

    ok = te_r is not None and te_r['mean'] > 0
    log('\n【判定】')
    log(f'  进化可行性：{"✓ 样本外仍为正，值得继续" if ok else "✗ 样本外不为正，进化只在记历史"}')
    log('  注意：只跑了 1 次进化、6 币 3 月，样本小，结论仅作方向参考')

    rep = os.path.join(OUT, 'evolve_v_summary.txt')
    with open(rep, 'w', encoding='utf-8') as f:
        f.write('\n'.join([
            f'{N_SYM} 币 × {MONTHS} 月  种群 {POP}  {GENS} 代',
            f'基因 {dict(zip(FEATS, best[:NF].round(3)))}',
            f'阈值 {best[NF]:.4f} TP {best[NF+1]:.4f} SL {best[NF+2]:.4f} 持有 {int(best[NF+3])}',
            f'训练段 {tr_r["n"] if tr_r else 0} 笔 均值 {tr_r["mean"] if tr_r else 0:.4f}',
            f'测试段 {te_r["n"] if te_r else 0} 笔 均值 {te_r["mean"] if te_r else 0:.4f}',
            f'判定 {"可行" if ok else "不可行/存疑"}',
        ]))
    log(f'\n报告 {rep}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
