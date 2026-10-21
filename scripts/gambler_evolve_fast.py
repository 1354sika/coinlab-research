"""
gambler_evolve_fast.py · 11 维进化（加速版）

相对 v1 的三处改动：
  1. 多进程 4 核（每次评估独立，天然可并行）
  2. 数据采样 1/5（63 万根，速度 ×5）
  3. 每代落盘 CSV（防意外关机丢失）

保留 11 维基因：7 特征权重 + 阈值 + TP + SL + 持有上限
"""
import sys, io, os, time, json
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
CACHE = r'D:\中信证券\coinlab\evidence\cycle\_evolve_cache.npz'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

N_SYM   = 6
NPZ_VER = 1
POP     = 40
GENS    = 100
ELITE   = 8
FEE     = 0.005
POS     = 0.10
MIN_TR  = 30
NPROC   = 4

FEATS = ['r5', 'r15', 'r60', 'r240', 'volr', 'vstd', 'tbr']
NF = len(FEATS)

_G = {}

log = lambda s='': print(s, flush=True)


def build_arrays():
    """把 6 币的特征矩阵/价格算好存 npz（只做一次）"""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if int(np.asarray(d['ver']).ravel()[0]) == NPZ_VER:
            return d

    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32',
          'low': 'float32', 'close': 'float32', 'volume': 'float32',
          'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:N_SYM]

    pack = {}
    for sym in syms:
        g = df[df.symbol == sym].sort_values('open_time')
        c = g.close.values.astype(np.float64)
        h = g.high.values.astype(np.float64)
        l = g.low.values.astype(np.float64)
        v = g.volume.values.astype(np.float64)
        tb = g.taker_base.values.astype(np.float64)
        tq = g.taker_quote.values.astype(np.float64)
        n = len(c)
        if n < 3000:
            continue

        # 采样 1/5（保留顺序，不影响时间切分）
        idx = np.arange(0, n, 5)

        def rm(x, k):
            cs = np.cumsum(np.insert(x, 0, 0))
            o = np.full(len(x), np.nan)
            o[k-1:] = (cs[k:] - cs[:-k]) / k
            return o

        r1 = np.zeros(n); r1[1:] = c[1:] / np.maximum(c[:-1], 1e-12) - 1
        X = np.full((n, NF), np.nan)
        X[:, 0] = np.concatenate([np.full(5, np.nan), c[5:] / np.maximum(c[:-5], 1e-12) - 1])
        X[:, 1] = np.concatenate([np.full(15, np.nan), c[15:] / np.maximum(c[:-15], 1e-12) - 1])
        X[:, 2] = np.concatenate([np.full(60, np.nan), c[60:] / np.maximum(c[:-60], 1e-12) - 1])
        X[:, 3] = np.concatenate([np.full(240, np.nan), c[240:] / np.maximum(c[:-240], 1e-12) - 1])
        X[:, 4] = v / np.maximum(rm(v, 60), 1e-12)
        X[:, 5] = np.sqrt(np.maximum(rm(r1*r1, 60) - rm(r1, 60)**2, 0))
        X[:, 6] = tb / np.maximum(tb + tq, 1e-9) - 0.5

        pack[f'{sym}_X'] = X[idx].astype(np.float32)
        pack[f'{sym}_h'] = h[idx].astype(np.float64)
        pack[f'{sym}_l'] = l[idx].astype(np.float64)
        pack[f'{sym}_c'] = c[idx].astype(np.float64)

    pack['ver'] = np.array([NPZ_VER])
    pack['syms'] = np.array(sorted({k.rsplit('_', 1)[0] for k in pack if k.endswith('_X')}))
    np.savez_compressed(CACHE, **pack)
    return np.load(CACHE, allow_pickle=True)


def init_worker(cache_path):
    d = np.load(cache_path, allow_pickle=True)
    syms = list(d['syms'])
    data = []
    for s in syms:
        X = d[f'{s}_X']; h = d[f'{s}_h']; l = d[f'{s}_l']; c = d[f'{s}_c']
        cut = int(len(c) * 0.7)
        data.append(dict(
            Xtr=X[:cut], htr=h[:cut], ltr=l[:cut], ctr=c[:cut],
            Xte=X[cut:], hte=h[cut:], lte=l[cut:], cte=c[cut:],
        ))
    mu = np.nanmean(np.vstack([x['Xtr'] for x in data]), axis=0)
    sd = np.nanstd(np.vstack([x['Xtr'] for x in data]), axis=0) + 1e-9
    _G['data'] = data
    _G['mu'] = mu
    _G['sd'] = sd


def run_one(gene, split='tr'):
    w = gene[:NF]; th = gene[NF]
    tp = gene[NF+1]; sl = gene[NF+2]; hold = int(gene[NF+3])
    mu, sd = _G['mu'], _G['sd']
    rets = []
    for d in _G['data']:
        Xk = 'Xtr' if split == 'tr' else 'Xte'
        hk = 'htr' if split == 'tr' else 'hte'
        lk = 'ltr' if split == 'tr' else 'lte'
        ck = 'ctr' if split == 'tr' else 'cte'
        X, h, l, c = d[Xk], d[hk], d[lk], d[ck]
        Z = (X - mu) / sd
        score = np.nansum(Z * w, axis=1)
        sig = np.isfinite(score) & (score > th)
        n = len(c)
        i = 0
        while i < n - 1:
            if not sig[i]:
                i += 1
                continue
            entry = c[i]
            if not np.isfinite(entry) or entry <= 0:
                i += 1
                continue
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
    return (float(eq[-1]), float(dd), int(len(r)), float(r.mean()))


def fit_of(res):
    """适应度 = 单笔真实均值（用户口径：结果导向）
    回撤 > 60% 直接淘汰（硬约束，不做减法）
    """
    if res is None:
        return -1e6
    final, dd, n, mean = res
    if -dd > 0.60:
        return -1e6
    return mean * 100.0          # ×100 让数值在百分数量级


def evaluate(gene):
    return fit_of(run_one(gene, 'tr'))


def rand_gene():
    return np.array([
        *np.random.uniform(-2, 2, NF),
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(0.01, 0.10),
        np.random.uniform(0.005, 0.05),
        np.random.uniform(60, 1440),
    ])


def mutate(g, rate=0.3, scale=0.4):
    g2 = g.copy()
    for k in range(len(g2)):
        if np.random.rand() < rate:
            g2[k] += np.random.randn() * scale * (abs(g[k]) + 0.3)
    g2[:NF] = np.clip(g2[:NF], -2, 2)      # 权重限幅，防 -18 极端解
    g2[NF] = np.clip(g2[NF], -3, 3)
    g2[NF+1] = np.clip(g2[NF+1], 0.005, 0.15)
    g2[NF+2] = np.clip(g2[NF+2], 0.003, 0.08)
    g2[NF+3] = np.clip(g2[NF+3], 30, 1440)
    return g2


def main():
    t0 = time.time()
    log('=' * 90)
    log('gambler_evolve_fast · 11 维进化加速版')
    log(f'  {N_SYM} 币  采样1/5  种群{POP}  {GENS}代  {NPROC}进程')
    log('=' * 90)

    np.random.seed(11)
    log('\n准备数据缓存...')
    build_arrays()
    init_worker(CACHE)
    log(f'  {len(_G["data"])} 币  每币训练/测试已切  {time.time()-t0:.0f}s')

    csv_path = os.path.join(OUT, 'evolve_fast_gen.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('gen,best_mean_pct,avg_fit,second_mean_pct,n_pos\n')

    pop = [rand_gene() for _ in range(POP)]
    hist = []
    with Pool(NPROC, initializer=init_worker, initargs=(CACHE,)) as pool:
        for gen in range(GENS):
            fits = pool.map(evaluate, pop)
            order = np.argsort(fits)[::-1]
            pop = [pop[i] for i in order]
            fits = [fits[i] for i in order]
            best, second = fits[0], fits[1] if len(fits) > 1 else fits[0]
            avg = float(np.mean(fits))
            npos = int(sum(1 for x in fits if x > 0))
            hist.append((gen+1, best, avg, second, npos))

            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f'{gen+1},{best/100:.6f},{avg:.4f},{second/100:.6f},{npos}\n')

            if (gen + 1) % 5 == 0 or gen == 0:
                bm = best / 100 if best > -1e5 else float('nan')
                sm = second / 100 if second > -1e5 else float('nan')
                log(f'  gen {gen+1:>3}  best均值 {bm:>+7.3%}  第2均值 {sm:>+7.3%}  '
                    f'正收益 {npos:>2}/{POP}  {time.time()-t0:.0f}s')

            newpop = pop[:ELITE]
            while len(newpop) < POP:
                par = pop[np.random.randint(0, max(2, POP // 2))]
                newpop.append(mutate(par))
            pop = newpop

        best_gene = pop[0]
        tr = run_one(best_gene, 'tr')
        te = run_one(best_gene, 'te')

    log('\n' + '=' * 90)
    log('【最好策略】')
    log(f'  权重 {dict(zip(FEATS, best_gene[:NF].round(2)))}')
    log(f'  阈值 {best_gene[NF]:.3f}  TP {best_gene[NF+1]:.3f}  SL {best_gene[NF+2]:.3f}  持有 {int(best_gene[NF+3])}')
    for name, r in [('训练段', tr), ('测试段', te)]:
        if r is None:
            log(f'  {name}: 交易不足 {MIN_TR} 笔')
        else:
            log(f'  {name}: {r[2]:>6} 笔  均值 {r[3]:>+7.3%}  最终 {r[0]:>8.3f}x  回撤 {r[1]:>+7.1%}')

    log('\n【进化曲线（best 单笔均值）】')
    for h in hist[::10]:
        bm = h[1] / 100 if h[1] > -1e5 else float('nan')
        log(f'  gen {h[0]:>3}  best均值 {bm:>+7.3%}  正收益 {h[4]}/{POP}')

    if te is None:
        log(f'\n【判定】样本外交易不足 {MIN_TR} 笔，无法判定（不是"不为正"，是没测到）')
    else:
        ok = te[3] > 0
        log(f'\n【判定】样本外均值 {te[3]:+.3%} → {"为正，值得继续" if ok else "不为正，进化未找到可交易结构"}')
    log(f'  明细 CSV: {csv_path}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
