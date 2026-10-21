# -*- coding: utf-8 -*-
"""gambler_evolve · 概率演化组合：用遗传算法自己搜索信号组合

不是手工设计模型，而是：
  候选信号池（20 个布尔条件）→ 个体 = 信号子集(AND) → 适应度 = 做空净期望
  → 锦标赛选择 + 交叉 + 变异，进化 40 代
  前 30 天进化，后 30 天验证（防过拟合）
标签：做空 +2% 止盈 / -1% 止损（60 分钟窗口，双触保守算输）
"""
import os, gc, time, random
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
STEP = 100
FEE = 0.0004

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_evolve · 遗传算法自己搜信号组合（不做手工模型）')
log('=' * 100)

XS, YW, YL, TS = [], [], [], []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        win = (fmin <= cc * 0.98) & (fmax < cc * 1.01)     # 做空赢：先跌2%且未涨1%
        lose = (fmax >= cc * 1.01)                          # 做空输
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60, r240, r1440 = ret(1), ret(5), ret(15), ret(60), ret(240), ret(1440)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmax) & np.isfinite(fmin)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YW.append(win[idx]); YL.append(lose[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.vstack(XS); yw = np.concatenate(YW); yl = np.concatenate(YL); t = np.concatenate(TS)
del XS, YW, YL, TS; gc.collect()
log(f'样本 {len(X):,}  做空赢 {yw.mean():.1%}  做空输 {yl.mean():.1%}')

r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour = [X[:,i] for i in range(11)]

# ---------- 候选信号池（20 个布尔条件） ----------
def q(a, p): return float(np.nanpercentile(a, p))
SIG = [
 ('r15<-2%',   r15 < -0.02),
 ('r15<-5%',   r15 < -0.05),
 ('r15>+2%',   r15 >  0.02),
 ('r60<-5%',   r60 < -0.05),
 ('r60>+5%',   r60 >  0.05),
 ('r240<-10%', r240 < -0.10),
 ('r240>+10%', r240 >  0.10),
 ('r1440<-20%', r1440 < -0.20),
 ('r1440>+20%', r1440 >  0.20),
 ('volr>3',    volr > 3),
 ('volr>6',    volr > 6),
 ('volr<0.5',  volr < 0.5),
 ('vstd>p70',  vstd > q(vstd,70)),
 ('vstd>p90',  vstd > q(vstd,90)),
 ('tbr>0.55',  tbr > 0.55),
 ('tbr<0.45',  tbr < 0.45),
 ('trr>3',     trr > 3),
 ('hour 0-6',  (hour >= 0) & (hour < 6)),
 ('hour 12-18',(hour >= 12) & (hour < 18)),
 ('r1>0',      r1 > 0),
]
NS = len(SIG)
MASKS = np.column_stack([m for _, m in SIG])     # (N, 20)
log(f'信号池 {NS} 个，样本矩阵 {MASKS.shape}')

cut = np.median(t)
tr = t <= cut; te = t > cut
Mtr = MASKS[tr]; wtr = yw[tr]; ltr = yl[tr]
Mte = MASKS[te]; wte = yw[te]; lte = yl[te]
log(f'训练 {tr.sum():,}  测试 {te.sum():,}')

def fitness(genes):
    if len(genes) == 0:
        return -1.0, 0
    sel = np.ones(len(Mtr), bool)
    for g in genes:
        sel &= Mtr[:, g]
    n = sel.sum()
    if n < 50:
        return -1.0, n
    w = (wtr & sel).sum()
    l = (ltr & sel).sum()
    if w + l == 0:
        return -1.0, n
    exp = (w * 0.02 - l * 0.01) / n - FEE
    return float(exp), int(n)

def evaluate(genes):
    exp, n = fitness(genes)
    # 小样本惩罚
    return exp * min(1.0, n / 200.0)

POP = 160
GENS = 40
random.seed(7); np.random.seed(7)

def rand_ind():
    k = random.randint(1, 5)
    return sorted(random.sample(range(NS), k))

pop = [rand_ind() for _ in range(POP)]
best_hist = []
t_evo = time.time()
for gen in range(GENS):
    scored = [(evaluate(ind), ind) for ind in pop]
    scored.sort(key=lambda x: -x[0])
    if not best_hist or scored[0][0] > best_hist[-1][0]:
        best_hist.append((scored[0][0], list(scored[0][1])))
    else:
        best_hist.append(best_hist[-1])
    # 精英 + 锦标赛
    elites = [list(ind) for _, ind in scored[:8]]
    newpop = elites[:]
    while len(newpop) < POP:
        a = max(random.sample(scored[:60], 3), key=lambda x: x[0])[1]
        b = max(random.sample(scored[:60], 3), key=lambda x: x[0])[1]
        # 交叉：取并集的一半
        child = sorted(set(a) | set(b))
        random.shuffle(child)
        child = sorted(child[:max(1, (len(a)+len(b))//2)])
        if random.random() < 0.3:
            if random.random() < 0.5 and len(child) > 1:
                child.remove(random.choice(child))
            elif len(child) < 8:
                child.append(random.randrange(NS)); child = sorted(set(child))
        newpop.append(child)
    pop = newpop
    if gen % 10 == 0 or gen == GENS-1:
        log(f'  gen {gen+1:>2}/{GENS}  best_train_exp={scored[0][0]:+.4f}  genes={scored[0][1]}  累计{time.time()-t_evo:.0f}s')

log(f'进化耗时 {time.time()-t_evo:.0f}s')

# ---------- 测试段验证 top5 ----------
scored = sorted([(evaluate(ind), ind) for ind in pop], key=lambda x: -x[0])[:5]
log('\n【测试段验证 Top5】')
log(f'{"组合":60s}{"训练期望":>10s}{"测试样本":>9s}{"测试赢率":>9s}{"测试赢":>8s}{"测试输":>8s}{"测试净期望":>11s}')
log('-' * 120)
for fig, genes in scored:
    names = '+'.join(SIG[g][0] for g in genes)
    sel = np.ones(len(Mte), bool)
    for g in genes:
        sel &= Mte[:, g]
    n = sel.sum()
    if n == 0:
        log(f'{names:60s}{fig:>10.4f}{0:>9}')
        continue
    w = (wte & sel).sum(); l = (lte & sel).sum()
    wr = w / n
    exp = (w * 0.02 - l * 0.01) / n - FEE
    log(f'{names[:60]:60s}{fig:>10.4f}{n:>9,}{wr:>9.1%}{w:>8,}{l:>8,}{exp:>11.3%}')

# 基准
log('\n【基准】')
n0 = len(Mte); w0 = wte.sum(); l0 = lte.sum()
log(f'  全样本做空：n={n0:,} 赢率={w0/n0:.1%} 净期望={(w0*0.02-l0*0.01)/n0-FEE:.3%}')
log(f'  门槛：赢率需 > 34.7%（maker）')

log('\n【判读】')
log('  · 若测试期望仍负 ⇒ 进化只在训练集找到噪声，不是真规律')
log('  · 若某组合测试>0且样本>100 ⇒ 这是进化出来的真信号，值得进一步验证')

out = os.path.join(OUT, 'evolve_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
