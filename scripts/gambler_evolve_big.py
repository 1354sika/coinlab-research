# -*- coding: utf-8 -*-
"""gambler_evolve_big · 进化搜索第二轮（信号池扩到 44，种群 600，代数 250）

评价：净期望（做空 TP2%/SL1%）× 样本量惩罚，训练段评估
验证：测试段三段（前/中/后）+ 分币集中度 + 滑点 0.3% 后
输出：Top 20 组合及全部验证数字
"""
import os, gc, time, random
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 50; FEE = 0.0004
TP, SL = 0.02, 0.01

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_evolve_big · 进化第二轮（信号池 44 / 种群 600 / 250 代）')
log('=' * 100)

FS, FMX, FMN, TS, SS = [], [], [], [], []
sym_id = {}
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
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
        r240, r1440 = ret(240), ret(1440)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().values
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,vstd1440,hour])
        fmax_rel = fmax / cc - 1.0
        fmin_rel = fmin / cc - 1.0
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmax_rel) & np.isfinite(fmin_rel)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        FS.append(F[idx].astype('float32'))
        FMX.append(fmax_rel[idx].astype('float32'))
        FMN.append(fmin_rel[idx].astype('float32'))
        TS.append(ot[idx]); SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(FS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN)
t = np.concatenate(TS); s = np.concatenate(SS)
del FS, FMX, FMN, TS, SS; gc.collect()
r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,vstd1440,hour = [X[:,i] for i in range(12)]
log(f'样本 {len(X):,}  币 {len(sym_id)}')

cut = np.median(t); trm = t <= cut; tem = t > cut
# 胜负标签（做空 TP2/SL1，60 分钟）
W = (fmn <= -TP) & (fmx < SL)
L = (fmx >= SL)
log(f'训练段 {trm.sum():,}  测试段 {tem.sum():,}  做空赢率基线 {W[trm].mean():.1%}')

def qz(a, p): return float(np.nanpercentile(a[trm], p))
SIG = [
 ('r1<-1%', r1 < -0.01), ('r1>+1%', r1 > 0.01),
 ('r5<-2%', r5 < -0.02), ('r5>+3%', r5 > 0.03),
 ('r15<-2%', r15 < -0.02), ('r15<-5%', r15 < -0.05), ('r15>+2%', r15 > 0.02),
 ('r60<-5%', r60 < -0.05), ('r60>+5%', r60 > 0.05), ('r60>+10%', r60 > 0.10),
 ('r240<-10%', r240 < -0.10), ('r240>+5%', r240 > 0.05), ('r240>+10%', r240 > 0.10),
 ('r1440<-20%', r1440 < -0.20), ('r1440>+20%', r1440 > 0.20),
 ('volr>1.5', volr > 1.5), ('volr>3', volr > 3), ('volr>6', volr > 6),
 ('volr<0.5', volr < 0.5), ('volr<0.3', volr < 0.3),
 ('vstd>p50', vstd > qz(vstd,50)), ('vstd>p70', vstd > qz(vstd,70)),
 ('vstd>p90', vstd > qz(vstd,90)), ('vstd>p97', vstd > qz(vstd,97)), ('vstd<p50', vstd < qz(vstd,50)),
 ('vstd1440>p90', vstd1440 > qz(vstd1440,90)),
 ('tbr>0.55', tbr > 0.55), ('tbr>0.6', tbr > 0.6), ('tbr>0.7', tbr > 0.7),
 ('tbr<0.4', tbr < 0.4), ('tbr<0.45', tbr < 0.45),
 ('trr>1', trr > 1), ('trr>3', trr > 3), ('trr<0.5', trr < 0.5), ('trr<0.3', trr < 0.3),
 ('hour 0-6', (hour >= 0) & (hour < 6)), ('hour 6-12', (hour >= 6) & (hour < 12)),
 ('hour 12-18', (hour >= 12) & (hour < 18)), ('hour 18-24', (hour >= 18) & (hour < 24)),
 ('r1>0', r1 > 0), ('r1<0', r1 < 0),
 ('r5>0', r5 > 0), ('r5<0', r5 < 0),
]
NS = len(SIG)
MASKS = np.column_stack([m for _, m in SIG]).astype(np.float32)   # (N,NS)
Mtr = MASKS[trm]; Mte = MASKS[tem]
wtr = W[trm].astype(np.float64); ltr = L[trm].astype(np.float64)
wte = W[tem].astype(np.float64); lte = L[tem].astype(np.float64)
log(f'信号 {NS} 个')

def best_genes(g):
    """返回 (净期望, 样本数)"""
    if len(g) == 0: return -9.0, 0
    sel = np.prod(Mtr[:, g], axis=1) > 0.5 if len(g) <= 8 else (Mtr[:, g].sum(axis=1) >= len(g))
    n = sel.sum()
    if n < 120: return -9.0, int(n)
    w = wtr[sel].sum(); l = ltr[sel].sum()
    if w + l == 0: return -9.0, int(n)
    return float((w*TP - l*SL)/n - FEE), int(n)

def ev(g):
    e, n = best_genes(g)
    if e < -1: return e
    return e * min(1.0, n / 400.0)

POP, GENS = 600, 250
random.seed(23); np.random.seed(23)
pop = [sorted(random.sample(range(NS), random.randint(1, 6))) for _ in range(POP)]
best = (-9, None)
t_evo = time.time()
for gen in range(GENS):
    scored = [(ev(ind), ind) for ind in pop]
    scored.sort(key=lambda x: -x[0])
    if scored[0][0] > best[0]:
        best = (scored[0][0], list(scored[0][1]))
    elites = [list(ind) for _, ind in scored[:20]]
    newpop = elites[:]
    while len(newpop) < POP:
        a = max(random.sample(scored[:120], 3), key=lambda x: x[0])[1]
        b = max(random.sample(scored[:120], 3), key=lambda x: x[0])[1]
        child = sorted(set(a) | set(b)); random.shuffle(child)
        child = sorted(child[:max(1, (len(a)+len(b))//2)])
        if random.random() < 0.35:
            if random.random() < 0.5 and len(child) > 1:
                child.remove(random.choice(child))
            elif len(child) < 12:
                child.append(random.randrange(NS)); child = sorted(set(child))
        newpop.append(child)
    pop = newpop
    if gen % 50 == 0:
        log(f'  gen {gen+1:>3}/{GENS}  best={scored[0][0]:+.4f}  [{len(scored[0][1])} 基因]  {time.time()-t_evo:.0f}s')
log(f'进化耗时 {time.time()-t_evo:.0f}s')

def genes_str(g):
    return '+'.join(SIG[i][0] for i in g)

# 收集 Top 组合（按测试段净期望，n>=100）
cands = {}
for ind in pop:
    g = tuple(ind)
    if g in cands: continue
    cands[g] = ev(ind)
ordered = sorted(cands.items(), key=lambda x: -x[1])[:200]

log('\n' + '=' * 100)
log('【测试段验证】Top 20（n>=100）')
log('=' * 100)
log(f'  {"组合":<70s}{"n":>7s}{"赢率":>8s}{"净期望":>9s}{"滑点0.3%":>10s}')
shown = 0
for g, _ in ordered:
    if shown >= 20: break
    sel = np.prod(Mte[:, list(g)], axis=1) > 0.5 if len(g) <= 8 else (Mte[:, list(g)].sum(axis=1) >= len(g))
    n = sel.sum()
    if n < 100: continue
    w = wte[sel].sum(); l = lte[sel].sum()
    wr = w / n; net = (w*TP - l*SL)/n - FEE
    log(f'  {genes_str(g)[:68]:<70s}{n:>7,}{wr:>8.1%}{net:>+9.3%}{net-0.003:>+10.3%}')
    shown += 1

log('\n' + '=' * 100)
log('【三段验证】Top5 组合的测试段前/中/后')
log('=' * 100)
med1 = np.median(t[tem]); med2 = np.median(t[tem & (t > med1)])
seg = {'前': tem & (t <= med1), '中': tem & (t > med1) & (t <= med2), '后': tem & (t > med2)}
shown = 0
for g, _ in ordered:
    if shown >= 5: break
    sel_all = np.prod(Mte[:, list(g)], axis=1) > 0.5 if len(g) <= 8 else (Mte[:, list(g)].sum(axis=1) >= len(g))
    if sel_all.sum() < 100: continue
    log(f'\n  {genes_str(g)}')
    for name, mm in seg.items():
        sel = (np.prod(MASKS[:, list(g)], axis=1) > 0.5) & mm if len(g) <= 8 else ((MASKS[:, list(g)].sum(axis=1) >= len(g)) & mm)
        n = sel.sum()
        if n == 0: continue
        w = W[sel].sum(); l = L[sel].sum()
        if w + l == 0: continue
        log(f'    {name}段 n={n:>6,} 赢率 {w/(w+l):>6.1%} 净 {(w*TP-l*SL)/n-FEE:>+8.3%}')
    shown += 1

log('\n' + '=' * 100)
log('【判读】')
log('=' * 100)
log('  · 三段同号为正 + 滑点0.3%后仍为正 = 可作资产候选')
log('  · 若只有一个组合满足 = 进化在该信号池内已收敛（还需换信号池）')

out = os.path.join(OUT, 'evolve_big_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
