# -*- coding: utf-8 -*-
"""gambler_evolve_verify · 验证进化出的组合 + 更大规模进化搜索

A 精确验证：r15>+2% & volr<0.5 & vstd>p90 & tbr>0.55 做空
  - 测试段切两半（各半赢率/期望）
  - 按币分组（是否集中在少数币）
  - 每天次数
  - 扩样本（STEP=50，样本翻倍）
B 更大进化：种群 400、代数 150，看是否有更稳的组合
"""
import os, gc, time, random
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60
STEP = 50
FEE = 0.0004

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_evolve_verify · A 验证组合（扩样本/两半/分币） + B 更大进化')
log('=' * 100)

XS, YW, YL, TS, SS = [], [], [], [], []
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
        win = (fmin <= cc * 0.98) & (fmax < cc * 1.01)
        lose = (fmax >= cc * 1.01)
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
        hour = ((ot[:m] // 3600000) % 24).astype('float64')
        F = np.column_stack([r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmax) & np.isfinite(fmin)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        XS.append(F[idx].astype('float32'))
        YW.append(win[idx]); YL.append(lose[idx]); TS.append(ot[idx])
        SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(XS); yw = np.concatenate(YW); yl = np.concatenate(YL)
t = np.concatenate(TS); s = np.concatenate(SS)
del XS, YW, YL, TS, SS; gc.collect()
r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,hour = [X[:,i] for i in range(11)]
log(f'样本 {len(X):,}  币 {len(sym_id)}  做空赢 {yw.mean():.1%}')

cut = np.median(t)
trm = t <= cut; tem = t > cut

# 训练段定标 p90
P90 = float(np.nanpercentile(vstd[trm], 90))
log(f'训练段 vstd p90 = {P90:.5f}')

# ---------- A 验证组合 ----------
COMBO = (r15 > 0.02) & (volr < 0.5) & (vstd > P90) & (tbr > 0.55)
log('\n' + '=' * 100)
log('【A】验证组合：r15>+2% & volr<0.5 & vstd>p90 & tbr>0.55（做空 +2%/-1%）')
log('=' * 100)

def stat(mask, name):
    n = mask.sum()
    if n == 0:
        log(f'{name:24s} n=0'); return
    w = (yw & mask).sum(); l = (yl & mask).sum()
    wr = w / n
    exp = (w * 0.02 - l * 0.01) / n - FEE
    log(f'{name:24s} n={n:>7,}  赢率 {wr:>6.1%}  赢 {w:>6,}  输 {l:>6,}  净期望 {exp:>+7.3%}')

stat(tem & COMBO, '测试段整体')
med2 = np.median(t[tem])
stat(tem & COMBO & (t <= med2), '测试段前半')
stat(tem & COMBO & (t > med2), '测试段后半')

# 分币集中度
sel = tem & COMBO
sv = s[sel]
if len(sv):
    cnt = pd.Series(sv).value_counts()
    log(f'  落在 {len(cnt)} 个币；单币最多 {cnt.iloc[0]} 笔（{cnt.iloc[0]/len(sv):.1%}）；前 5 币占 {cnt.iloc[:5].sum()/len(sv):.1%}')

# 每天次数
days = (t[tem].max() - t[tem].min()) / 86400000.0
log(f'  测试段 {days:.0f} 天，每天 {sel.sum()/days:.2f} 次（STEP={STEP} 口径，全量需 ×{STEP}）')
# 扩大阈值附近看稳健性
log('\n  阈值敏感性（测试段）：')
for vlo in [0.45, 0.5, 0.55, 0.6]:
    for vhi in [0.5, 0.6]:
        pass
log(f'  {"volr上限":>10s}{"tbr下限":>10s}{"n":>8s}{"赢率":>8s}{"净期望":>10s}')
for vu in [0.4, 0.5, 0.6]:
    for tl in [0.5, 0.55, 0.6]:
        m2 = (r15 > 0.02) & (volr < vu) & (vstd > P90) & (tbr > tl) & tem
        n = m2.sum()
        if n < 20:
            continue
        w = (yw & m2).sum(); l = (yl & m2).sum()
        log(f'  {vu:>10.2f}{tl:>10.2f}{n:>8,}{w/n:>8.1%}{(w*0.02-l*0.01)/n-FEE:>+10.3%}')

# ---------- B 更大进化 ----------
log('\n' + '=' * 100)
log('【B】更大进化：种群 400，代数 150')
log('=' * 100)

def qz(a, p): return float(np.nanpercentile(a, p))
SIG = [
 ('r15<-2%', r15 < -0.02), ('r15<-5%', r15 < -0.05), ('r15>+2%', r15 > 0.02),
 ('r60<-5%', r60 < -0.05), ('r60>+5%', r60 > 0.05),
 ('r240<-10%', r240 < -0.10), ('r240>+10%', r240 > 0.10),
 ('r1440<-20%', r1440 < -0.20), ('r1440>+20%', r1440 > 0.20),
 ('volr>3', volr > 3), ('volr>6', volr > 6), ('volr<0.5', volr < 0.5), ('volr<0.3', volr < 0.3),
 ('vstd>p70', vstd > qz(vstd,70)), ('vstd>p90', vstd > qz(vstd,90)), ('vstd>p97', vstd > qz(vstd,97)),
 ('tbr>0.55', tbr > 0.55), ('tbr<0.45', tbr < 0.45),
 ('trr>3', trr > 3), ('trr<0.5', trr < 0.5),
 ('hour 0-6', (hour >= 0) & (hour < 6)), ('hour 12-18', (hour >= 12) & (hour < 18)),
 ('r1>0', r1 > 0), ('r5>+3%', r5 > 0.03),
]
NS = len(SIG)
MASKS = np.column_stack([m for _, m in SIG])
Mtr = MASKS[trm]; Mte = MASKS[tem]
wtr = yw[trm]; ltr = yl[trm]; wte = yw[tem]; lte = yl[tem]
log(f'信号 {NS} 个，训练 {trm.sum():,}，测试 {tem.sum():,}')

def fitr(genes):
    if not genes: return -1.0, 0
    sel = np.ones(len(Mtr), bool)
    for g in genes: sel &= Mtr[:, g]
    n = sel.sum()
    if n < 120: return -1.0, n
    w = (wtr & sel).sum(); l = (ltr & sel).sum()
    if w + l == 0: return -1.0, n
    return float((w*0.02 - l*0.01)/n - FEE), int(n)

def ev(genes):
    e, n = fitr(genes)
    return e * min(1.0, n / 400.0)

POP, GENS = 400, 150
random.seed(11); np.random.seed(11)
pop = [sorted(random.sample(range(NS), random.randint(1, 5))) for _ in range(POP)]
best = None
t_evo = time.time()
for gen in range(GENS):
    scored = [(ev(ind), ind) for ind in pop]
    scored.sort(key=lambda x: -x[0])
    if best is None or scored[0][0] > best[0]:
        best = (scored[0][0], list(scored[0][1]))
    elites = [list(ind) for _, ind in scored[:12]]
    newpop = elites[:]
    while len(newpop) < POP:
        a = max(random.sample(scored[:80], 3), key=lambda x: x[0])[1]
        b = max(random.sample(scored[:80], 3), key=lambda x: x[0])[1]
        child = sorted(set(a) | set(b)); random.shuffle(child)
        child = sorted(child[:max(1, (len(a)+len(b))//2)])
        if random.random() < 0.35:
            if random.random() < 0.5 and len(child) > 1:
                child.remove(random.choice(child))
            elif len(child) < 10:
                child.append(random.randrange(NS)); child = sorted(set(child))
        newpop.append(child)
    pop = newpop
    if gen % 30 == 0 or gen == GENS-1:
        log(f'  gen {gen+1:>3}/{GENS}  best={scored[0][0]:+.4f}  genes={scored[0][1]}  {time.time()-t_evo:.0f}s')
log(f'进化耗时 {time.time()-t_evo:.0f}s')

# 测试段验证 Top8（按测试期望排序）
res = []
seen = set()
for _, ind in [(ev(ind), ind) for ind in pop]:
    key = tuple(ind)
    if key in seen: continue
    seen.add(key)
    sel = np.ones(len(Mte), bool)
    for g in ind: sel &= Mte[:, g]
    n = sel.sum()
    if n < 100: continue
    w = (wte & sel).sum(); l = (lte & sel).sum()
    res.append((n, w/n, (w*0.02-l*0.01)/n-FEE, ind, w, l))
res.sort(key=lambda x: -x[2])
log('\n【B 结果】测试段 Top8（按测试净期望排序，n>=100）')
log(f'{"组合":62s}{"n":>7s}{"赢率":>7s}{"净期望":>9s}')
log('-' * 110)
for n, wr, ex, ind, w, l in res[:8]:
    nm = '+'.join(SIG[g][0] for g in ind)
    log(f'{nm[:62]:62s}{n:>7,}{wr:>7.1%}{ex:>+9.3%}')

out = os.path.join(OUT, 'evolve_verify_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
