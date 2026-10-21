# -*- coding: utf-8 -*-
"""gambler_exhaust56 · 穷举 5~6 条件组合（在 1~4 已有结果之上补全）

C(43,5)=962,598  C(43,6)=6,096,454
内存控制：k>=5 用堆保留 Top5000，不存全量
进度：每 50 万组合打印一次
"""
import os, gc, time, itertools, heapq
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
log('gambler_exhaust56 · 穷举 5~6 条件组合')
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
W = (fmn <= -TP) & (fmx < SL)
L = (fmx >= SL)

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
log(f'信号 {NS} 个')

def pack(m):
    return np.packbits(np.ascontiguousarray(m.T), axis=1)
Mtr = np.column_stack([m[trm] for _, m in SIG])
Mte = np.column_stack([m[tem] for _, m in SIG])
Ptr = pack(Mtr); Pte = pack(Mte)
wtr = W[trm]; ltr = L[trm]; wte = W[tem]; lte = L[tem]
Pwtr = np.packbits(wtr.astype(np.uint8)); Pltr = np.packbits(ltr.astype(np.uint8))
Pwte = np.packbits(wte.astype(np.uint8)); Plte = np.packbits(lte.astype(np.uint8))
try:
    _bc = np.bitwise_count
    def bc(a): return int(_bc(a).sum())
    log('popcount: np.bitwise_count')
except AttributeError:
    TAB = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    def bc(a): return int(TAB[a].sum())
    log('popcount: 查表')

def eval_packed(P, idxs, Pw, Pl):
    m = P[idxs[0]].copy()
    for i in idxs[1:]:
        m &= P[i]
    n = bc(m); w = bc(m & Pw); l = bc(m & Pl)
    return n, w, l

t_enum = time.time()
HEAP = []          # (net, n, w, l, combo)，k>=5 保 Top5000
total = 0
for k in range(5, 7):
    cnt = 0
    for combo in itertools.combinations(range(NS), k):
        n, w, l = eval_packed(Ptr, combo, Pwtr, Pltr)
        cnt += 1; total += 1
        if n < 200: continue
        net = (w*TP - l*SL)/n - FEE
        heapq.heappush(HEAP, (float(net), n, w, l, combo))
        if len(HEAP) > 5000:
            heapq.heappop(HEAP)
        if cnt % 500000 == 0:
            log(f'  k={k} 已 {cnt:,} 组合  累计 {total:,}  {time.time()-t_enum:.0f}s')
    log(f'  k={k} 枚举完  {time.time()-t_enum:.0f}s  堆内 {len(HEAP)}')
log(f'穷举完成，共 {total:,} 个组合，耗时 {time.time()-t_enum:.0f}s')

def gs(c): return '+'.join(SIG[i][0] for i in c)

rows = heapq.nlargest(5000, HEAP)

log('\n' + '=' * 100)
log('【训练段 Top30（5~6 条件）】')
log('=' * 100)
log(f'  {"组合":<80s}{"n":>7s}{"净期望":>9s}{"赢率":>8s}{"触发率":>8s}')
for net, n, w, l, c in rows[:30]:
    log(f'  {gs(c)[:78]:<80s}{n:>7,}{net:>+9.3%}{w/max(n,1):>8.1%}{(w+l)/n:>8.1%}')

log('\n' + '=' * 100)
log('【测试段验证 Top30（n>=100）】')
log('=' * 100)
log(f'  {"组合":<80s}{"n":>7s}{"赢率":>8s}{"触发率":>8s}{"净期望":>9s}{"滑点0.3%":>10s}')
ver = []
for net, n, w, l, c in rows:
    n2, w2, l2 = eval_packed(Pte, c, Pwte, Plte)
    if n2 < 100: continue
    net2 = (w2*TP - l2*SL)/n2 - FEE
    ver.append((float(net2), n2, w2, l2, c))
ver.sort(key=lambda x: -x[0])
for net2, n2, w2, l2, c in ver[:30]:
    log(f'  {gs(c)[:78]:<80s}{n2:>7,}{w2/max(n2,1):>8.1%}{(w2+l2)/n2:>8.1%}{net2:>+9.3%}{net2-0.003:>+10.3%}')

log('\n' + '=' * 100)
log('【三段验证 Top10】')
log('=' * 100)
med1 = np.median(t[tem]); med2 = np.median(t[tem & (t > med1)])
seg_masks = [('前', tem & (t <= med1)), ('中', tem & (t > med1) & (t <= med2)), ('后', tem & (t > med2))]
for net2, n2, w2, l2, c in ver[:10]:
    log(f'\n  {gs(c)}')
    for nm, mm in seg_masks:
        sel = np.ones(mm.sum(), bool)
        for i in c:
            sel &= SIG[i][1][mm]
        n3 = sel.sum()
        if n3 == 0: continue
        w3 = (W[mm] & sel).sum(); l3 = (L[mm] & sel).sum()
        if w3 + l3 == 0: continue
        log(f'    {nm}段 n={n3:>6,} 赢率(w/n) {w3/max(n3,1):>6.1%} 净 {(w3*TP-l3*SL)/max(n3,1)-FEE:>+8.3%}')

log('\n' + '=' * 100)
log('【判读】')
log('=' * 100)
log('  · 5~6 条件空间已全枚举（无遗漏）')
log('  · 若 Top 与 1~4 条件结果同族 ⇒ 再往上加条件不再产生新机制')
log('  · 三段同号为正 + 滑点0.3%后为正 = 资产候选')

out = os.path.join(OUT, 'exhaust56_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
