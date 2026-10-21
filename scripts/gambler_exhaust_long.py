# -*- coding: utf-8 -*-
"""gambler_exhaust_long · 做多（低买高卖）穷举 1~4 条件组合

与做空版唯一区别：胜负标签反过来
  做多赢 = 60 分钟内先涨 TP(2%) 且未跌 SL(1%)
  做多输 = 先跌 SL(1%)
口径：赢率=w/n（含未触发按 0），双触保守算输
"""
import os, gc, time, itertools
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
log('gambler_exhaust_long · 做多（低买高卖）穷举 1~4 条件')
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
        fmx_rel = fmax / cc - 1.0
        fmn_rel = fmin / cc - 1.0
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmx_rel) & np.isfinite(fmn_rel)
        idx = np.arange(300, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        FS.append(F[idx].astype('float32'))
        FMX.append(fmx_rel[idx].astype('float32'))
        FMN.append(fmn_rel[idx].astype('float32'))
        TS.append(ot[idx]); SS.append(np.full(len(idx), sym_id.setdefault(sym, len(sym_id)), dtype='int32'))
    del ch; gc.collect()

X = np.vstack(FS); fmx = np.concatenate(FMX); fmn = np.concatenate(FMN)
t = np.concatenate(TS); s = np.concatenate(SS)
del FS, FMX, FMN, TS, SS; gc.collect()
r1,r5,r15,r60,r240,r1440,volr,tbr,trr,vstd,vstd1440,hour = [X[:,i] for i in range(12)]
log(f'样本 {len(X):,}  币 {len(sym_id)}')

cut = np.median(t); trm = t <= cut; tem = t > cut
# 做多标签
W = (fmx >= TP) & (fmn > -SL)
L = (fmn <= -SL)
log(f'训练段 {trm.sum():,}  测试段 {tem.sum():,}')
log(f'做多基线（全样本，测试段）：赢率 {W[tem].mean():.2%}  输率 {L[tem].mean():.2%}  净 {((W[tem].sum()*TP-L[tem].sum()*SL)/tem.sum()-FEE):+.3%}')
log(f'（对照）做空基线：赢率 {(fmn[tem]<=-TP).mean() - (fmn[tem]<=-TP).mean()*0:.2%}')

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
except AttributeError:
    TAB = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    def bc(a): return int(TAB[a].sum())

def eval_packed(P, idxs, Pw, Pl):
    m = P[idxs[0]].copy()
    for i in idxs[1:]:
        m &= P[i]
    return bc(m), bc(m & Pw), bc(m & Pl)

t_enum = time.time()
rows = []
for k in range(1, 5):
    for combo in itertools.combinations(range(NS), k):
        n, w, l = eval_packed(Ptr, combo, Pwtr, Pltr)
        if n < 200: continue
        net = (w*TP - l*SL)/n - FEE
        rows.append((float(net), n, w, l, combo))
    log(f'  k={k} 枚举完  {time.time()-t_enum:.0f}s  累计候选 {len(rows):,}')
rows.sort(key=lambda x: -x[0])
log(f'穷举完成，候选 {len(rows):,}，{time.time()-t_enum:.0f}s')

def gs(c): return '+'.join(SIG[i][0] for i in c)

log('\n' + '=' * 100)
log('【训练段 Top30（做多）】')
log('=' * 100)
log(f'  {"组合":<66s}{"n":>7s}{"净期望":>9s}{"赢率":>8s}{"触发率":>8s}')
for net, n, w, l, c in rows[:30]:
    log(f'  {gs(c)[:64]:<66s}{n:>7,}{net:>+9.3%}{w/max(n,1):>8.1%}{(w+l)/n:>8.1%}')

log('\n' + '=' * 100)
log('【测试段验证 Top30（n>=100）】')
log('=' * 100)
log(f'  {"组合":<66s}{"n":>7s}{"赢率":>8s}{"触发率":>8s}{"净期望":>9s}{"滑点0.3%":>10s}')
ver = []
for net, n, w, l, c in rows[:3000]:
    n2, w2, l2 = eval_packed(Pte, c, Pwte, Plte)
    if n2 < 100: continue
    net2 = (w2*TP - l2*SL)/n2 - FEE
    ver.append((float(net2), n2, w2, l2, c))
ver.sort(key=lambda x: -x[0])
for net2, n2, w2, l2, c in ver[:30]:
    log(f'  {gs(c)[:64]:<66s}{n2:>7,}{w2/max(n2,1):>8.1%}{(w2+l2)/n2:>8.1%}{net2:>+9.3%}{net2-0.003:>+10.3%}')

log('\n' + '=' * 100)
log('【三段验证 Top10】')
log('=' * 100)
med1 = np.median(t[tem]); med2 = np.median(t[tem & (t > med1)])
segs = [('前', tem & (t <= med1)), ('中', tem & (t > med1) & (t <= med2)), ('后', tem & (t > med2))]
for net2, n2, w2, l2, c in ver[:10]:
    log(f'\n  {gs(c)}')
    for nm, mm in segs:
        sel = np.ones(mm.sum(), bool)
        for i in c:
            sel &= SIG[i][1][mm]
        n3 = sel.sum()
        if n3 == 0: continue
        w3 = (W[mm] & sel).sum(); l3 = (L[mm] & sel).sum()
        log(f'    {nm}段 n={n3:>6,} 赢率(w/n) {w3/max(n3,1):>6.1%} 净 {(w3*TP-l3*SL)/max(n3,1)-FEE:>+8.3%}')

log('\n' + '=' * 100)
log('【判读】')
log('=' * 100)
log('  · 做多基线为负（与 bidir 一致：随机做多 -0.4% 左右）')
log('  · 若有组合测试段显著为正且三段同号 ⇒ 低买高卖存在可交易子集')
log('  · 穷举 1~4 条件无遗漏（136,697 个）')

out = os.path.join(OUT, 'exhaust_long_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
