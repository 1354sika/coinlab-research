# -*- coding: utf-8 -*-
"""时间切分验证：把「短窗口急跌」信号在时间上切三段，看是否同号为正。

严谨性要点：
  · 特征只用当根及之前（r1/r5/r15 是回看量，无前视）
  · 出场用收盘价确认（不用 high/low）
  · 每段独立统计，三段同号为正才算稳健
  · 事件去重：同币相邻命中点间隔 >= GAP 根（默认 60）
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'
FN = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_502.csv'
COST = 0.0015
MAXH = 240
STEP = 60
GAP = 60

t0 = time.time()
df = pd.read_csv(os.path.join(DATA, FN), usecols=['symbol', 'open_time', 'close', 'volume'])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
df = df.dropna(subset=['open_time'])
df['open_time'] = df['open_time'].astype('int64')
for c in ('close', 'volume'):
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
df = df.dropna()
print('rows %d  %.1fs' % (len(df), time.time() - t0))

recs = []
paths = {}
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    c = g['close'].to_numpy(np.float64)
    t = g['open_time'].to_numpy()
    n = len(c)
    if n < 1500:
        continue
    paths[sym] = c
    def ret(k):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0; return r
    r1, r5, r15 = ret(1), ret(5), ret(15)
    ok = np.isfinite(r1) & np.isfinite(r5) & np.isfinite(r15)
    idx = np.arange(1440, n, STEP)
    idx = idx[ok[idx]]
    if len(idx) == 0:
        continue
    for i in idx:
        recs.append((int(t[i]), sym, int(i), float(r1[i]), float(r5[i]), float(r15[i])))

C = pd.DataFrame(recs, columns=['t_ms', 'sym', 'i', 'r1', 'r5', 'r15'])
del df; import gc; gc.collect()
print('candidates %d  syms %d  %.1fs' % (len(C), C['sym'].nunique(), time.time() - t0))

SIGS = [('r1<-1%', C.r1 < -0.01), ('r5<-2%', C.r5 < -0.02), ('r15<-2%', C.r15 < -0.02)]
GRIDS = [(0.02, 0.02), (0.03, 0.01), (0.03, 0.02)]

def trades_for(mask, TP, SL):
    sub = C[mask]
    if len(sub) == 0:
        return None
    out = []
    for sym, g in sub.groupby('sym', sort=False):
        c = paths[sym]
        idxs = g['i'].to_numpy()
        tmss = g['t_ms'].to_numpy()
        # 事件去重：同币相邻命中间隔 >= GAP
        keep = [0]
        for k in range(1, len(idxs)):
            if idxs[k] - idxs[keep[-1]] >= GAP:
                keep.append(k)
        for k in keep:
            i = int(idxs[k]); tms = int(tmss[k])
            ep = c[i]
            up = ep * (1 + TP); dn = ep * (1 - SL)
            end = min(i + MAXH, len(c) - 1)
            seg = c[i + 1:end + 1]
            if len(seg) == 0:
                continue
            hu = np.where(seg >= up)[0]; hd = np.where(seg <= dn)[0]
            ju = hu[0] if len(hu) else 10**9
            jd = hd[0] if len(hd) else 10**9
            j = min(ju, jd)
            if j == 10**9:
                px = seg[-1]
            elif ju <= jd:
                px = up
            else:
                px = dn
            out.append((tms + (j + 1) * 60000, px / ep - 1.0 - COST))
    if not out:
        return None
    out.sort()
    return np.array([x[1] for x in out]), np.array([x[0] for x in out])

L = []
L.append('时间切分验证  file=%s' % FN)
L.append('=' * 100)
L.append('出场：收盘确认｜成本 %.2f%% 往返｜最长持 %d 根｜事件去重 GAP=%d 根' % (COST * 100, MAXH, GAP))
L.append('样本 %d 个候选 / %d 币' % (len(C), C['sym'].nunique()))
L.append('')
L.append('%-10s %-8s %6s %7s %8s %8s %8s %8s' % ('信号', 'TP/SL', '段', '笔数', '胜率', '单笔均值', '段终值', '结论'))
L.append('-' * 100)

for name, mask in SIGS:
    for TP, SL in GRIDS:
        r = trades_for(mask, TP, SL)
        if r is None or len(r[0]) < 30:
            L.append('%-10s %-8s  (样本不足)' % (name, '%.0f/%.0f%%' % (TP*100, SL*100)))
            continue
        rr, tt = r
        q = np.array_split(np.arange(len(rr)), 3)
        segs = []
        for k, ix in enumerate(q):
            rs = rr[ix]
            segs.append((len(rs), (rs > 0).mean(), rs.mean(), float(np.prod(1.0 + rs))))
        allpos = all(s[2] > 0 for s in segs)
        for k, (n_, w_, m_, e_) in enumerate(segs):
            L.append('%-10s %-8s %6d %7d %7.1f%% %+8.4f%% %8.3f %8s' % (
                name if k == 0 else '', ('%.0f/%.0f%%' % (TP*100, SL*100)) if k == 0 else '',
                k + 1, n_, w_ * 100, m_ * 100, e_, ('稳健' if allpos else ('同号正' if m_ > 0 else '')) if k == 0 else ('OK' if m_ > 0 else 'NEG')))
        L.append('')

L.append('--- 判读 ---')
L.append('  · 三段单笔均值全部 > 0 = 稳健（不是某一段行情的产物）')
L.append('  · 有一两段为负 = 不稳健，不能用')
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘确认触线，不用 high/low；实盘还需排队与滑点。')
L.append('2. 时间切分按样本数三等分，不是按日历等分。')
L.append('3. 单一数据集，含样本内参数（TP/SL 是前一轮扫出来的）。')

out = os.path.join(EVI, 'split_validate.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
