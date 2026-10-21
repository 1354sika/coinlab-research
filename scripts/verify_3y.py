# -*- coding: utf-8 -*-
"""三年（实为 9 年）1m 全量验证：43 信号 × 出场网格，带时间切分 + 事件去重。

数据：E:\\binance\\klines_1m\\1m\\<SYMBOL>.csv（每币一文件，2017 起）
严谨性：
  · 特征只用当根及之前（回看量）
  · 出场用收盘价确认触线（不用 high/low 影线）
  · 时间切分：全样本时间中位数之前=训练段，之后=验证段，两段分别统计
  · 事件去重：同币相邻命中点间隔 >= GAP 根
  · 成本 0.15% 往返
"""
import os, sys, glob, time
from collections import defaultdict
import numpy as np
import pandas as pd

D = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
COST = 0.0015
MAXH = 240          # 最长持 4 小时（1m 根数）
STEP = 60           # 候选点采样：每小时一个
GAP = 60            # 同币事件去重间隔
MIN_BARS = 5000     # 少于 5000 根的币跳过（不足 3.5 天）
LIMIT_SYMS = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0=全部

TPs = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0.02', '0.03'])]
SLs = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.01', '0.02'])]

files = sorted(glob.glob(os.path.join(D, '*.csv')))
if LIMIT_SYMS:
    files = files[:LIMIT_SYMS]
print('files %d' % len(files), flush=True)

AGG = {}            # (sig, TP, SL, 30天块) -> [n, sum_r, n_win]  流式累加，不吃内存
BLKN = defaultdict(int)   # 30天块 -> 该块总笔数（用于定中位切点）
t0 = time.time()
done = 0
for fp in files:
    sym = os.path.basename(fp)[:-4]
    try:
        df = pd.read_csv(fp, usecols=['open_time', 'close', 'volume', 'quote_volume', 'trades', 'taker_base'])
    except Exception as e:
        print('SKIP %s %s' % (sym, e), flush=True)
        continue
    if len(df) < MIN_BARS:
        continue
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df = df.dropna(subset=['open_time'])
    for c in ('close', 'volume', 'quote_volume', 'trades', 'taker_base'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df = df.dropna()
    if len(df) < MIN_BARS:
        continue

    ot = df['open_time'].to_numpy().astype('int64')
    c = df['close'].to_numpy(np.float64)
    vol = df['volume'].to_numpy(np.float64)
    q = df['quote_volume'].to_numpy(np.float64)
    tr = df['trades'].to_numpy(np.float64)
    tb = df['taker_base'].to_numpy(np.float64)
    n = len(c)

    def ret(k):
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = c[k:] / c[:-k] - 1.0
        return r

    r1 = ret(1); r5 = ret(5); r15 = ret(15); r60 = ret(60); r240 = ret(240); r1440 = ret(1440)
    qma = pd.Series(q).rolling(60, min_periods=10).mean().to_numpy()
    volr = np.where(qma > 0, q / np.maximum(qma, 1e-12), np.nan)
    tbr = np.where(vol > 0, tb / np.maximum(vol, 1e-12), np.nan)
    trma = pd.Series(tr).rolling(60, min_periods=10).mean().to_numpy()
    trr = np.where(trma > 0, tr / np.maximum(trma, 1e-12), np.nan)
    vstd = pd.Series(r1).rolling(15, min_periods=5).std().to_numpy()
    vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().to_numpy()
    hour = ((ot // 3600000) % 24).astype('float64')

    F = np.column_stack([r1, r5, r15, r60, r240, r1440, volr, tbr, trr, vstd, vstd1440, hour])
    ok = np.isfinite(F).all(axis=1)
    idx = np.arange(1500, n - MAXH - 1, STEP)
    if len(idx) == 0:
        continue
    idx = idx[ok[idx]]
    if len(idx) == 0:
        continue

    # 事件去重
    keep = []
    last = -10**9
    for i in idx:
        if i - last >= GAP:
            keep.append(i); last = i
    idx = np.array(keep)
    if len(idx) == 0:
        continue

    sub = F[idx]
    r1_, r5_, r15_, r60_, r240_, r1440_, volr_, tbr_, trr_, vstd_, v1440_, hour_ = [sub[:, k] for k in range(12)]

    # 分位阈值用本币全历史（诚实边界注明）
    def qz(a, p):
        v = np.nanpercentile(a, p)
        return v if np.isfinite(v) else 0.0

    SIG = [
        ('r1<-1%', r1_ < -0.01), ('r5<-2%', r5_ < -0.02), ('r15<-2%', r15_ < -0.02),
        ('r15<-5%', r15_ < -0.05), ('r60<-5%', r60_ < -0.05), ('r240<-10%', r240_ < -0.10),
        ('r1440<-20%', r1440_ < -0.20), ('r15>+2%', r15_ > 0.02), ('r60>+5%', r60_ > 0.05),
        ('r1440>+20%', r1440_ > 0.20), ('volr>3', volr_ > 3), ('volr<0.5', volr_ < 0.5),
        ('vstd>p90', vstd_ > qz(vstd_, 90)), ('vstd<p50', vstd_ < qz(vstd_, 50)),
        ('vstd1440>p90', v1440_ > qz(v1440_, 90)),
        ('tbr>0.6', tbr_ > 0.6), ('tbr<0.4', tbr_ < 0.4), ('trr>1', trr_ > 1), ('trr<0.5', trr_ < 0.5),
        ('hour 0-6', (hour_ >= 0) & (hour_ < 6)), ('hour 6-12', (hour_ >= 6) & (hour_ < 12)),
        ('hour 12-18', (hour_ >= 12) & (hour_ < 18)), ('hour 18-24', (hour_ >= 18) & (hour_ < 24)),
    ]

    sig_flags = np.column_stack([m for _, m in SIG])
    sig_names = [s for s, _ in SIG]
    for k, i in enumerate(idx):
        flags_k = sig_flags[k]
        t_exit_base = int(ot[i])
        ep = c[i]
        seg = c[i + 1:i + 1 + MAXH]
        if len(seg) == 0:
            continue
        rel = seg / ep - 1.0
        # 只对命中的信号记录（省内存）
        hits = np.where(flags_k)[0]
        if len(hits) == 0:
            continue
        for TP in TPs:
            hu = np.where(rel >= TP)[0]
            ju = hu[0] if len(hu) else 10**9
            for SL in SLs:
                hd = np.where(rel <= -SL)[0]
                jd = hd[0] if len(hd) else 10**9
                j = min(ju, jd)
                if j == 10**9:
                    r = rel[-1] - COST
                    j = len(seg)          # 持满 MAXH 未触线：出场时间 = 最后一根
                elif ju <= jd:
                    r = TP - COST
                else:
                    r = -SL - COST
                texit = t_exit_base + (j + 1) * 60000
                blk = int(texit // 2592000000)      # 30 天一块，切点精度到块
                for h in hits:
                    k = (sig_names[h], TP, SL, blk)
                    a = AGG.get(k)
                    if a is None:
                        a = AGG[k] = [0, 0.0, 0]
                    a[0] += 1
                    a[1] += r
                    if r > 0:
                        a[2] += 1
                    BLKN[blk] += 1

    done += 1
    if done % 20 == 0:
        print('  [%d/%d] %s bars=%d rows=%d  %.0fs' % (done, len(files), sym, n, sum(BLKN.values()), time.time() - t0), flush=True)

print('done %d syms  rows %d  %.0fs' % (done, sum(BLKN.values()), time.time() - t0), flush=True)

# ---------------- 汇总（流式聚合，不吃内存） ----------------
tot = sum(BLKN.values())
print('agg buckets %d  rows %d  %.0fs' % (len(AGG), tot, time.time() - t0), flush=True)

# 中位切点：按 30 天块的累计笔数取 50% 分位
acc = 0
cut_blk = None
for b in sorted(BLKN):
    acc += BLKN[b]
    if acc >= tot / 2.0:
        cut_blk = b
        break
if cut_blk is None:
    cut_blk = max(BLKN) if BLKN else 0

stat = defaultdict(lambda: [0, 0.0, 0, 0, 0.0, 0])   # (sig,TP,SL) -> [ntr,sum_tr,wtr,nva,sum_va,wva]
for (sig, TP, SL, b), a in AGG.items():
    s = stat[(sig, TP, SL)]
    if b <= cut_blk:
        s[0] += a[0]; s[1] += a[1]; s[2] += a[2]
    else:
        s[3] += a[0]; s[4] += a[1]; s[5] += a[2]

BLKMS = 2592000000
tmin = min(BLKN) * BLKMS if BLKN else 0
tmax = (max(BLKN) + 1) * BLKMS - 1 if BLKN else 0
cutms = (cut_blk + 1) * BLKMS - 1


def _d(ms):
    """安全的时间戳格式化（防溢出/非法值）"""
    try:
        ms = int(ms)
        if ms <= 0 or ms > 4102444800000:
            return '?'
        return pd.to_datetime(ms, unit='ms').strftime('%Y-%m-%d')
    except Exception:
        return '?'

L = []
L.append('1m 全量验证（E 盘 1m 数据，每币一文件；流式聚合，不内存累积）')
L.append('=' * 116)
L.append('出场：收盘价确认触线｜成本 %.2f%% 往返｜最长持 %d 根｜事件去重 GAP=%d 根｜采样 STEP=%d 根' % (
    COST * 100, MAXH, GAP, STEP))
L.append('样本 %d 个币｜展开后记录 %d 条｜时间 %s ~ %s' % (
    done, tot,
    _d(tmin), _d(tmax)))
L.append('时间切分点 %s（之前=训练段，之后=验证段；按 30 天块对齐）' % _d(cutms))
L.append('')

L.append('%-12s %-7s %-7s %8s %11s %8s %11s %8s' % (
    '信号', 'TP', 'SL', '训练n', '训练均值', '验证n', '验证均值', '标记'))
L.append('-' * 116)

best = []
for (sig, TP, SL), s in sorted(stat.items()):
    ntr, str_, wtr, nva, sva, wva = s
    if ntr < 30 or nva < 30:
        continue
    mtr = str_ / ntr
    mva = sva / nva
    both = (mtr > 0) and (mva > 0)
    L.append('%-12s %-7.1f%% %-7.1f%% %8d %+11.4f%% %8d %+11.4f%% %8s' % (
        sig, TP * 100, SL * 100, ntr, mtr * 100, nva, mva * 100, '稳健' if both else ''))
    best.append((both, mva, sig, TP, SL, ntr, mtr, nva, wva / max(nva, 1)))

L.append('')
L.append('--- 两段同号为正（真正稳健）的信号 ---')
stable = [x for x in best if x[0]]
stable.sort(key=lambda x: -x[1])
if stable:
    for both, mva, sig, TP, SL, ntr, mtr, nva, wva in stable:
        L.append('  %-12s TP=%.1f%% SL=%.1f%%  训练 %+.4f%% (n=%d)  验证 %+.4f%% (n=%d, 胜率 %.1f%%)' % (
            sig, TP * 100, SL * 100, mtr * 100, ntr, mva * 100, nva, wva * 100))
else:
    L.append('  （无）—— 没有任何信号在训练段和验证段同时为正')

L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘价确认触线，不用 high/low；实盘还需排队与滑点。')
L.append('2. 分位阈值（vstd/p50、p90 等）用各币全历史算，含轻微前视。')
L.append('3. 时间切分按 30 天块对齐（近中位数），不是按日历等分。')
L.append('4. 事件去重只去同币相邻命中，跨币同波未去重。')
L.append('5. 数据来自 Binance data-api（现货）；未含下架币；本次进入统计 %d 个币。' % done)

out = os.path.join(EVI, 'verify_3y_full.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
