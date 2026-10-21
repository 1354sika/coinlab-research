# -*- coding: utf-8 -*-
"""
sigma_ratio.py — 验证用户的质疑「是不是方式不对」

上一轮 3σ/3σ 全亏。最可能的方式错误：
  止盈和止损用了**同一个系数** → 盈亏比 1:1
  → 需要胜率 >50% 才不亏（扣 0.25% 成本后需要更高）
  → 实测胜率 52.6%，刚好被成本吃穿

而历史所有赚钱的组合（TP3%/SL1%）都是 **3:1 盈亏比**。

本轮把 TP 和 SL 拆开扫：
  TP = k_tp · σ
  SL = k_sl · σ
  k_tp ∈ {1, 1.5, 2, 3, 5, 8}
  k_sl ∈ {0.3, 0.5, 1, 1.5, 2, 3}
共 36 组，每组出单笔均值 + 池模拟 + 时间切分。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'sigma_ratio.txt')

COST  = 0.0025
H     = 480
STEP  = 60
DROP  = 0.02
VOLR  = 3.0
SIG_N = 240

K_TP = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
K_SL = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

L = []
def p(s=''):
    print(s)
    L.append(s)

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    p('=' * 112)
    p('止盈止损分开扫  file=%s  TP=k_tp·σ  SL=k_sl·σ  COST=%.2f%%' % (fname, COST * 100))
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs' % (len(d), time.time() - t0))

    # 收集所有入场点的路径（只存必要信息，供后续 36 组复用）
    # 每笔: (t_entry, sym, sigma, entry, 路径的 high 序列前 H, low 序列前 H, close 序列前 H)
    paths = []
    syms = sorted(d.symbol.unique())
    for si, sym in enumerate(syms):
        g = d[d.symbol == sym].sort_values('open_time')
        c = g.close.to_numpy(np.float64)
        h = g.high.to_numpy(np.float64)
        lo = g.low.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
        tt = g.open_time.to_numpy(np.int64)
        n = len(c)
        if n < SIG_N + H + 300:
            continue
        lr = np.zeros(n, np.float64)
        lr[1:] = np.log(c[1:] / c[:-1])
        sig = pd.Series(lr).rolling(SIG_N, min_periods=SIG_N // 2).std().to_numpy()
        r24 = np.zeros(n, np.float64)
        r24[24:] = c[24:] / c[:-24] - 1.0
        vma = pd.Series(v).rolling(96, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        cand = np.flatnonzero((r24 <= -DROP) & (volr >= VOLR) & (sig > 0))
        if len(cand) == 0:
            continue
        keep, last = [], -10**9
        for i in cand:
            if i - last >= STEP and i + H < n:
                keep.append(i); last = i
        for i in keep:
            sg = sig[i]
            if not (np.isfinite(sg) and sg > 0 and c[i] > 0):
                continue
            paths.append((int(tt[i]), sym, float(sg), float(c[i]),
                          h[i + 1:i + 1 + H].astype(np.float32),
                          lo[i + 1:i + 1 + H].astype(np.float32),
                          c[i + 1:i + 1 + H].astype(np.float32)))
        if (si + 1) % 25 == 0:
            p('  [%d/%d] paths=%d  %.0fs' % (si + 1, len(syms), len(paths), time.time() - t0))
    p('done paths=%d  %.0fs' % (len(paths), time.time() - t0))
    p()

    # 每笔先算好「每个 k 组合」的收益
    res = {}   # (kt, ks) -> list of (t_entry, t_exit, r)
    for kt in K_TP:
        for ks in K_SL:
            res[(kt, ks)] = []
    for (t_in, sym, sg, entry, ph, pl, pc) in paths:
        m = len(pc)
        if m < 5:
            continue
        for kt in K_TP:
            up = entry * (1.0 + kt * sg)
            ht = ph >= up
            it = int(np.argmax(ht)) if ht.any() else 10**9
            for ks in K_SL:
                dn = entry * (1.0 - ks * sg)
                hl = pl <= dn
                il = int(np.argmax(hl)) if hl.any() else 10**9
                if it < 10**9 and it <= il:
                    r = kt * sg; j = it
                elif il < 10**9:
                    r = -ks * sg; j = il
                else:
                    r = pc[-1] / entry - 1.0; j = m - 1
                res[(kt, ks)].append((t_in, t_in + (j + 1) * 60000, float(r - COST)))

    p('=== 一、单笔均值矩阵（已扣 %.2f%%，行=k_tp，列=k_sl）' % (COST * 100))
    hdr = '  %-8s' % '' + ''.join('%9.1f' % k for k in K_SL)
    p(hdr)
    best = (None, -9)
    for kt in K_TP:
        row = '  kt=%-5.1f' % kt
        for ks in K_SL:
            arr = np.array([x[2] for x in res[(kt, ks)]], np.float64)
            v = arr.mean() * 100 if len(arr) else -99
            row += '%9.3f' % v
            if v > best[1]:
                best = ((kt, ks), v)
        p(row)
    p('  => 单笔最优：k_tp=%.1f  k_sl=%.1f  均值 %.4f%%' % (best[0][0], best[0][1], best[1]))

    # 池模拟
    def pool_sim(recs, N):
        eq, peak, mdd = 1.0, 1.0, 0.0
        pending = []
        for (t_in, t_exit, r) in sorted(recs, key=lambda x: x[1]):
            freed = [s for s in pending if s[0] <= t_in]
            pending = [s for s in pending if s[0] > t_in]
            for s in freed:
                eq *= (1.0 + s[1] / N)
                peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
            if len(pending) < N:
                pending.append((t_exit, r))
        for s in pending:
            eq *= (1.0 + s[1] / N)
            peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
        return eq, mdd

    p()
    p('=== 二、池模拟：只看单笔均值为正的组合（池 N=20）')
    p('  %-8s %-8s %8s %10s %11s %10s' % ('k_tp', 'k_sl', '笔数', '均值/笔', '期末(池20)', '回撤'))
    p('  ' + '-' * 60)
    winners = []
    for kt in K_TP:
        for ks in K_SL:
            recs = res[(kt, ks)]
            if not recs:
                continue
            arr = np.array([x[2] for x in recs], np.float64)
            if arr.mean() <= 0:
                continue
            eq, mdd = pool_sim(recs, 20)
            winners.append((kt, ks, len(recs), arr.mean(), eq, mdd))
    winners.sort(key=lambda x: -x[4])
    for (kt, ks, n, mu, eq, mdd) in winners[:15]:
        p('  %-8.1f %-8.1f %8d %9.4f%% %10.4fx %9.2f%%' % (kt, ks, n, mu * 100, eq, mdd * 100))
    if not winners:
        p('  (没有一组单笔均值为正)')

    p()
    p('=== 三、时间切分：对前 5 名做前后段检验')
    if winners:
        p('  %-8s %-8s %10s %10s %11s %11s' % ('k_tp', 'k_sl', '前段均值', '后段均值', '前段期末', '后段期末'))
        p('  ' + '-' * 62)
        for (kt, ks, n, mu, eq, mdd) in winners[:5]:
            recs = res[(kt, ks)]
            ts = [x[0] for x in recs]
            tmid = int(np.median(ts))
            a = [x for x in recs if x[0] <= tmid]
            b = [x for x in recs if x[0] > tmid]
            aa = np.array([x[2] for x in a], np.float64)
            bb = np.array([x[2] for x in b], np.float64)
            ea = pool_sim(a, 20)[0] if len(a) > 30 else 0
            eb = pool_sim(b, 20)[0] if len(b) > 30 else 0
            p('  %-8.1f %-8.1f %9.4f%% %9.4f%% %10.4fx %10.4fx'
              % (kt, ks, aa.mean() * 100, bb.mean() * 100, ea, eb))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径，止盈假设能成交（成交率未计）。')
    p('2. σ 用过去 240 根 1m，无前视；k_tp/k_sl 是全样本扫出来的（含样本内）。')
    p('3. 同一根内 TP/SL 同触按 TP 优先（乐观）。')
    p('4. 入场点每 60 根（1h）取一个，样本重叠严重。')
    p('5. 池模拟用 t_exit 结算，未模拟保证金/费率/最小下单量。')
    p('6. 已下架归零币不在池内（幸存者偏差）。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
