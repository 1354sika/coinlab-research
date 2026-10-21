# -*- coding: utf-8 -*-
"""
timegate_pool.py — 把 timegate 扫出的正期望格接进池模拟

timegate 结果（142 天，固定 TP3%/SL1%，扣 0.25%）：
  3m 跌 3%  -> +0.1913%/笔（214 笔）  <- 大样本最优
  15m 跌 3% -> +0.1654%/笔（316 笔）
  30m 跌 5% -> +0.1291%/笔（139 笔）
  60m 跌 3% -> +0.1209%/笔（295 笔）

本轮把这几格接进「单账号 N 仓并发、满仓跳过」的真实约束，看还剩多少。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'timegate_pool.txt')

COST  = 0.0025
TP    = 0.03
SL    = 0.01
HOURS = 8
VOLR  = 3.0

# (尺度分钟, 跌幅)
CASES = [(3, 0.03), (15, 0.03), (30, 0.05), (60, 0.03), (3, 0.05)]
POOLS = [1, 2, 5, 10, 20, 50]

L = []
def p(s=''):
    print(s)
    L.append(s)

def run_case(d, scale, drop):
    """返回 [(t_in, t_exit, r_net), ...]"""
    H = max(2, int(HOURS * 60 / scale))
    vol_win = 96
    recs = []
    for sym, g in d.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        t = g.open_time.to_numpy(np.int64)
        o = g.open.to_numpy(np.float64)
        h = g.high.to_numpy(np.float64)
        lo = g.low.to_numpy(np.float64)
        c = g.close.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
        n = len(c)
        if n < scale * (vol_win + H) + 600:
            continue
        bucket = t // (scale * 60000)
        cuts = np.flatnonzero(np.diff(bucket)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [n]))
        nb = len(starts)
        if nb < vol_win + H + 100:
            continue
        bo = o[starts]
        bh = np.array([h[a:b].max() for a, b in zip(starts, ends)], np.float64)
        bl = np.array([lo[a:b].min() for a, b in zip(starts, ends)], np.float64)
        bc = c[ends - 1]
        bv = np.array([v[a:b].sum() for a, b in zip(starts, ends)], np.float64)
        bt = t[starts]
        vma = pd.Series(bv).rolling(vol_win, min_periods=vol_win // 2).mean().to_numpy()
        volr = np.where(vma > 0, bv / vma, 1.0)
        r = np.zeros(nb, np.float64)
        r[1:] = bc[1:] / bc[:-1] - 1.0

        sig = np.flatnonzero((r <= -drop) & (volr >= VOLR))
        if len(sig) == 0:
            continue
        keep, last = [], -10**9
        STEP = max(1, scale)
        for i in sig:
            if i - last >= STEP and i + 1 + H < nb:
                keep.append(i); last = i
        for i in keep:
            entry = bo[i + 1] if bo[i + 1] > 0 else bc[i]
            if entry <= 0:
                continue
            seg_h = bh[i + 1:i + 1 + H]
            seg_l = bl[i + 1:i + 1 + H]
            seg_c = bc[i + 1:i + 1 + H]
            m = len(seg_c)
            if m < 3:
                continue
            up = entry * (1 + TP); dn = entry * (1 - SL)
            ht = seg_h >= up; hl = seg_l <= dn
            it = int(np.argmax(ht)) if ht.any() else 10**9
            il = int(np.argmax(hl)) if hl.any() else 10**9
            if it < 10**9 and it <= il:
                rr = TP; j = it
            elif il < 10**9:
                rr = -SL; j = il
            else:
                rr = seg_c[-1] / entry - 1.0; j = m - 1
            t_in = int(bt[i + 1])
            t_exit = int(bt[min(i + 1 + j, nb - 1)])
            recs.append((t_in, t_exit, float(rr - COST)))
    return recs

def pool_sim(recs, N):
    eq, peak, mdd = 1.0, 1.0, 0.0
    pending = []
    taken = 0
    for (t_in, t_exit, r) in sorted(recs, key=lambda x: x[1]):
        freed = [s for s in pending if s[0] <= t_in]
        pending = [s for s in pending if s[0] > t_in]
        for s in freed:
            eq *= (1.0 + s[1] / N)
            peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
        if len(pending) < N:
            pending.append((t_exit, r)); taken += 1
    for s in pending:
        eq *= (1.0 + s[1] / N)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return eq, mdd, taken

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    p('=' * 112)
    p('正期望格接池模拟  file=%s  TP=%.0f%%/SL=%.0f%%  COST=%.2f%%' % (fname, TP * 100, SL * 100, COST * 100))
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'open': 'float32', 'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs' % (len(d), time.time() - t0))

    for (sc, dp) in CASES:
        t1 = time.time()
        recs = run_case(d, sc, dp)
        if not recs:
            p('%dm 跌 %.0f%%: 无信号' % (sc, dp * 100)); continue
        arr = np.array([x[2] for x in recs], np.float64)
        p()
        p('=== %dm 跌 %.0f%%  信号 %d 笔  单笔均值 %.4f%%  胜率 %.1f%%'
          % (sc, dp * 100, len(recs), arr.mean() * 100, (arr > 0).mean() * 100))
        p('    %-6s %8s %12s %10s' % ('池N', '成交笔', '期末倍数', '最大回撤'))
        p('    ' + '-' * 42)
        for N in POOLS:
            eq, mdd, taken = pool_sim(recs, N)
            p('    %-6d %8d %11.4fx %9.2f%%' % (N, taken, eq, mdd * 100))

        # 时间切分（池 20）
        ts = [x[0] for x in recs]
        tmid = int(np.median(ts))
        a = [x for x in recs if x[0] <= tmid]
        b = [x for x in recs if x[0] > tmid]
        aa = np.array([x[2] for x in a], np.float64)
        bb = np.array([x[2] for x in b], np.float64)
        ea, _, _ = pool_sim(a, 20) if len(a) > 30 else (0, 0, 0)
        eb, _, _ = pool_sim(b, 20) if len(b) > 30 else (0, 0, 0)
        p('    时间切分(池20)：前段 %d笔 %.4f%% 期末%.4fx  |  后段 %d笔 %.4f%% 期末%.4fx'
          % (len(a), aa.mean() * 100, ea, len(b), bb.mean() * 100, eb))
        p('    %.0fs' % (time.time() - t1))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径，止盈假设能成交（成交率未计）。')
    p('2. 成本 0.25%% 往返已扣。')
    p('3. TP/SL 固定 3%%/1%%，不是每格最优。')
    p('4. 池模拟用 t_exit 结算，未模拟保证金/费率/最小下单量。')
    p('5. 142 天单一样本，含样本内。')
    p('6. 已下架归零币不在池内（幸存者偏差）。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
