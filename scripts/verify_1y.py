# -*- coding: utf-8 -*-
"""
verify_1y.py — 用 1 年数据验证「跌 3%」是不是真规律

142 天（1m_wide）的结论：3m~60m 跌 3% 是正期望，但五组里四组时间切分前后段符号相反。
142 天只够切 2 段，切不出结论。

本脚本用 klines_1m_1y.csv（60 币、约 1 年、4.2GB）做 **4 段切分**：
  每段约 3 个月，看四段是否全部为正。
  · 四段全正 => 真规律
  · 有正有负 => 142 天那次是运气

同时输出池模拟（N=10/20/50），因为池约束才是真实约束。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'verify_1y.txt')

COST  = 0.0025
TP    = 0.03
SL    = 0.01
HOURS = 8
VOLR  = 3.0

CASES = [(3, 0.03), (5, 0.03), (15, 0.03), (30, 0.03), (60, 0.03), (15, 0.05), (60, 0.05)]
POOLS = [10, 20, 50]
NSEG  = 4

L = []
def p(s=''):
    print(s)
    L.append(s)

def run_case(d, scale, drop):
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
            recs.append((int(bt[i + 1]), int(bt[min(i + 1 + j, nb - 1)]), float(rr - COST)))
    return recs

def pool_sim(recs, N):
    eq, peak, mdd = 1.0, 1.0, 0.0
    pending = []; taken = 0
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
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_1y.csv'
    p('=' * 112)
    p('1 年验证   file=%s   TP=%.0f%%/SL=%.0f%%   COST=%.2f%%   分段=%d' % (fname, TP * 100, SL * 100, COST * 100, NSEG))
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'open': 'float32', 'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs  币 %d' % (len(d), time.time() - t0, d.symbol.nunique()))
    tmin, tmax = int(d.open_time.min()), int(d.open_time.max())
    p('跨度 %s ~ %s  (%.1f 天)' % (pd.to_datetime(tmin, unit='ms'), pd.to_datetime(tmax, unit='ms'),
                                  (tmax - tmin) / 86400000))

    for (sc, dp) in CASES:
        t1 = time.time()
        recs = run_case(d, sc, dp)
        if len(recs) < 50:
            p()
            p('=== %dm 跌 %.0f%%: 只有 %d 笔，跳过' % (sc, dp * 100, len(recs)))
            continue
        arr = np.array([x[2] for x in recs], np.float64)
        p()
        p('=== %dm 跌 %.0f%%   信号 %d 笔   单笔均值 %.4f%%   胜率 %.1f%%'
          % (sc, dp * 100, len(recs), arr.mean() * 100, (arr > 0).mean() * 100))

        # ---- 4 段切分 ----
        R = pd.DataFrame(recs, columns=['t_in', 't_exit', 'r'])
        edges = np.linspace(tmin, tmax, NSEG + 1).astype(np.int64)
        p('  %-6s %-22s %8s %10s %10s' % ('段', '区间', '笔数', '均值/笔', '期末(池20)'))
        p('  ' + '-' * 62)
        seg_pos = 0
        for k in range(NSEG):
            lo = edges[k]; hi = edges[k + 1]
            sub = R[(R.t_in >= lo) & (R.t_in < hi)]
            if len(sub) < 20:
                p('  %-6d %-22s %8d %10s %10s' % (k + 1, '样本不足', len(sub), '-', '-'))
                continue
            eq, _, _ = pool_sim(list(sub.itertuples(index=False, name=None)), 20)
            mu = sub.r.mean() * 100
            if mu > 0:
                seg_pos += 1
            p('  %-6d %-22s %8d %9.4f%% %9.4fx'
              % (k + 1, '%s~%s' % (pd.to_datetime(lo, unit='ms').strftime('%m-%d'),
                                   pd.to_datetime(hi, unit='ms').strftime('%m-%d')),
                 len(sub), mu, eq))
        p('  => %d / %d 段为正' % (seg_pos, NSEG))

        # ---- 池模拟（全样本）----
        line = '  池模拟：'
        for N in POOLS:
            eq, mdd, taken = pool_sim(recs, N)
            line += '  池%d %.4fx(回撤%.1f%%,成交%d笔)' % (N, eq, mdd * 100, taken)
        p(line)
        p('  %.0fs' % (time.time() - t1))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径，止盈假设能成交（成交率未计）。')
    p('2. 成本 0.25%% 往返已扣。')
    p('3. TP/SL 固定 3%%/1%%，不是每格最优。')
    p('4. 4 段切分按时间等分，每段约 3 个月。')
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
