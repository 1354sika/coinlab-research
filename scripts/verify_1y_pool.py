# -*- coding: utf-8 -*-
"""
verify_1y_pool.py — 目标是「每天 +0.1%」，看哪个池能到

换算：
  每天 +0.1%  =>  年化 1.001^365 = 1.4405x  =>  +44.05%
  每天 +0.05% =>  年化 1.0005^365 = 1.2001x =>  +20.01%

已知：3m 跌 3%（TP3%/SL1%，1 年，4/4 段正）
  池 10 = 1.2370x/年 => 每天 +0.0582%
  池 20 = 1.1271x/年 => 每天 +0.0327%

本脚本补测池 1/2/3/5，看更集中的仓位能不能到每天 0.1%。
同时输出 4 段切分（每段年化、是否全正）。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'verify_1y_pool.txt')

COST  = 0.0025
TP    = 0.03
SL    = 0.01
HOURS = 8
VOLR  = 3.0
SCALE = 3
DROP  = 0.03
POOLS = [1, 2, 3, 5, 10, 20, 50]
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
    p('目标：每天 +0.1%%（年化 1.4405x）   file=%s' % fname)
    p('策略：3m 跌 3%% + TP3%%/SL1%% + 成本 0.25%%   池 N 扫描')
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'open': 'float32', 'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs  币 %d' % (len(d), time.time() - t0, d.symbol.nunique()))
    tmin, tmax = int(d.open_time.min()), int(d.open_time.max())
    days = (tmax - tmin) / 86400000
    p('跨度 %s ~ %s  (%.1f 天)' % (pd.to_datetime(tmin, unit='ms'), pd.to_datetime(tmax, unit='ms'), days))

    recs = run_case(d, SCALE, DROP)
    arr = np.array([x[2] for x in recs], np.float64)
    p()
    p('信号 %d 笔   单笔均值 %.4f%%   胜率 %.1f%%' % (len(recs), arr.mean() * 100, (arr > 0).mean() * 100))
    p()

    # ---- 目标对照 ----
    p('=== 零、目标换算')
    p('  每天 +0.10%%  =>  年化 %.4fx  (+%.1f%%)' % (1.001 ** 365, (1.001 ** 365 - 1) * 100))
    p('  每天 +0.05%%  =>  年化 %.4fx  (+%.1f%%)' % (1.0005 ** 365, (1.0005 ** 365 - 1) * 100))
    p('  每天 +0.02%%  =>  年化 %.4fx  (+%.1f%%)' % (1.0002 ** 365, (1.0002 ** 365 - 1) * 100))
    p()

    # ---- 池扫描 ----
    p('=== 一、池 N 扫描（全样本 %.0f 天）' % days)
    p('  %-6s %8s %12s %12s %12s %10s %10s' %
      ('池N', '成交笔', '期末倍数', '年化', '日化', '最大回撤', '达标0.1%?'))
    p('  ' + '-' * 78)
    target_annual = 1.001 ** 365
    rows = []
    for N in POOLS:
        eq, mdd, taken = pool_sim(recs, N)
        ann = eq ** (365.0 / days) if eq > 0 else 0.0
        daily = ann ** (1.0 / 365) - 1.0 if ann > 0 else -1.0
        ok = 'YES' if daily >= 0.001 else 'no'
        p('  %-6d %8d %11.4fx %11.4fx %11.4f%% %9.2f%% %9s'
          % (N, taken, eq, ann, daily * 100, mdd * 100, ok))
        rows.append((N, eq, ann, daily, mdd))

    # ---- 4 段切分（每段年化）----
    p()
    p('=== 二、4 段切分（每段独立复利，看是否每段都有正收益）')
    R = pd.DataFrame(recs, columns=['t_in', 't_exit', 'r'])
    edges = np.linspace(tmin, tmax, NSEG + 1).astype(np.int64)
    p('  %-6s %-6s %-22s %8s %10s %12s %12s' %
      ('池N', '段', '区间', '笔数', '均值/笔', '段内倍数', '段年化'))
    p('  ' + '-' * 82)
    for N in [3, 5, 10, 20]:
        seg_pos = 0
        for k in range(NSEG):
            lo, hi = edges[k], edges[k + 1]
            sub = R[(R.t_in >= lo) & (R.t_in < hi)]
            if len(sub) < 20:
                p('  %-6d %-6d %-22s %8d %10s %12s %12s' % (N, k + 1, '样本不足', len(sub), '-', '-', '-'))
                continue
            eq, _, _ = pool_sim(list(sub.itertuples(index=False, name=None)), N)
            seg_days = (hi - lo) / 86400000
            ann = eq ** (365.0 / seg_days) if eq > 0 else 0.0
            if eq > 1:
                seg_pos += 1
            p('  %-6d %-6d %-22s %8d %9.4f%% %11.4fx %11.4fx'
              % (N, k + 1,
                 '%s~%s' % (pd.to_datetime(lo, unit='ms').strftime('%m-%d'),
                            pd.to_datetime(hi, unit='ms').strftime('%m-%d')),
                 len(sub), sub.r.mean() * 100, eq, ann))
        p('  %-6d %-6s %-22s %8s %10s %12s %11d/4 段为正' % (N, '', '', '', '', '', seg_pos))
        p()

    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径，止盈假设能成交（成交率未计）。')
    p('2. 成本 0.25%% 往返已扣。')
    p('3. TP/SL 固定 3%%/1%%，不是每格最优。')
    p('4. 池模拟用 t_exit 结算，未模拟保证金/费率/最小下单量。')
    p('5. 60 币样本，已下架归零币不在池内（幸存者偏差）。')
    p('6. 年化由 %.0f 天折算，池越小回撤越深、越容易在中途被打到接近 0。' % days)

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
