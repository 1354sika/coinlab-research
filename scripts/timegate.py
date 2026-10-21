# -*- coding: utf-8 -*-
"""
timegate.py — 扫「时间尺度 × 跌幅门槛」，找"跌了会反弹"的真正门槛

上一轮结论：1m 跌 2% 无论怎么配 TP/SL 都是负的（36 组全负）。
本轮要回答：是不是"时间尺度选错了"——把 1m 换成 3m/5m/15m/30m/1h/4h，
跌幅门槛从 2% 扫到 20%，看哪一格能翻正。

输出：二维矩阵（行=时间尺度，列=跌幅门槛），每格 = 单笔均值（已扣 0.25%）。
伴随：笔数矩阵，防止小样本假象。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'timegate.txt')

COST  = 0.0025
TP    = 0.03       # 固定止盈 3%
SL    = 0.01       # 固定止损 1%
HOURS = 8          # 持仓上限 8 小时
VOLR  = 3.0

SCALES = [1, 3, 5, 15, 30, 60, 240]                       # 分钟
DROPS  = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]       # 跌幅门槛

L = []
def p(s=''):
    print(s)
    L.append(s)

def run_scale(d, scale):
    """把 1m 重采样成 scale 分钟，返回 [(t_in, r_net), ...]"""
    H = max(2, int(HOURS * 60 / scale))
    vol_win = 96
    recs = []
    rej = 0
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
        # 重采样：按 open_time // (scale*60000) 分桶
        bucket = t // (scale * 60000)
        # 边界
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
        # 量比
        vma = pd.Series(bv).rolling(vol_win, min_periods=vol_win // 2).mean().to_numpy()
        volr = np.where(vma > 0, bv / vma, 1.0)
        # 单根跌幅
        r = np.zeros(nb, np.float64)
        r[1:] = bc[1:] / bc[:-1] - 1.0
        for drop in DROPS:
            sig = np.flatnonzero((r <= -drop) & (volr >= VOLR))
            if len(sig) == 0:
                continue
            keep, last = [], -10**9
            STEP = max(1, scale)   # 每 scale 根取一个，即每 scale 分钟一次
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
                up = entry * (1 + TP)
                dn = entry * (1 - SL)
                ht = seg_h >= up
                hl = seg_l <= dn
                it = int(np.argmax(ht)) if ht.any() else 10**9
                il = int(np.argmax(hl)) if hl.any() else 10**9
                if it < 10**9 and it <= il:
                    rr = TP
                elif il < 10**9:
                    rr = -SL
                else:
                    rr = seg_c[-1] / entry - 1.0
                recs.append((int(bt[i + 1]), scale, drop, float(rr - COST)))
    return recs

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    p('=' * 112)
    p('时间尺度 × 跌幅门槛   file=%s   TP=%.0f%%/SL=%.0f%%   COST=%.2f%%   持仓<=%dh'
      % (fname, TP * 100, SL * 100, COST * 100, HOURS))
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'open': 'float32', 'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs  币 %d' % (len(d), time.time() - t0, d.symbol.nunique()))

    allrecs = []
    for sc in SCALES:
        t1 = time.time()
        rr = run_scale(d, sc)
        p('  尺度 %4dm  %7d 笔  %.0fs' % (sc, len(rr), time.time() - t1))
        allrecs.extend(rr)
    p('总 %d 笔  %.0fs' % (len(allrecs), time.time() - t0))

    R = pd.DataFrame(allrecs, columns=['t', 'scale', 'drop', 'r'])
    p()
    p('=== 一、单笔均值矩阵（已扣 %.2f%%，行=时间尺度，列=跌幅门槛）' % (COST * 100))
    hdr = '  %-8s' % '尺度' + ''.join('%9.0f%%' % (x * 100) for x in DROPS)
    p(hdr)
    best = (None, -9)
    for sc in SCALES:
        row = '  %-8d' % sc
        for dp in DROPS:
            sub = R[(R.scale == sc) & (R['drop'] >= dp - 1e-12) & (R['drop'] <= dp + 1e-12)]
            v = sub.r.mean() * 100 if len(sub) >= 20 else float('nan')
            row += '%9.3f' % v if np.isfinite(v) else '%9s' % '-'
            if np.isfinite(v) and v > best[1]:
                best = ((sc, dp), v)
        p(row)
    if best[0]:
        p('  => 最优格：%dm 跌 %.0f%%  →  %.4f%%/笔' % (best[0][0], best[0][1] * 100, best[1]))
    else:
        p('  => 没有任何一格 >=20 笔且均值为正')

    p()
    p('=== 二、笔数矩阵（防止小样本假象）')
    p('  %-8s' % '尺度' + ''.join('%9.0f%%' % (x * 100) for x in DROPS))
    for sc in SCALES:
        row = '  %-8d' % sc
        for dp in DROPS:
            sub = R[(R.scale == sc) & (np.abs(R['drop'] - dp) < 1e-12)]
            row += '%9d' % len(sub)
        p(row)

    p()
    p('=== 三、正期望格排名（>=100 笔才计入）')
    p('  %-8s %-10s %10s %10s %10s' % ('尺度', '跌幅', '笔数', '均值/笔', '中位/笔'))
    p('  ' + '-' * 52)
    rows = []
    for sc in SCALES:
        for dp in DROPS:
            sub = R[(R.scale == sc) & (np.abs(R['drop'] - dp) < 1e-12)]
            if len(sub) >= 100:
                rows.append((sc, dp, len(sub), sub.r.mean(), sub.r.median()))
    rows.sort(key=lambda x: -x[3])
    for (sc, dp, n, mu, md) in rows[:12]:
        p('  %-8d %-10.0f%% %10d %9.4f%% %9.4f%%' % (sc, dp * 100, n, mu * 100, md * 100))
    n_pos = sum(1 for x in rows if x[3] > 0)
    p('  => 正期望格 %d / %d' % (n_pos, len(rows)))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径，止盈假设能成交（成交率未计）。')
    p('2. TP/SL 固定 3%%/1%%，不是每格最优（每格最优会过拟合）。')
    p('3. 重采样用 open_time 分桶，桶内首开/末收/极值/量和。')
    p('4. 入场点每 scale 根取一个，尺度越大重叠越少。')
    p('5. 已下架归零币不在池内（幸存者偏差）。')
    p('6. 单一样本（142 天），含样本内。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
