# -*- coding: utf-8 -*-
"""pool_sim.py — 单账号、多币并发持仓、下单动作串行。

用户的真实口径：
  · 资金可以同时挂在多个币上（各占 1/N）
  · 但「买/卖」这个动作一次只能执行一个（这里按时间戳先后串行，忽略毫秒冲突）
  · 每个币各自独立持有到 TP / SL / 24h，互不影响
  · 出场用 C 模式（延迟一根成交），扣往返成本

用法: python -X utf8 pool_sim.py klines_1m_wide.csv 0.50 0.035
"""
import sys
import numpy as np
import grid3

HIT = 127


def first_idx(m):
    idx = m.argmax(axis=1).astype(np.int32)
    idx[~m.any(axis=1)] = HIT
    return idx


def main():
    fname = sys.argv[1]
    tp = float(sys.argv[2])
    sl = float(sys.argv[3])
    cost = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
    tf = 300 if '5m' in fname else (60 if '1m' in fname else (86400 if '1d' in fname else 3600))
    if tf < 300:
        tf = 300

    hh, ll, cc, ts = grid3.collect(fname, tf)
    N, H = hh.shape
    jt = first_idx(hh >= tp)
    js = first_idx(ll <= -sl)

    gross = np.empty(N, dtype=np.float64)
    exit_bar = np.empty(N, dtype=np.int32)
    for i in range(N):
        a, b = int(jt[i]), int(js[i])
        if a == HIT and b == HIT:
            gross[i] = cc[i, -1]
            exit_bar[i] = H - 1
            continue
        first = a if a < b else b
        nxt = first + 1
        if nxt < H:
            gross[i] = cc[i, nxt]
            exit_bar[i] = nxt
        else:
            gross[i] = cc[i, first]
            exit_bar[i] = first

    order = np.argsort(ts)
    span_years = float(ts[order[-1]] - ts[order[0]]) / (365.0 * 86400000.0)

    print('=' * 104)
    print('%s  TP=%.1f%%  SL=%.1f%%  往返成本 %.2f%%  区间 %.2f 年  信号 %d'
          % (fname, tp * 100, sl * 100, cost * 100, span_years, N))
    print('=' * 104)
    print('口径：多币并发持仓（资金分 N 份），每笔独立持有到 TP/SL/24h，下单动作串行')
    print('')

    hdr = '%-8s %10s %10s %10s %14s %10s %10s'
    print(hdr % ('仓位池 N', '实际成交', '年笔数', '平均占用', '本金100→', '年化', '最大回撤'))
    print('-' * 104)

    for pool in (1, 5, 10, 20, 50, 100):
        release = np.full(pool, -1, dtype=np.int64)
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        ntr = 0
        for i in order:
            t = int(ts[i])
            k = int(np.argmin(release))
            if release[k] > t:
                continue
            r = gross[i] - cost
            eq *= (1.0 + r / pool)
            peak = max(peak, eq)
            if peak > 0:
                mdd = min(mdd, eq / peak - 1.0)
            release[k] = t + (int(exit_bar[i]) + 1) * tf * 1000
            ntr += 1
        occ = float(pool - np.mean(release > ts[order[0]])) if pool > 1 else 1.0
        ann = (eq ** (1.0 / span_years) - 1.0) if eq > 0 else -1.0
        print(hdr % ('%d' % pool, '%d 笔' % ntr, '%.0f' % (ntr / span_years),
                     '%.1f' % occ,
                     ('%.3e' % (eq * 100)) if eq * 100 > 1e6 else '%.2f 元' % (eq * 100),
                     ('%.1f%%' % (ann * 100)) if abs(ann) < 1e4 else ('%.2e' % ann),
                     '%.1f%%' % (mdd * 100)))
    print('')
    print('读法：N 越大，能接住的信号越多，但每笔只占 1/N 资金；')
    print('      「平均占用」是池子里平均有几个仓位在持仓（越接近 N 说明资金越满）。')
    print('诚实边界：C 模式（下一根收盘成交）已含反应延迟；下单动作冲突按时间戳先后处理，未模拟 API 限频排队；')
    print('          止损按触价即成交，真实市价止损仍有滑点；未计最小下单量/资金费率；样本含样本内最优参数。')


if __name__ == '__main__':
    main()
