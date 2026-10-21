# -*- coding: utf-8 -*-
"""strict_serial.py — 严格单账号：一笔资金、一次只持一笔、平了才开下一笔。

这是「单账号 + 机器 24h 盯盘」的真实约束：
  · 资金只有一份 → 不能同时持 40 个币
  · 机器不停机 → 上一笔一平，立刻找下一个信号，不等收盘、不等明天
  · 出场用 C 模式（延迟一根成交，已含反应时间），并扣往返成本

用法: python -X utf8 strict_serial.py klines_1m_wide.csv 0.50 0.035 [成本]
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
        tf = 300      # collect 内部把 1m 聚合成 5m 桶

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
    span_ms = float(ts[order[-1]] - ts[order[0]])
    years = span_ms / (365.0 * 86400000.0)

    busy_until = -1
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    picked = []
    skip = 0
    for i in order:
        t = int(ts[i])
        if t < busy_until:
            skip += 1
            continue
        r = gross[i] - cost
        eq *= (1.0 + r)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        busy_until = t + (int(exit_bar[i]) + 1) * tf * 1000
        picked.append(i)

    ntr = len(picked)
    hold_bars = float(np.mean([exit_bar[i] for i in picked])) if picked else 0.0
    print('=' * 96)
    print('%s  TP=%.1f%%  SL=%.1f%%  往返成本 %.2f%%  持有上限 %d 根(=%.1f 小时)'
          % (fname, tp * 100, sl * 100, cost * 100, H, H * tf / 3600.0))
    print('=' * 96)
    print('信号总数 %d   区间 %.2f 年   平均每天信号 %.1f' % (N, years, N / (years * 365)))
    print('')
    print('严格单账号（一次只持一笔）:')
    print('  实际成交         %d 笔  (= %.0f 笔/年)' % (ntr, ntr / years))
    print('  因"手上有仓"跳过  %d 笔  (占信号 %.1f%%)' % (skip, skip / max(N, 1) * 100))
    print('  平均持仓         %.0f 根 (= %.1f 小时)' % (hold_bars, (hold_bars + 1) * tf / 3600.0))
    print('  单笔均值         %+.4f%%' % ((gross[picked].mean() - cost) * 100))
    print('  胜率             %.1f%%' % ((gross[picked] - cost > 0).mean() * 100))
    print('')
    print('  本金 100 →       %s' % (('%.3e' % (eq * 100)) if eq * 100 > 1e6 else '%.2f 元' % (eq * 100)))
    print('  年化             %s' % (('%.1f%%' % ((eq ** (1 / years) - 1) * 100))
                                    if abs(eq ** (1 / years) - 1) < 1e4 else
                                    ('%.2e' % (eq ** (1 / years) - 1))))
    print('  最大回撤         %.1f%%' % (mdd * 100))
    print('')
    print('诚实边界：C 模式（下一根收盘成交）已含反应延迟；止损按触价即成交，真实市价止损仍有滑点；')
    print('          资金占用按「入场到出场全覆盖」保守估计，实盘可更早挂单但会漏信号；')
    print('          未计交易所最小下单量、资金费率、API 限频；样本含样本内最优参数。')


if __name__ == '__main__':
    main()
