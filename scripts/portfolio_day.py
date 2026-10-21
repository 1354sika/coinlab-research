# -*- coding: utf-8 -*-
"""portfolio_day.py — 把逐笔结果换成「单账号、日频、等权分散」的组合口径。

为什么需要：逐笔均值 +22% 不等于账户每天 +22%。
  同一天可能有 40 个信号同时入场并各自持有 24h，单账号资金只能分 40 份，
  组合日收益 = 当天被选中各笔的等权平均，而不是「每笔都满仓」。

用法: python -X utf8 portfolio_day.py klines_1m_wide.csv 0.50 0.035 [同时持仓上限]
"""
import sys
import numpy as np
import grid3

COST2 = 0.0008


def main():
    fname = sys.argv[1]
    tp = float(sys.argv[2])
    sl = float(sys.argv[3])
    maxpos = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    tf = 300 if '5m' in fname else (60 if '1m' in fname else (86400 if '1d' in fname else 3600))

    arr = grid3.collect(fname, tf)
    hh, ll, cc, ts = arr
    N = len(ts)
    jt = grid3.first_hit(hh, tp, True)
    js = grid3.first_hit(ll, -sl, False)
    last = cc[:, -1]
    none = (jt == 127) & (js == 127)
    net = np.where(none, last - COST2,
                   np.where(jt < js, tp - COST2, -sl - COST2)).astype(np.float64)

    o = np.argsort(ts)
    ts, net = ts[o], net[o]
    day = ts // 86400000
    uniq, inv = np.unique(day, return_inverse=True)

    # 每个交易日：当天所有信号；若超过 maxpos，按出现顺序取前 maxpos（保守：不挑赢家）
    dayret, cnt_all, cnt_used = [], [], []
    for d in range(len(uniq)):
        idx = np.flatnonzero(inv == d)
        cnt_all.append(len(idx))
        use = idx[:maxpos]
        cnt_used.append(len(use))
        dayret.append(net[use].mean())
    dayret = np.array(dayret)
    cnt_all = np.array(cnt_all)

    print('=' * 96)
    print('%s  TP=%.1f%%  SL=%.1f%%  持有 %d 根  同时持仓上限 %d'
          % (fname, tp * 100, sl * 100, grid3.hold_bars(tf), maxpos))
    print('=' * 96)
    print('交易日 %d   信号 %d   平均每天信号 %.1f（最多 %d）'
          % (len(uniq), N, cnt_all.mean(), cnt_all.max()))
    print('实际每天用仓 %d 个（超出上限的被丢弃）' % int(np.mean(cnt_used)))
    print('')

    eq = np.cumprod(1.0 + dayret)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    print('逐笔均值           %+.4f%%  胜率 %.1f%%' % (net.mean() * 100, (net > 0).mean() * 100))
    print('组合日收益 均值     %+.4f%%  中位 %+.4f%%' % (dayret.mean() * 100, np.median(dayret) * 100))
    print('日收益分位   p5 %+.2f%%  p25 %+.2f%%  p75 %+.2f%%  p95 %+.2f%%'
          % tuple(np.percentile(dayret, [5, 25, 75, 95]) * 100))
    print('正收益日占比        %.1f%%' % ((dayret > 0).mean() * 100))
    print('组合终值            %.4e  (起 1.0)' % eq[-1])
    print('最大回撤            %.1f%%' % (dd.min() * 100))
    print('')
    print('--- 若只允许 1 个仓位（严格单账号一次一笔）---')
    one = np.array([net[np.flatnonzero(inv == d)[0]] for d in range(len(uniq))])
    eq1 = np.cumprod(1.0 + one)
    dd1 = eq1 / np.maximum.accumulate(eq1) - 1.0
    print('日收益 均值 %+.4f%%  胜率 %.1f%%  终值 %.4e  最大回撤 %.1f%%'
          % (one.mean() * 100, (one > 0).mean() * 100, eq1[-1], dd1.min() * 100))
    print('')
    print('读法：组合日收益 = 当天被选中各笔的等权平均。'
          '若它只有逐笔均值的 1/20 左右，说明收益靠「每天 40 笔并发」撑着，单账号吃不到。')
    print('诚实边界：零滑点；TP 按 high、SL 按 low 触价；每天按 UTC 日切；信号重叠未去重；'
          '未模拟逐仓保证金/资金费率/手续费分档。')


if __name__ == '__main__':
    main()
