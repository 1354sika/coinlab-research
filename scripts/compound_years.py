# -*- coding: utf-8 -*-
"""compound_years.py — 复利到底滚成什么样：1/2/3/5 年，三种仓位池。

口径：单笔只占 1/pool 资金，赚了下一笔仓位同比例变大（这就是复利），
      收益从历史笔分布里有放回重采样。

用法: python -X utf8 compound_years.py klines_1m_wide.csv 0.50 0.035 0.0015
"""
import sys
import numpy as np
import grid3

HIT = 127


def first_idx(m):
    idx = m.argmax(axis=1).astype(np.int32)
    idx[~m.any(axis=1)] = HIT
    return idx


def collect_returns(fname, tp, sl, pool, cost):
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
    release = np.full(pool, -1, dtype=np.int64)
    rs = []
    for i in order:
        t = int(ts[i])
        k = int(np.argmin(release))
        if release[k] > t:
            continue
        rs.append(gross[i] - cost)
        release[k] = t + (int(exit_bar[i]) + 1) * tf * 1000
    rs = np.array(rs)
    span_days = (float(ts[order[-1]]) - float(ts[order[0]])) / 86400000.0
    return rs, len(rs) / span_days, span_days


def main():
    fname = sys.argv[1]
    tp = float(sys.argv[2])
    sl = float(sys.argv[3])
    cost = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015

    pool0 = 20
    rs, per_day, span_days = collect_returns(fname, tp, sl, pool0, cost)
    print('=' * 100)
    print('%s  TP=%.1f%%  SL=%.1f%%  成本 %.2f%%' % (fname, tp * 100, sl * 100, cost * 100))
    print('=' * 100)
    print('历史 %d 笔 / %.0f 天 = %.2f 笔/天；单笔期望 %+.4f%%' % (len(rs), span_days, per_day, rs.mean() * 100))
    print('')

    rng = np.random.default_rng(20261020)
    sims = 2000
    chunk = 200
    hdr = '%-8s %-6s %10s %12s %12s %12s %12s'
    print(hdr % ('仓位池', '年数', '模拟笔数', 'P5(最差5%)', '中位', 'P95', '跌到30%概率'))
    print('-' * 100)

    for pool in (5, 10, 20):
        for years in (1, 2, 3, 5):
            n = int(round(per_day * 365 * years))
            finals = np.empty(sims)
            ruined = 0
            done = 0
            while done < sims:
                m = min(chunk, sims - done)
                draw = rng.choice(rs, size=(m, n), replace=True)
                lg = np.log1p(draw / pool).sum(axis=1)
                eq = np.exp(lg)
                finals[done:done + m] = eq
                ruined += int((eq < 0.3).sum())
                done += m
            q = np.percentile(finals, [5, 50, 95])
            print(hdr % ('%d' % pool, '%d' % years, '%d' % n,
                         fmt(q[0]), fmt(q[1]), fmt(q[2]), '%.2f%%' % (ruined / sims * 100)))
        print('')
    print('读法：中位 = 一半情形超过它；P5 = 最差 5% 的情形。仓位池越大，单笔影响越小、复利越慢但回撤越浅。')
    print('诚实边界：笔级重采样打散了「同一波行情连亏」的时间相关性，真实回撤更深、多年终值分布更宽；')
    print('          历史仅 %.0f 天，3/5 年是纯外推，只说明「若分布不变」会怎样，不代表会发生。' % span_days)
    print('          零滑点（止损触价即成交）；未计资金费率/限频/最小下单量。')


def fmt(v):
    if v >= 1e4:
        return '%.2e' % v
    return '%.2f' % v


if __name__ == '__main__':
    main()
