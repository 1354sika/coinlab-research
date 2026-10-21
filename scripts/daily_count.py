# -*- coding: utf-8 -*-
"""daily_count.py — 仓位池口径下，每天实际成交几笔（分布，不只是均值）。

用法: python -X utf8 daily_count.py klines_1m_wide.csv 0.50 0.035 20 0.0015
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
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    cost = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0015
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
    per_day = {}
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for i in order:
        t = int(ts[i])
        k = int(np.argmin(release))
        if release[k] > t:
            continue
        r = gross[i] - cost
        eq *= (1.0 + r / pool)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        release[k] = t + (int(exit_bar[i]) + 1) * tf * 1000
        d = t // 86400000
        per_day[d] = per_day.get(d, 0) + 1

    all_days = int((int(ts[order[-1]]) - int(ts[order[0]])) // 86400000) + 1
    cnt = np.array(list(per_day.values()))
    zero_days = all_days - len(per_day)
    years = all_days / 365.0
    span_years = (float(ts[order[-1]]) - float(ts[order[0]])) / (365.0 * 86400000.0)
    ann = (eq ** (1 / span_years) - 1) if eq > 0 else -1

    print('=' * 88)
    print('%s  TP=%.1f%%  SL=%.1f%%  仓位池 N=%d  成本 %.2f%%'
          % (fname, tp * 100, sl * 100, pool, cost * 100))
    print('=' * 88)
    print('自然日 %d 天（%.2f 年），其中有成交的日子 %d 天，零成交日 %d 天'
          % (all_days, years, len(per_day), zero_days))
    print('')
    print('每天成交笔数：')
    print('  均值        %.2f 笔/天' % (cnt.sum() / all_days))
    print('  有成交日均值 %.2f 笔/天' % (cnt.sum() / len(per_day)))
    print('  中位        %.1f 笔/天' % np.median(cnt))
    print('  p10 %.0f   p90 %.0f   最多 %d' % (np.percentile(cnt, 10), np.percentile(cnt, 90), cnt.max()))
    print('  分布：1笔内 %d 天 | 2-5笔 %d 天 | 6-15笔 %d 天 | 16-30笔 %d 天 | >30笔 %d 天'
          % ((cnt == 1).sum(), ((cnt >= 2) & (cnt <= 5)).sum(), ((cnt >= 6) & (cnt <= 15)).sum(),
             ((cnt >= 16) & (cnt <= 30)).sum(), (cnt > 30).sum()))
    print('')
    print('结果：本金 100 → %.2f 元   年化 %.1f%%   最大回撤 %.1f%%'
          % (eq * 100, ann * 100, mdd * 100))
    print('')
    print('诚实边界：C 模式（下一根收盘成交）已含反应延迟；止损按触价即成交，真实市价止损仍有滑点；')
    print('          未计最小下单量/资金费率/API 限频；样本含样本内最优参数。')


if __name__ == '__main__':
    main()
