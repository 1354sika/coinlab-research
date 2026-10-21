# -*- coding: utf-8 -*-
"""prob_profile.py — 用概率语言描述这个策略：赢面、赔率、连亏、破产概率。

用户口径：「我信概率」——不要求每笔都对，只要频率兑现。
所以这里不给单一点估计，给分布：
  · 单笔赢面 / 平均赢 / 平均输 / 赔率
  · 历史最长连亏
  · 蒙特卡洛：从历史笔收益有放回重采样，模拟一年，看终值与最大回撤的分布

用法: python -X utf8 prob_profile.py klines_1m_wide.csv 0.50 0.035 20 0.0015
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
    rs = []        # 每笔净收益（毛 - 成本），按成交时间顺序
    for i in order:
        t = int(ts[i])
        k = int(np.argmin(release))
        if release[k] > t:
            continue
        rs.append(gross[i] - cost)
        release[k] = t + (int(exit_bar[i]) + 1) * tf * 1000
    rs = np.array(rs)
    n = len(rs)
    span_days = (float(ts[order[-1]]) - float(ts[order[0]])) / 86400000.0
    per_day = n / span_days

    print('=' * 96)
    print('%s  TP=%.1f%%  SL=%.1f%%  池 %d  成本 %.2f%%'
          % (fname, tp * 100, sl * 100, pool, cost * 100))
    print('=' * 96)
    print('历史成交 %d 笔  /  %.0f 天  =  %.2f 笔/天' % (n, span_days, per_day))
    print('')

    win = rs > 0
    print('--- 单笔分布 ---')
    print('赢面        %.1f%%   （输面 %.1f%%）' % (win.mean() * 100, (1 - win.mean()) * 100))
    print('平均赢      %+.2f%%' % (rs[win].mean() * 100))
    print('平均输      %+.2f%%' % (rs[~win].mean() * 100))
    r_win = rs[win].mean()
    r_lose = abs(rs[~win].mean())
    print('赔率        %.2f : 1   （平均赢 / 平均输）' % (r_win / r_lose))
    print('单笔期望    %+.4f%%' % (rs.mean() * 100))
    print('盈亏平衡所需赢面  %.1f%%   （实际 %.1f%%，余量 %+.1f 个点）'
          % (r_lose / (r_win + r_lose) * 100, win.mean() * 100,
             win.mean() * 100 - r_lose / (r_win + r_lose) * 100))
    print('')

    # 历史最长连亏
    mx = cur = 0
    for v in rs:
        if v < 0:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    print('历史最长连亏  %d 笔  （按 %d 仓位分，等于账户连跌 %.1f%%）'
          % (mx, pool, (1 - (1 - r_lose / pool) ** mx) * 100 if mx else 0))
    print('')

    # 蒙特卡洛：一年（per_day * 365 笔）
    n_year = int(round(per_day * 365))
    sims = 2000
    rng = np.random.default_rng(20261020)
    finals = np.empty(sims)
    mdds = np.empty(sims)
    ruined = 0
    for s in range(sims):
        draw = rng.choice(rs, size=n_year, replace=True)
        step = 1.0 + draw / pool
        eq = np.cumprod(step)
        finals[s] = eq[-1]
        peak = np.maximum.accumulate(eq)
        mdds[s] = (eq / peak - 1.0).min()
        if eq[-1] < 0.3:
            ruined += 1
    print('--- 蒙特卡洛：一年 %d 笔（笔级重采样 %d 次，池 %d）---' % (n_year, sims, pool))
    print(hdr_line('终值（本金 1.0）', finals))
    print(hdr_line('最大回撤', mdds * 100, pct=False))
    print('一年内跌到本金 30%% 以下的概率  %.2f%%' % (ruined / sims * 100))
    print('')
    print('诚实边界：笔级重采样破坏了「同一波行情连亏」的时间相关性，真实回撤会比这里更深；')
    print('          历史只有 %.0f 天，重采样只是在已有分布内打乱，不会告诉你未来有没有更差的笔。' % span_days)
    print('          零滑点（止损按触价成交）；未计最小下单量/资金费率/限频；样本含样本内最优参数。')


def hdr_line(name, x, pct=True):
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    if pct:
        return ('%-14s P5 %.3f | P25 %.3f | 中位 %.3f | P75 %.3f | P95 %.3f'
                % (name, qs[0], qs[1], qs[2], qs[3], qs[4]))
    return ('%-14s P5 %.1f%% | P25 %.1f%% | 中位 %.1f%% | P75 %.1f%% | P95 %.1f%%'
            % (name, qs[0], qs[1], qs[2], qs[3], qs[4]))


if __name__ == '__main__':
    main()
