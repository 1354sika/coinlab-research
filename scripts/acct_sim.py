# -*- coding: utf-8 -*-
"""acct_sim.py — 单账号、本金 100、真实成交假设（延迟一根 + 成本）下的账户曲线。

口径：
  · 收益用 C 模式（触价后按下一根收盘成交），最保守
  · 每天按时间顺序最多开 MAXPOS 个仓位，每仓固定占总资金 1/MAXPOS
  · 当天开不满就留现金；货币单位无关，起点 1.0 = 100 元
  · 持有 24h，日频轮动

用法: python -X utf8 acct_sim.py klines_1m_wide.csv 0.50 0.035
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
    tf = 300 if '5m' in fname else (60 if '1m' in fname else (86400 if '1d' in fname else 3600))

    hh, ll, cc, ts = grid3.collect(fname, tf)
    N, H = hh.shape
    jt = first_idx(hh >= tp)
    js = first_idx(ll <= -sl)

    gross = np.empty(N, dtype=np.float64)
    for i in range(N):
        a, b = int(jt[i]), int(js[i])
        if a == HIT and b == HIT:
            gross[i] = cc[i, -1]
            continue
        first = a if a < b else b
        nxt = first + 1
        gross[i] = cc[i, nxt] if nxt < H else cc[i, first]

    day = ts // 86400000
    uniq, inv = np.unique(day, return_inverse=True)
    nday = len(uniq)
    years = nday / 365.0

    print('=' * 104)
    print('%s  TP=%.1f%%  SL=%.1f%%  持有 %d 根(=%.1f 小时)'
          % (fname, tp * 100, sl * 100, H, H * tf / 3600.0))
    print('信号总数 %d  交易日 %d (=%.2f 年)  平均每天信号 %.1f'
          % (N, nday, years, N / nday))
    print('=' * 104)
    print('全部信号年化笔数 ≈ %.0f 笔/年（这是「信号密度」，不是单账号能做的量）' % (N / years))
    print('')

    hdr = '%-8s %-10s %10s %12s %10s %11s %10s'
    print(hdr % ('持仓上限', '成本/往返', '年执行笔数', '本金100→', '年化', '最大回撤', '日均持仓'))
    print('-' * 104)

    for maxpos in (1, 3, 5, 10, 20, 50):
        for cost in (0.0004, 0.0015, 0.0030):
            eq = 1.0
            peak = 1.0
            mdd = 0.0
            ntr = 0
            used = 0
            for d in range(nday):
                idx = np.flatnonzero(inv == d)
                idx = idx[:maxpos]
                if len(idx) == 0:
                    continue
                r = gross[idx] - cost
                dayret = r.sum() / maxpos
                eq *= (1.0 + dayret)
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1.0)
                ntr += len(idx)
                used += len(idx)
            ann = (eq ** (1.0 / years) - 1.0) if eq > 0 else -1.0
            tag = '%.2f%%' % (cost * 100)
            if abs(cost - 0.0008) < 1e-9:
                tag += '*'
            print(hdr % ('%d' % maxpos, tag, '%.0f' % (ntr / years),
                         ('%.3e' % (eq * 100)) if eq * 100 > 1e6 else '%.2f' % (eq * 100),
                         ('%.1f%%' % (ann * 100)) if abs(ann) < 1e4 else ('%.2e' % ann),
                         '%.1f%%' % (mdd * 100), '%.1f' % (used / nday)))
    print('')
    print('读法：年执行笔数 ≈ 信号年化笔数 × (持仓上限 / 每天信号数) 封顶。')
    print('      「本金100→」是该上限下的真实账户终值（100 元起，每仓固定 1/上限 资金）。')
    print('诚实边界：C 模式（下一根收盘成交）已含反应延迟；止损按触价即成交，真实市价止损仍有滑点；')
    print('          日频轮动把「持有 24h」近似为「当天开、次日结」，真实资金占用更紧；成交率未计。')


if __name__ == '__main__':
    main()
