# -*- coding: utf-8 -*-
"""cost_sweep.py — 在「延迟一根」这个最保守的成交假设上，再叠滑点与手续费。

为什么这么做：之前所有数字都用 0.08% 往返成本 + 零滑点。
但 A 模式(buy at high)的收益本身依赖「同一秒成交」，滑点恰恰是杀这一点的。
所以这里在 C 模式(延迟一根)上扫一组「往返总成本」，看还剩多少。

用法: python -X utf8 cost_sweep.py klines_1m_wide.csv 0.50 0.035
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

    # C 模式毛收益（延迟一根成交，不含成本）
    grossC = np.empty(N, dtype=np.float64)
    for i in range(N):
        a, b = int(jt[i]), int(js[i])
        if a == HIT and b == HIT:
            grossC[i] = cc[i, -1]
            continue
        first = a if a < b else b
        nxt = first + 1
        grossC[i] = cc[i, nxt] if nxt < H else cc[i, first]

    # A 模式毛收益（high 触价，不含成本）
    grossA = np.where((jt == HIT) & (js == HIT), cc[:, -1],
                      np.where(jt < js, tp, -sl)).astype(np.float64)

    print('=' * 100)
    print('%s  TP=%.1f%%  SL=%.1f%%  持有 %d 根(=%.1f 小时)  信号 %d'
          % (fname, tp * 100, sl * 100, H, H * tf / 3600.0, N))
    print('=' * 100)
    print('毛均值：A(high触价) %+.4f%%   C(延迟一根) %+.4f%%' % (grossA.mean() * 100, grossC.mean() * 100))
    print('')

    hdr = '%-16s %10s %10s %8s %10s %14s'
    print(hdr % ('往返总成本', 'A 均值/笔', 'C 均值/笔', 'C 胜率', 'C 中位/笔', 'C 顺序复利'))
    print('-' * 100)
    for c in (0.0000, 0.0004, 0.0008, 0.0016, 0.0030, 0.0050, 0.0100, 0.0200):
        a = grossA - c
        b = grossC - c
        byd = np.clip(b, -0.999, None)
        fin = float(np.prod(1.0 + byd))
        tag = '%.2f%%' % (c * 100)
        if abs(c - 0.0008) < 1e-9:
            tag += '(旧口径)'
        print(hdr % (tag, '%+.4f%%' % (a.mean() * 100), '%+.4f%%' % (b.mean() * 100),
                     '%.1f%%' % ((b > 0).mean() * 100), '%+.4f%%' % (np.median(b) * 100),
                     ('%.3e' % fin) if abs(fin) > 1e6 else '%.3f' % fin))
    print('')
    print('现实参考：币安现货 maker 0.02%/侧(往返 0.04%) + 滑点≈0（限价单不滑，但可能不成交）；')
    print('          taker 0.075%/侧(往返 0.15%) + 滑点 0.05~0.20%/侧(往返 0.1~0.4%)。')
    print('          即：能挂成 maker 大约落在 0.04%~0.08%；被迫追价落在 0.25%~0.55%。')
    print('诚实边界：C 模式已用「下一根收盘」近似延迟；止损侧仍假设触价即成交（真实市价止损还有滑点）；')
    print('          信号重叠未去重；成交率未计。')


if __name__ == '__main__':
    main()
