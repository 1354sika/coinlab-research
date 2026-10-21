# -*- coding: utf-8 -*-
"""seg_verify.py — 把某个 (TP, SL) 固定下来，按时间切段看它是不是只靠某一段行情。

用法: python -X utf8 seg_verify.py klines_1m_wide.csv 0.50 0.035 [段数]
输出: 每段的 笔数 / 天数 / 均值 / 胜率 / 顺序复利终值（该段内），以及全样本对照。
"""
import sys
import numpy as np
import grid3

COST2 = 0.0008


def main():
    fname = sys.argv[1]
    tp = float(sys.argv[2])
    sl = float(sys.argv[3])
    nseg = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    tf = 300 if '5m' in fname else (60 if '1m' in fname else (86400 if '1d' in fname else 3600))

    print('=' * 96)
    print('%s  TP=%.1f%%  SL=%.1f%%  持有 %d 根(=%.1f 小时)  按时间切 %d 段'
          % (fname, tp * 100, sl * 100, grid3.hold_bars(tf),
             grid3.hold_bars(tf) * tf / 3600.0, nseg))
    print('=' * 96)

    arr = grid3.collect(fname, tf)
    if arr is None:
        print('无信号')
        return
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
    t0 = np.datetime64(int(ts[0]), 'ms')
    t1 = np.datetime64(int(ts[-1]), 'ms')
    print('信号 %d  区间 %s ~ %s' % (N, str(t0), str(t1)))
    print('')

    hdr = '%-6s %-24s %8s %7s %10s %8s %12s %10s'
    print(hdr % ('段', '起止', '笔数', '天数', '均值/笔', '胜率', '段内复利', '最惨一笔'))
    print('-' * 96)

    idx_all = np.arange(N)
    for k, idx in enumerate(np.array_split(idx_all, nseg)):
        if len(idx) == 0:
            continue
        n = net[idx]
        nd = len(np.unique(day[idx]))
        byd = np.clip(n, -0.999, None)
        final = float(np.prod(1.0 + byd)) if len(byd) else 1.0
        print(hdr % ('%d' % (k + 1),
                     '%s~%s' % (str(np.datetime64(int(ts[idx[0]]), 'ms'))[:10],
                                str(np.datetime64(int(ts[idx[-1]]), 'ms'))[:10]),
                     len(idx), nd, '%+.4f%%' % (n.mean() * 100), '%.1f%%' % ((n > 0).mean() * 100),
                     ('%.3e' % final) if abs(final) > 1e6 else '%.3f' % final,
                     '%.1f%%' % (n.min() * 100)))

    n = net
    byd = np.clip(n, -0.999, None)
    final = float(np.prod(1.0 + byd))
    print('-' * 96)
    print(hdr % ('全部', '%s~%s' % (str(t0)[:10], str(t1)[:10]),
                 N, len(np.unique(day)), '%+.4f%%' % (n.mean() * 100),
                 '%.1f%%' % ((n > 0).mean() * 100),
                 ('%.3e' % final) if abs(final) > 1e6 else '%.3f' % final,
                 '%.1f%%' % (n.min() * 100)))
    print('')
    print('读法：若某一段均值明显高于其它段（>3 倍），该结果就是那一段行情的产物，不是稳定规则。')
    print('诚实边界：零滑点；TP 按 high、SL 按 low 触价，假设能成交；信号重叠未去重；'
          '段内复利假设串行满仓。')


if __name__ == '__main__':
    main()
