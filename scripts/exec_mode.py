# -*- coding: utf-8 -*-
"""exec_mode.py — 同一组 (TP,SL) 在三种「成交假设」下分别是多少。

为什么必须做：+22.84%/笔 是用「当天 high 摸到 +50% 就当卖在 +50%」算的。
真实挂单要在那一秒排到队，还要祈祷它不只闪一下。三种口径：

  A  high 触价（现状）：high >= entry*(1+TP) 就当以 TP 成交
  B  close 确认      ：收盘价 >= entry*(1+TP) 才成交（要站稳，不是插针）
  C  延迟 1 根       ：high 触了，但按下一根收盘价成交（排队/滑点/反应时间）

止损侧同理：A 用 low 触价，B 用 close 破，C 延迟一根。

用法: python -X utf8 exec_mode.py klines_1m_wide.csv 0.50 0.035
"""
import sys
import numpy as np
import grid3

COST2 = 0.0008
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
    print('=' * 100)
    print('%s  TP=%.1f%%  SL=%.1f%%  持有 %d 根(=%.1f 小时)  信号 %d'
          % (fname, tp * 100, sl * 100, H, H * tf / 3600.0, N))
    print('=' * 100)

    jtA = first_idx(hh >= tp)
    jtB = first_idx(cc >= tp)
    jsA = first_idx(ll <= -sl)
    jsB = first_idx(cc <= -sl)

    def settle(jt, js, tp_px, sl_px):
        none = (jt == HIT) & (js == HIT)
        return np.where(none, cc[:, -1] - COST2,
                        np.where(jt < js, tp_px - COST2, sl_px - COST2)).astype(np.float64)

    # A：high 触 TP / low 触 SL
    netA = settle(jtA, jsA, tp, -sl)
    # B：close 确认
    netB = settle(jtB, jsB, tp, -sl)
    # C：延迟一根 —— 触了之后按下一根收盘价成交
    netC = np.empty(N, dtype=np.float64)
    for i in range(N):
        jt, js = int(jtA[i]), int(jsA[i])
        if jt == HIT and js == HIT:
            netC[i] = cc[i, -1] - COST2
            continue
        first = jt if jt < js else js
        nxt = first + 1
        px = cc[i, nxt] if nxt < H else cc[i, first]
        netC[i] = px - COST2

    hdr = '%-10s %8s %10s %8s %10s %12s %10s'
    print(hdr % ('成交假设', '笔数', '均值/笔', '胜率', '中位/笔', '顺序复利', '最惨一笔'))
    print('-' * 100)
    for name, net in (('A high触价', netA), ('B close确认', netB), ('C 延迟1根', netC)):
        byd = np.clip(net, -0.999, None)
        fin = float(np.prod(1.0 + byd))
        print(hdr % (name, N, '%+.4f%%' % (net.mean() * 100), '%.1f%%' % ((net > 0).mean() * 100),
                     '%+.4f%%' % (np.median(net) * 100),
                     ('%.3e' % fin) if abs(fin) > 1e6 else '%.3f' % fin,
                     '%.1f%%' % (net.min() * 100)))
    print('')
    print('A→B 掉幅 %.2f 个百分点；A→C 掉幅 %.2f 个百分点'
          % ((netA.mean() - netB.mean()) * 100, (netA.mean() - netC.mean()) * 100))
    print('读法：若 B/C 掉到负数或接近零，那 +22%% 全部来自「卖在当天最高点」这个不存在的假设。')
    print('诚实边界：零滑点（C 用下一根收盘近似排队与延迟）；信号重叠未去重；'
          '止损按触价即成交，真实市价止损还有滑点。')


if __name__ == '__main__':
    main()
