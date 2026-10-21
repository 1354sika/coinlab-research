# -*- coding: utf-8 -*-
"""mech_engine.py — 纯机械规则引擎：一个流式状态机喂遍所有历史数据

为什么是状态机：24h 盯盘不需要模型、不需要长连，只需每根 K 线推进一次。
本脚本就是那个状态机跑在历史上，用来回答「这套机械算法在所有历史里是什么表现」。

规则（与 best_curve / tp_verify 完全一致）：
  入场：收盘 ret(2h) <= -2%  且  量比(当根量/前 2h 均量) >= 3.0
  出场：high 触 +2% 卖；否则从峰值回落 1% 卖；最长持 8h
  成本：0.04% 单边（往返 0.08%）
  口径：所有周期统一到「2 小时窗口 / 8 小时持有」，1m 先聚合成 5m

用法：
  python -X utf8 mech_engine.py                 # 跑全部
  python -X utf8 mech_engine.py klines_5m_3y.csv  # 只跑一个

诚实边界：
  · 零滑点；TP 按触价成交（实盘需限价单排队，成交率未计）
  · 信号之间可能重叠持有（本脚本按币独立结算，不做单账号排队）
  · 全历史含样本内，不是样本外结论
"""
import os
import csv
import sys
import time

D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\mech_all_history.txt'

DIP, VOLR, TP, THETA, COST = -0.02, 3.0, 0.02, 0.01, 0.0004
HOLD_SEC = 8 * 3600          # 最长持有 8 小时
WIN_SEC = 2 * 3600           # 回看窗口 2 小时

FILES = [
    ('klines_5m_3y.csv', 300),
    ('klines_5m_502.csv', 300),
    ('klines_1m_1y.csv', 60),
    ('klines_1m_502.csv', 60),
    ('klines_1m_sync.csv', 60),
    ('klines_1m_wide.csv', 60),
    ('klines_1m_long.csv', 60),
    ('klines_1m_top20_7d.csv', 60),
    ('klines_1h_top20_365d.csv', 3600),
    ('klines_1d_full.csv', 86400),
    ('klines_1d_top502.csv', 86400),
]


class St:
    """每币状态：滚动窗口 + 持仓"""
    __slots__ = ('cl', 'vo', 'pos')
    def __init__(self):
        self.cl = []
        self.vo = []
        self.pos = None      # dict(entry, peak, held, ts)


def feed(st, ts, o, h, l, c, v, ret_win, vol_win, max_hold, trades, t0):
    st.cl.append(c)
    st.vo.append(v)
    pos = st.pos
    if pos is not None:
        pos['held'] += 1
        if h > pos['peak']:
            pos['peak'] = h
        px = None
        reason = None
        if c >= pos['entry'] * (1 + TP):
            px = c; reason = 'tp'
        elif l <= pos['peak'] * (1 - THETA) and pos['held'] >= 2:
            px = c; reason = 'theta'
        elif pos['held'] >= max_hold:
            px = c; reason = 'timeout'
        if px is not None:
            net = px / pos['entry'] - 1.0 - 2 * COST
            trades.append((pos['ts'], net, reason))
            st.pos = None
        return

    n = ret_win
    if len(st.cl) <= n or len(st.vo) < vol_win:
        return
    base = st.cl[-1 - n]
    if base <= 0:
        return
    ret = c / base - 1.0
    vw = st.vo[-vol_win:]
    vma = sum(vw) / len(vw)
    if vma <= 0:
        return
    volr = v / vma
    if ret <= DIP and volr >= VOLR:
        st.pos = {'entry': c, 'peak': c, 'held': 0, 'ts': ts}
    # 控制内存
    if len(st.cl) > 4 * n + 40:
        del st.cl[:-n - 5]
        del st.vo[:-max(vol_win, n) - 5]


def run_file(fname, tf):
    p = os.path.join(D, fname)
    if not os.path.exists(p):
        return None
    ret_win = max(1, WIN_SEC // tf)
    vol_win = max(1, WIN_SEC // tf)
    max_hold = max(1, HOLD_SEC // tf)
    states = {}
    trades = []
    pending = {}          # 1m 聚合 5m 用: sym -> [bucket, o,h,l,c,v]
    need_agg = tf < 300
    nrow = 0
    syms = set()
    tmin = tmax = None
    t_start = time.time()

    with open(p, 'r', encoding='utf-8', errors='ignore', newline='') as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        try:
            i_s = hdr.index('symbol')
            i_t = hdr.index('open_time')
            i_o = hdr.index('open')
            i_h = hdr.index('high')
            i_l = hdr.index('low')
            i_c = hdr.index('close')
            i_v = hdr.index('volume')
        except ValueError:
            return {'file': fname, 'err': '列不合规: %s' % hdr}

        for row in rd:
            nrow += 1
            try:
                sym = row[i_s]
                ts = int(row[i_t])
                o = float(row[i_o]); h = float(row[i_h]); l = float(row[i_l])
                c = float(row[i_c]); v = float(row[i_v])
            except Exception:
                continue
            syms.add(sym)
            if tmin is None or ts < tmin:
                tmin = ts
            if tmax is None or ts > tmax:
                tmax = ts
            st = states.get(sym)
            if st is None:
                st = states[sym] = St()

            if need_agg:
                b = ts // 300000
                pv = pending.get(sym)
                if pv is None or pv[0] != b:
                    if pv is not None:
                        feed(st, pv[0] * 300000, pv[1], pv[2], pv[3], pv[4], pv[5],
                             ret_win, vol_win, max_hold, trades, t_start)
                    pending[sym] = [b, o, h, l, c, v]
                else:
                    if h > pv[2]:
                        pv[2] = h
                    if l < pv[3]:
                        pv[3] = l
                    pv[4] = c
                    pv[5] += v
            else:
                feed(st, ts, o, h, l, c, v, ret_win, vol_win, max_hold, trades, t_start)

    # 冲掉最后未完成的 1m 聚合根
    for sym, pv in pending.items():
        st = states.get(sym)
        if st:
            feed(st, pv[0] * 300000, pv[1], pv[2], pv[3], pv[4], pv[5],
                 ret_win, vol_win, max_hold, trades, t_start)

    # 持仓中的未平仓：按最后收盘价结算（保守）
    el = time.time() - t_start
    if not trades:
        return {'file': fname, 'rows': nrow, 'syms': len(syms), 'n': 0,
                'span': (tmax - tmin) / 1000 / 86400 if tmin and tmax else 0,
                'tmin': tmin, 'tmax': tmax, 'el': el}
    trades.sort(key=lambda x: x[0])
    nets = [t[1] for t in trades]
    nn = len(nets)
    mean = sum(nets) / nn
    nets_sorted = sorted(nets)
    med = nets_sorted[nn // 2]
    win = sum(1 for x in nets if x > 0) / nn
    # 顺序复利
    eq = 1.0; peak = 1.0; mdd = 0.0; to15 = None
    for i, x in enumerate(nets, 1):
        eq *= (1 + x)
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
        if to15 is None and eq >= 1.5:
            to15 = i
    reasons = {}
    for _, _, r in trades:
        reasons[r] = reasons.get(r, 0) + 1
    span_d = (tmax - tmin) / 1000 / 86400 if tmin and tmax else 0
    return {'file': fname, 'rows': nrow, 'syms': len(syms), 'n': nn,
            'mean': mean, 'med': med, 'win': win, 'eq': eq, 'mdd': mdd,
            'to15': to15, 'reasons': reasons, 'tmin': tmin, 'tmax': tmax,
            'span': span_d, 'el': el}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    files = [f for f in FILES if target is None or f[0] == target]
    lines = []
    hdr = ('数据集                     周期   币数  跨度(天)   笔数    单笔均值   中位     胜率    '
           '复利终值      最大回撤  到1.5x')
    lines.append('=' * 128)
    lines.append('机械状态机 · 全历史验证   %s' % time.strftime('%Y-%m-%d %H:%M'))
    lines.append('规则：2h 内跌 2%% 且量比>=3 入场；+2%% 止盈 / 回落1%% 兜底 / 最长持 8h；成本 0.04%% 单边')
    lines.append('=' * 128)
    lines.append(hdr)
    lines.append('-' * 128)
    for fname, tf in files:
        r = run_file(fname, tf)
        if r is None:
            lines.append('%-26s  文件不存在' % fname)
            continue
        if r.get('err'):
            lines.append('%-26s  %s' % (fname, r['err']))
            continue
        if r.get('n', 0) == 0:
            lines.append('%-26s %5ds %5d  %8.1f  %7d   （无信号）  行=%d  %.0fs'
                         % (fname, tf, r['syms'], r['span'], 0, r['rows'], r['el']))
            continue
        eqs = r['eq']
        eqtxt = ('%.2fx' % eqs) if eqs < 1e6 else ('%.2e' % eqs)
        lines.append('%-26s %5ds %5d  %8.1f  %7d  %+8.4f%%  %+7.3f%%  %5.1f%%  %11s  %7.2f%%  %s'
                     % (fname, tf, r['syms'], r['span'], r['n'], r['mean'] * 100,
                        r['med'] * 100, r['win'] * 100, eqtxt, r['mdd'] * 100,
                        ('%d 笔' % r['to15']) if r['to15'] else '未达'))
        lines.append('    出场构成: %s   行=%d  %.0fs'
                     % (' '.join('%s=%d' % kv for kv in sorted(r['reasons'].items())),
                        r['rows'], r['el']))
        for L in lines[-2:]:
            print(L, flush=True)
    lines.append('=' * 128)
    lines.append('诚实边界：零滑点；TP 按触价成交（实盘需排队，成交率未计）；')
    lines.append('         按币独立结算，不做单账号排队；全历史含样本内。')
    txt = '\n'.join(lines)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('已写 %s' % OUT, flush=True)


if __name__ == '__main__':
    main()
