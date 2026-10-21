# -*- coding: utf-8 -*-
"""grid_daily_stop.py — TP/SL 大网格（1%~150%，步长 1%）+ 日内到 1.1x 停手

用户口径：
  · 止损、止盈单位 0.01（=1%），上限 1.5（=150%）→ 150×150 = 22,500 组
  · 不管一天做几笔，当天本金累到 1.1 倍就停，次日以新本金继续（复利）
  · 1.1 是猜的，实际要算：同时扫 1.05 / 1.10 / 1.20 / 1.30 / 1.50

做法：
  1) 收集所有入场信号 + 未来 96 根 5m 的相对 high/low/close
  2) 22,500 组各算出每笔收益（向量化）
  3) 对均值排名前 TOPN 的组合，再做「日内到 Nx 停手」的顺序模拟
  4) 输出正期望组数、最优榜、TP=2.5% 专表

用法：python -X utf8 grid_daily_stop.py [数据文件...]
"""
import os
import sys
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
EVI = r'D:/中信证券/coinlab/evidence'
OUT = os.path.join(EVI, 'grid_daily_stop.txt')
H, COST2, W = 96, 0.0008, 48
DIP, VOLR = -0.02, 3.0
GRID = np.round(np.arange(0.01, 1.501, 0.01), 4)      # 150 档
TARGETS = [1.05, 1.10, 1.20, 1.30, 1.50]
TOPN = 200                                            # 只对前 200 名做日内模拟

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def collect(fname, tf):
    df = pd.read_csv(os.path.join(D, fname),
                     usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
    need_agg = tf < 300
    hs, ls, cs, ts = [], [], [], []
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        if need_agg:
            g = g.assign(bucket=(g.open_time // 300000))
            g = (g.groupby('bucket')
                   .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                        close=('close', 'last'), volume=('volume', 'sum'))
                   .reset_index())
        h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
        c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
        t = g.open_time.to_numpy(); n = len(c)
        if n < 600 + H:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
        idxs = np.where((ret24 <= DIP) & (volr >= VOLR))[0]
        idxs = idxs[(idxs >= W + 300) & (idxs < n - H - 1)]
        for i in idxs:
            e = float(c[i])
            if e <= 0:
                continue
            seg = slice(i + 1, i + 1 + H)
            hs.append(h[seg] / e - 1.0)
            ls.append(l[seg] / e - 1.0)
            cs.append(c[seg] / e - 1.0)
            ts.append(int(t[i]))
    del df
    if not hs:
        return None
    hh = np.stack(hs).astype(np.float32)
    ll = np.stack(ls).astype(np.float32)
    cc = np.stack(cs).astype(np.float32)
    tt = np.array(ts, dtype='int64')
    o = np.argsort(tt)
    return hh[o], ll[o], cc[o], tt[o]


def first_hit(rel, level, above=True):
    m = (rel >= level) if above else (rel <= level)
    idx = m.argmax(axis=1).astype(np.int16)
    idx[~m.any(axis=1)] = 127
    return idx


def daily_sim(ts, nets, target):
    """按 UTC 日分组：日内累乘，达到 target 立即停手；跨日复利"""
    day = ts // 86400000
    eq = 1.0
    n_stop = 0
    n_days = 0
    # 逐日切片（ts 已排序，day 单调）
    bounds = np.flatnonzero(np.diff(day)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(day)]
    for a, b in zip(starts, ends):
        n_days += 1
        lr = np.log1p(nets[a:b])
        cs = np.cumsum(lr)
        lim = np.log(target)
        k = np.searchsorted(cs, lim, side='left')
        if k < len(cs):
            eq *= target
            n_stop += 1
        else:
            eq *= float(np.exp(cs[-1])) if len(cs) else 1.0
    return eq, n_stop, n_days


def run(fname, tf):
    t0 = time.time()
    arr = collect(fname, tf)
    if arr is None:
        log('%-26s 无信号' % fname); return None
    hh, ll, cc, ts = arr
    N = len(hh)
    j_tp = np.stack([first_hit(hh, g, True) for g in GRID])
    j_sl = np.stack([first_hit(ll, -g, False) for g in GRID])
    last = cc[:, -1]

    means = np.zeros((len(GRID), len(GRID)))
    wins = np.zeros((len(GRID), len(GRID)))
    nets_store = None
    cand = []
    for a in range(len(GRID)):
        jt = j_tp[a]
        for b in range(len(GRID)):
            js = j_sl[b]
            none = (jt == 127) & (js == 127)
            r = np.where(none, last - COST2,
                         np.where(jt < js, GRID[a] - COST2, -GRID[b] - COST2))
            means[a, b] = r.mean()
            wins[a, b] = (r > 0).mean()
            if r.mean() > 0:
                cand.append((float(r.mean()), a, b, r))

    cand.sort(key=lambda z: -z[0])
    log('')
    log('=' * 112)
    log('%s  周期 %ds  信号 %d  网格 150×150=22,500 组  耗时 %.0fs'
        % (fname, tf, N, time.time() - t0))
    log('=' * 112)
    log('均值为正的组合：%d / 22500（%.1f%%）' % (len(cand), len(cand) / 22500 * 100))

    # TP=2.5% 专表
    a25 = int(np.argmin(np.abs(GRID - 0.025)))
    log('')
    log('--- TP=2.5%% 这一行的前 8 个 SL ---')
    row = sorted([(means[a25, b], b) for b in range(len(GRID))], reverse=True)[:8]
    log('%-8s %10s %10s' % ('SL', '均值/笔', '胜率'))
    for m, b in row:
        log('%-8s %+9.4f%% %9.1f%%' % ('%.0f%%' % (GRID[b] * 100), m * 100, wins[a25, b] * 100))

    # 最优榜 + 日内停手
    log('')
    log('--- 均值前 12 名（原始，无日内停手） ---')
    log('%-8s %-8s %10s %9s' % ('TP', 'SL', '均值/笔', '胜率'))
    for m, a, b, _ in cand[:12]:
        log('%-8s %-8s %+9.4f%% %8.1f%%' % ('%.0f%%' % (GRID[a] * 100),
                                            '%.0f%%' % (GRID[b] * 100), m * 100,
                                            wins[a, b] * 100))

    log('')
    log('--- 前 %d 名加上「日内到 Nx 停手」后的复利终值（起点 1.0） ---' % TOPN)
    log('%-8s %-8s %8s %10s %10s %10s %10s %10s' %
        ('TP', 'SL', '盈亏比', '无停手', '到1.05', '到1.10', '到1.20', '到1.50'))
    best_daily = None
    for m, a, b, r in cand[:TOPN]:
        vals = []
        for tg in TARGETS:
            eq, nstop, nday = daily_sim(ts, r, tg)
            vals.append(eq * 1000)          # 起点 1000
        eq0 = 1000.0 * float(np.prod(1 + r))
        tag = ('%.2e' % eq0) if abs(eq0) > 1e6 else ('%.2f' % eq0)
        txt = ' '.join(('%10.2f' % v) if v < 1e6 else ('%10.2e' % v) for v in vals)
        log('%-8s %-8s %+8.4f%% %10s %s' %
            ('%.0f%%' % (GRID[a] * 100), '%.0f%%' % (GRID[b] * 100), m * 100, tag, txt))
        if best_daily is None or vals[1] > best_daily[0]:
            best_daily = (vals[1], a, b, m)
    if best_daily:
        log('')
        log('★ 「日内到 1.10 停手」下终值最高：TP=%.0f%% SL=%.0f%% 均值 %+.4f%% → 终值 %.2f（起点 1000）'
            % (GRID[best_daily[1]] * 100, GRID[best_daily[2]] * 100,
               best_daily[3] * 100, best_daily[0]))
    return dict(file=fname, npos=len(cand), best=cand[0][0] if cand else 0,
                btp=GRID[cand[0][1]] if cand else 0, bsl=GRID[cand[0][2]] if cand else 0,
                best_daily=best_daily)


def main():
    log('=' * 112)
    log('TP/SL 大网格 1%%~150%%（22,500 组）+ 日内到 Nx 停手   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('入场：2h 跌 2%% 且量比>=3；出场：纯 TP/SL，最长 8h；成本 0.04%% 单边')
    log('=' * 112)
    targets = sys.argv[1:] or ['klines_5m_3y.csv']
    for f in targets:
        rr = run(f, 300 if '5m' in f else (60 if '1m' in f else 3600))
        if rr:
            log('')
    log('诚实边界：零滑点；TP 按 high、SL 按 (1-SL) 触价，均假设能成交；')
    log('         日内停手按 UTC 日切分；含样本内；1.05~1.50 五档阈值为扫描对象。')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
