# -*- coding: utf-8 -*-
"""by_symbol.py — 输赢集中在哪些币

口径：在线版（无前视），θ=1%、TP=2%、最长 8h、成本 0.04% 单边。
输出：按币的笔数/均值/胜率/总贡献，以及集中度（前 N 个币吃掉了多少盈亏）。
"""
import os
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
OUT = r'D:/中信证券/coinlab/evidence/by_symbol.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR, TP, THETA = -0.02, 3.0, 0.02, 0.01

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def run(fname):
    t0 = time.time()
    df = pd.read_csv(os.path.join(D, fname),
                     usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
    per = {}
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
        c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
        n = len(c)
        if n < 600 + H:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
        nets = []
        for i in range(W + 300, n - H - 1):
            if not (ret24[i] <= DIP and volr[i] >= VOLR):
                continue
            entry = float(c[i]); peak = entry; out = None
            for j in range(1, H):
                k = i + j
                hi = float(h[k]); lo = float(l[k]); cl = float(c[k])
                if hi > peak:
                    peak = hi
                if hi >= entry * (1 + TP):
                    out = TP - 2 * COST; break
                if lo <= peak * (1 - THETA):
                    out = cl / entry - 1.0 - 2 * COST; break
            if out is None:
                out = float(c[i + H - 1]) / entry - 1.0 - 2 * COST
            nets.append(out)
        if nets:
            a = np.array(nets)
            per[sym] = (len(a), a.mean(), (a > 0).mean(), a.sum(), a.min(), a.max())
    del df

    rows = [(s, *v) for s, v in per.items()]
    rows.sort(key=lambda x: x[4])          # 按总贡献升序（最亏在前）
    tot = sum(r[4] for r in rows)
    npnl = sum(r[1] for r in rows)
    log('')
    log('%s  θ=1%%  币数 %d  总笔数 %d  总盈亏 %+.1f（单位:百分点/笔之和）'
        % (fname, len(rows), sum(r[1] for r in rows), tot))
    log('')
    log('=== 最亏的 12 个币 ===')
    log('%-13s %7s %9s %7s %10s %9s %9s' %
        ('币', '笔数', '均值/笔', '胜率', '总贡献', '最惨一笔', '最好一笔'))
    log('-' * 74)
    for r in rows[:12]:
        log('%-13s %7d %+8.3f%% %6.1f%% %+10.1f %+8.1f%% %+8.1f%%'
            % (r[0], r[1], r[2] * 100, r[3] * 100, r[4], r[5] * 100, r[6] * 100))
    log('')
    log('=== 最赚的 12 个币 ===')
    log('%-13s %7s %9s %7s %10s %9s %9s' %
        ('币', '笔数', '均值/笔', '胜率', '总贡献', '最惨一笔', '最好一笔'))
    log('-' * 74)
    for r in rows[-12:][::-1]:
        log('%-13s %7d %+8.3f%% %6.1f%% %+10.1f %+8.1f%% %+8.1f%%'
            % (r[0], r[1], r[2] * 100, r[3] * 100, r[4], r[5] * 100, r[6] * 100))

    neg = [r for r in rows if r[4] < 0]
    pos = [r for r in rows if r[4] >= 0]
    log('')
    log('=== 集中度 ===')
    log('亏损币 %d 个，合计亏 %+.1f；盈利币 %d 个，合计赚 %+.1f；净 %+.1f'
        % (len(neg), sum(r[4] for r in neg), len(pos), sum(r[4] for r in pos), tot))
    worst10 = sum(r[4] for r in rows[:10])
    log('最亏的 10 个币合计 %+.1f，占总亏损 %.1f%%'
        % (worst10, worst10 / sum(r[4] for r in neg) * 100 if neg else 0))
    best10 = sum(r[4] for r in rows[-10:])
    log('最赚的 10 个币合计 %+.1f，占总盈利 %.1f%%'
        % (best10, best10 / sum(r[4] for r in pos) * 100 if pos else 0))
    log('')
    log('=== 币级正负分布 ===')
    for th_pct in (0.2, 0.1, 0.0):
        cnt = sum(1 for r in rows if r[2] > th_pct)
        log('  均值 > %+4.1f%% 的币：%d / %d' % (th_pct * 100, cnt, len(rows)))
    log('')
    log('耗时 %.0fs' % (time.time() - t0))


def main():
    log('=' * 96)
    log('输赢集中在哪些币（在线版 θ=1%%）   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 96)
    run('klines_5m_3y.csv')
    log('')
    log('诚实边界：零滑点、成本 0.04%% 单边、按币独立结算、含样本内；')
    log('         "总贡献" = 该币所有笔收益之和（百分点），用于看谁在拖后腿。')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
