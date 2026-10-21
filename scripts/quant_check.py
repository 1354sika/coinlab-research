# -*- coding: utf-8 -*-
"""quant_check.py — 为什么「胜率 63.9%、中位 +1.92%」还是不行

回答：均值是负的。赢的笔只赢 1.92%（TP 上限），输的笔没有上限。
本脚本把 θ=8% / 3% / 1% 三档的**逐笔收益分位数**和**复利曲线崩溃点**打出来。
"""
import os
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
OUT = r'D:/中信证券/coinlab/evidence/quant_check.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR, TP = -0.02, 3.0, 0.02
THETAS = [0.08, 0.03, 0.01]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def run(fname):
    t0 = time.time()
    df = pd.read_csv(os.path.join(D, fname),
                     usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
    trades = {th: [] for th in THETAS}
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
        c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
        t = g.open_time.to_numpy(); n = len(c)
        if n < 600 + H:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
        for i in range(W + 300, n - H - 1):
            if not (ret24[i] <= DIP and volr[i] >= VOLR):
                continue
            entry = float(c[i])
            for th in THETAS:
                peak = entry
                done = False
                for j in range(1, H):
                    k = i + j
                    hi = float(h[k]); lo = float(l[k]); cl = float(c[k])
                    if hi > peak:
                        peak = hi
                    if hi >= entry * (1 + TP):
                        trades[th].append((int(t[k]), TP - 2 * COST)); done = True; break
                    if lo <= peak * (1 - th):
                        trades[th].append((int(t[k]), cl / entry - 1.0 - 2 * COST)); done = True; break
                if not done:
                    cl = float(c[i + H - 1])
                    trades[th].append((int(t[i + H - 1]), cl / entry - 1.0 - 2 * COST))
    del df

    for th in THETAS:
        rows = sorted(trades[th], key=lambda x: x[0])
        nets = np.array([r[1] for r in rows])
        q = np.percentile(nets, [1, 5, 25, 50, 75, 95, 99])
        log('')
        log('θ=%.1f%%  笔数 %d  均值 %+.4f%%' % (th * 100, len(nets), nets.mean() * 100))
        log('  分位  p1 %+.2f%%  p5 %+.2f%%  p25 %+.2f%%  中位 %+.2f%%  p75 %+.2f%%  p95 %+.2f%%  p99 %+.2f%%'
            % tuple(q * 100))
        loss = nets[nets <= 0]
        gain = nets[nets > 0]
        log('  赢 %d 笔 平均 %+.2f%%（上限就是 TP 的 +1.92%%，赢不大）'
            % (len(gain), gain.mean() * 100 if len(gain) else 0))
        log('  亏 %d 笔 平均 %+.2f%%（没有止损，亏可以无限）  最惨一笔 %+.1f%%'
            % (len(loss), loss.mean() * 100 if len(loss) else 0, nets.min() * 100))
        # 复利曲线：找从峰值跌到 50% 以下的时刻
        eq = 1.0; peak = 1.0; mdd = 0.0; broke_at = None
        for idx, x in enumerate(nets, 1):
            eq *= (1 + x)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
            if broke_at is None and eq < 0.5:
                broke_at = idx
        log('  复利终值 %.3fx  最大回撤 %.2f%%  首次跌破半仓：%s'
            % (eq, mdd * 100, ('第 %d 笔' % broke_at) if broke_at else '从未'))
    log('')
    log('耗时 %.0fs' % (time.time() - t0))


def main():
    log('=' * 96)
    log('为什么「胜率 63.9%%、中位 +1.92%%」仍然赚不到钱   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 96)
    run('klines_5m_3y.csv')
    log('')
    log('诚实边界：零滑点、成本 0.04%% 单边、按币独立结算、含样本内。')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
