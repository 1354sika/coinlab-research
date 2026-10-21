# -*- coding: utf-8 -*-
"""theta_scan_online.py — 在线版（无前视）扫描回落兜底 θ

背景：compare_online_vs_lookahead 证明 best_curve 的 +1.0822% 里 97.6% 来自前视。
      在线版 θ=1% 时全历史为负（-0.07%~-0.05%/笔）。
      本脚本扫 θ = 0.5% / 1% / 1.5% / 2% / 3% / 5%，看放宽兜底能不能转正。

规则（在线、谁先触发谁生效）：
  入场：5m 收盘 ret24 <= -2% 且 量比(当根量/前24根均量) >= 3
  出场：优先 high 触及 +2% 卖；否则 low 从入场后最高价回落 θ 卖；最长 96 根 5m
  成本：0.04% 单边

用法：python -X utf8 theta_scan_online.py [数据文件名]
"""
import os
import sys
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
OUT = r'D:/中信证券/coinlab/evidence/theta_scan_online.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR, TP = -0.02, 3.0, 0.02
THETAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def run(fname):
    p = os.path.join(D, fname)
    t0 = time.time()
    df = pd.read_csv(p, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
    all_trades = {th: [] for th in THETAS}
    nsig = 0

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
            nsig += 1
            entry = float(c[i])
            # 每个 θ 独立推进出场
            for th in THETAS:
                peak = entry
                done = False
                for j in range(1, H):
                    k = i + j
                    hi = float(h[k]); lo = float(l[k]); cl = float(c[k])
                    if hi > peak:
                        peak = hi
                    if hi >= entry * (1 + TP):
                        all_trades[th].append((int(t[k]), TP - 2 * COST, j)); done = True; break
                    if lo <= peak * (1 - th):
                        all_trades[th].append((int(t[k]), cl / entry - 1.0 - 2 * COST, j)); done = True; break
                if not done:
                    cl = float(c[i + H - 1])
                    all_trades[th].append((int(t[i + H - 1]), cl / entry - 1.0 - 2 * COST, H - 1))

    log('')
    log('%s  信号 %d 个  耗时 %.0fs' % (fname, nsig, time.time() - t0))
    log('%-8s %8s %10s %10s %8s %10s %10s %8s' %
        ('θ', '笔数', '均值/笔', '中位', '胜率', '复利终值', '最大回撤', '到1.5x'))
    log('-' * 86)
    for th in THETAS:
        rows = sorted(all_trades[th], key=lambda x: x[0])
        if not rows:
            continue
        nets = np.array([r[1] for r in rows])
        eq = 1.0; peak = 1.0; mdd = 0.0; to15 = None
        for idx, x in enumerate(nets, 1):
            eq *= (1 + x)
            peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
            if to15 is None and eq >= 1.5:
                to15 = idx
        eqtxt = ('%.3fx' % eq) if abs(eq) < 1e6 else ('%.2e' % eq)
        log('%-8s %8d %+9.4f%% %+9.4f%% %7.1f%% %12s %9.2f%% %8s' %
            ('%.1f%%' % (th * 100), len(nets), nets.mean() * 100,
             float(np.median(nets)) * 100, (nets > 0).mean() * 100,
             eqtxt, mdd * 100, ('%d笔' % to15) if to15 else '未达'))
    del df


def main():
    log('=' * 96)
    log('在线版 · 回落兜底 θ 扫描（无前视，谁先触发谁生效）   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('规则：2h 跌 2%% 且量比>=3 入场；+2%% 止盈 / 峰值回落 θ 卖 / 最长 8h；成本 0.04%% 单边')
    log('=' * 96)
    files = sys.argv[1:] or ['klines_5m_3y.csv']
    for f in files:
        run(f)
    log('')
    log('诚实边界：零滑点；TP 按 high 触价、θ 按 low 触价，均假设能成交（实盘需排队）；')
    log('         按币独立结算；含样本内。')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
