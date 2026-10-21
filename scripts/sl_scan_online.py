# -*- coding: utf-8 -*-
"""sl_scan_online.py — 在线版加「成本硬止损」，扫 SL × θ

背景（theta_scan_online.txt）：θ 从 0.5% 扫到 8% 全为负，因为「从峰值回落」不是止损——
价格一路跌时峰值不动，它永远不触发，直到 8 小时到期。
本脚本补上真正缺的一环：**跌破入场价 SL% 就砍**。

规则（在线，谁先触发谁生效）：
  入场：5m 收盘 ret24 <= -2% 且 量比 >= 3
  出场优先级（同一根内）：
    1) high 触 +2%          → 止盈
    2) low 跌破 entry*(1-SL) → 硬止损，按 entry*(1-SL) 成交
    3) low 从峰值回落 θ      → 跟踪止盈，按 close 成交
    4) 持满 96 根 5m（8h）
  成本：0.04% 单边

用法：python -X utf8 sl_scan_online.py [数据文件]
"""
import os
import sys
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
OUT = r'D:/中信证券/coinlab/evidence/sl_scan_online.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR, TP = -0.02, 3.0, 0.02
SLS = [0.01, 0.02, 0.03, 0.05, 0.99]      # 0.99 = 不设止损
THETAS = [0.01, 0.015, 0.03]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def run(fname):
    p = os.path.join(D, fname)
    t0 = time.time()
    df = pd.read_csv(p, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
    combos = [(sl, th) for sl in SLS for th in THETAS]
    trades = {k: [] for k in combos}
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
            for (sl, th) in combos:
                peak = entry
                out = None
                for j in range(1, H):
                    k = i + j
                    hi = float(h[k]); lo = float(l[k]); cl = float(c[k])
                    if hi > peak:
                        peak = hi
                    if hi >= entry * (1 + TP):
                        out = (int(t[k]), TP - 2 * COST); break
                    if lo <= entry * (1 - sl):
                        out = (int(t[k]), -sl - 2 * COST); break
                    if lo <= peak * (1 - th):
                        out = (int(t[k]), cl / entry - 1.0 - 2 * COST); break
                if out is None:
                    cl = float(c[i + H - 1])
                    out = (int(t[i + H - 1]), cl / entry - 1.0 - 2 * COST)
                trades[(sl, th)].append(out)

    log('')
    log('%s  信号 %d 个  耗时 %.0fs' % (fname, nsig, time.time() - t0))
    log('%-7s %-6s %8s %11s %10s %8s %11s %9s %8s' %
        ('止损SL', 'θ', '笔数', '均值/笔', '中位', '胜率', '复利终值', '最大回撤', '到1.5x'))
    log('-' * 92)
    best = None
    for (sl, th) in combos:
        rows = sorted(trades[(sl, th)], key=lambda x: x[0])
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
        log('%-7s %-6s %8d %+10.4f%% %+9.4f%% %7.1f%% %12s %8.2f%% %8s' %
            ('--' if sl > 0.5 else '%.1f%%' % (sl * 100), '%.1f%%' % (th * 100),
             len(nets), nets.mean() * 100, float(np.median(nets)) * 100,
             (nets > 0).mean() * 100, eqtxt, mdd * 100,
             ('%d笔' % to15) if to15 else '未达'))
        if best is None or nets.mean() > best[0]:
            best = (nets.mean(), sl, th)
    if best:
        log('')
        log('最优点：SL=%s θ=%s 均值 %+.4f%%' %
            ('不设' if best[1] > 0.5 else '%.1f%%' % (best[1] * 100),
             '%.1f%%' % (best[2] * 100), best[0] * 100))
    del df


def main():
    log('=' * 100)
    log('在线版 · 加成本硬止损 · SL × θ 扫描   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('规则：2h 跌 2%% 且量比>=3 入场；+2%% 止盈 / 跌破 entry-SL 硬止损 / 峰值回落 θ / 最长 8h')
    log('=' * 100)
    for f in (sys.argv[1:] or ['klines_5m_3y.csv']):
        run(f)
    log('')
    log('诚实边界：零滑点；TP 按 high 触价、SL 按 entry*(1-SL) 成交、θ 按 close 成交，')
    log('         均假设能成交（实盘需排队）；同一根内 SL 与 TP 同触时按 SL 保守处理；')
    log('         按币独立结算；含样本内。')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
