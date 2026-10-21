# -*- coding: utf-8 -*-
"""bucket_scan.py — 「胜率高但亏」能不能靠事前筛选避开

已知：θ=1% / TP=2% 时 60,294 笔胜率 43.7%、均值 -0.0807%。
本脚本不改出场，只看**入场那一刻能知道的东西**能不能把亏损笔筛掉：
  特征：量比 volr、跌幅深度 ret24、该币近 30 天波动率、UTC 小时
输出：每个特征分 5 桶，各桶笔数/均值/胜率。若某桶稳定为正，就是可用的入场过滤器。
"""
import os
import numpy as np
import pandas as pd
import time

D = r'D:/中信证券/coinlab/data'
OUT = r'D:/中信证券/coinlab/evidence/bucket_scan.txt'
H, COST, W = 96, 0.0004, 48
DIP, VOLR, TP, THETA = -0.02, 3.0, 0.02, 0.01
FNAME = 'klines_5m_3y.csv'

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(D, FNAME),
                     usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
    recs = []
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
        # 该币近 30 天（8640 根 5m）日波动率代理：滚动 std of 5m return × sqrt(288)
        r5 = np.zeros(n); r5[1:] = c[1:] / c[:-1] - 1.0
        sd = pd.Series(r5).rolling(288, min_periods=288).std().to_numpy() * np.sqrt(288)
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
            recs.append((sym, int(t[i]), float(volr[i]), float(ret24[i]),
                         float(sd[i]) if np.isfinite(sd[i]) else np.nan, out))
    del df
    R = pd.DataFrame(recs, columns=['sym', 't', 'volr', 'ret24', 'sd', 'net'])
    log('')
    log('样本 %d 笔  总体均值 %+.4f%%  胜率 %.1f%%' %
        (len(R), R.net.mean() * 100, (R.net > 0).mean() * 100))

    def bucket(col, labels, q=5):
        sub = R.dropna(subset=[col]).copy()
        sub['b'] = pd.qcut(sub[col], q, labels=False, duplicates='drop')
        log('')
        log('--- 按 %s 分桶（%s）---' % (labels, col))
        log('%-14s %8s %11s %9s %8s' % ('桶(区间)', '笔数', '均值/笔', '中位', '胜率'))
        log('-' * 56)
        for b, g in sub.groupby('b'):
            lo = g[col].min(); hi = g[col].max()
            log('%-14s %8d %+10.4f%% %+8.3f%% %7.1f%%' %
                ('[%.4f,%.4f]' % (lo, hi), len(g), g.net.mean() * 100,
                 g.net.median() * 100, (g.net > 0).mean() * 100))

    bucket('volr', '量比')
    bucket('ret24', '跌幅深度')
    bucket('sd', '该币日波动率')
    R['hour'] = pd.to_datetime(R.t, unit='ms', utc=True).dt.hour
    log('')
    log('--- 按 UTC 小时 ---')
    log('%-6s %8s %11s %9s %8s' % ('小时', '笔数', '均值/笔', '中位', '胜率'))
    log('-' * 46)
    for hr, g in R.groupby('hour'):
        if len(g) < 200:
            continue
        log('%-6d %8d %+10.4f%% %+8.3f%% %7.1f%%' %
            (hr, len(g), g.net.mean() * 100, g.net.median() * 100, (g.net > 0).mean() * 100))
    log('')
    log('--- 币级：按该币历史日均笔数（活跃度）分 3 档 ---')
    cnt = R.groupby('sym').size()
    R['act'] = R.sym.map(cnt)
    R['ab'] = pd.qcut(R.act, 3, labels=False)
    for b, g in R.groupby('ab'):
        log('  档%d  笔数 %6d  均值 %+8.4f%%  胜率 %5.1f%%  (币数 %d)' %
            (b, len(g), g.net.mean() * 100, (g.net > 0).mean() * 100, g.sym.nunique()))
    log('')
    log('耗时 %.0fs' % (time.time() - t0))
    log('诚实边界：零滑点、成本 0.04% 单边、按币独立结算、含样本内；分桶为事后划分，')
    log('         某桶为正在样本内不代表样本外成立，需时间切分复核。')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
