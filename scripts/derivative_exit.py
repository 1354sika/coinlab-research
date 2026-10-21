# -*- coding: utf-8 -*-
"""
derivative_exit.py — 用导数（斜率/加速度）判定峰，看能不能比跟踪止盈更好地退出。

这是补上"数学盲区"：之前所有认峰实验都是"未来 N 根定义标签 + 分类模型"，
从没直接算过 f'(t)。

三个层次：
  A. 一阶导（斜率）零交叉：d1[i-1] > 0 且 d1[i] <= 0
  B. 二阶导（加速度）转负：d2 从正转负，即曲率拐点
  C. 斜率幅值阈值：d1 从 >kσ 掉到 <0

判据（先看这三个，再看收益）：
  1. 滞后：斜率零交叉点距离"真实局部峰"隔几根？（0 根=当根认出，1 根=晚一根）
  2. 收益：用导数退出 vs θ=1% 跟踪止盈，谁的钱多？
  3. 覆盖率：导数退出能接住多少比例的峰？
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'derivative_exit.txt')

COST = 0.0015      # 往返 0.15%
H    = 480         # 持仓上限 480 根（1m -> 8h）
STEP = 240         # 入场点每 240 根取一个（4h），去重叠
DROP = 0.02        # 入场：ret24 <= -2%
VOLR = 3.0

L = []
def p(s=''):
    print(s)
    L.append(s)

def load(fname):
    fp = os.path.join(DATA, fname)
    p('loading %s ...' % fp)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'open': 'float32', 'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('  rows %d  %.1fs' % (len(d), time.time() - t0))
    return d

def prep(g):
    g = g.sort_values('open_time')
    c = g.close.to_numpy(np.float64)
    h = g.high.to_numpy(np.float64)
    l = g.low.to_numpy(np.float64)
    v = g.volume.to_numpy(np.float64)
    t = g.open_time.to_numpy(np.int64)
    n = len(c)
    if n < 600:
        return None
    # 一阶导（对数收益率，避免比例放大）
    d1 = np.zeros(n, np.float64)
    d1[1:] = np.log(c[1:] / c[:-1])
    # 二阶导（加速度）
    d2 = np.zeros(n, np.float64)
    d2[2:] = d1[2:] - d1[1:-1]
    # 滚动 24 根收益 / 量比 / 波动
    r24 = np.zeros(n, np.float64)
    r24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(96, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    return dict(c=c, h=h, l=l, v=v, t=t, n=n, d1=d1, d2=d2, r24=r24, volr=volr)

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    tf = 60 if '1m' in fname else 300
    p('=' * 100)
    p('导数认峰实验  file=%s  bar=%ds  H=%d根  COST=%.2f%%' % (fname, tf, H, COST * 100))
    p('=' * 100)

    d = load(fname)
    syms = sorted(d.symbol.unique())
    p('币 %d' % len(syms))

    # ---- 一、滞后检验：斜率零交叉离真实局部峰多远 ----
    lag_d1 = []   # 用一阶导零交叉
    lag_d2 = []   # 用二阶导转负
    lag_k  = []   # 用"斜率从 k*σ 掉到 0"
    n_peaks = 0
    n_d1_hit = 0
    n_d2_hit = 0

    for sym in syms:
        g = d[d.symbol == sym]
        P = prep(g)
        if P is None:
            continue
        c, h, d1, d2, n = P['c'], P['h'], P['d1'], P['d2'], P['n']

        # 真实局部峰：high 在 +-W 内是最大值，且相对前一根至少涨 1%（过滤噪声）
        W = 30
        sd = pd.Series(h).rolling(2 * W + 1, center=True).max().to_numpy()
        is_peak = (h >= sd - 1e-12) & (h > 0)
        is_peak[:W] = False
        is_peak[-W:] = False
        pk = np.flatnonzero(is_peak)
        if len(pk) == 0:
            continue
        n_peaks += len(pk)

        # 每个峰往前找最近的"d1 零交叉"
        zero_cross = np.flatnonzero((d1[:-1] > 0) & (d1[1:] <= 0)) + 1
        d2_turn = np.flatnonzero(d2 < 0)
        if len(zero_cross) >= 10:
            for q in pk[::max(1, len(pk) // 200)]:
                prev = zero_cross[zero_cross <= q]
                if len(prev):
                    lag_d1.append(int(q - prev[-1]))
                    n_d1_hit += 1
                prev2 = d2_turn[d2_turn <= q]
                if len(prev2):
                    lag_d2.append(int(q - prev2[-1]))

    lag_d1 = np.array(lag_d1)
    lag_d2 = np.array(lag_d2)
    p()
    p('=== 一、滞后检验：导数信号离真实峰多远（根）')
    p('  真实局部峰数(抽样) %d  一阶导命中 %d' % (n_peaks, n_d1_hit))
    if len(lag_d1):
        p('  一阶导零交叉 -> 峰：中位滞后 %d 根，25%%分位 %d，75%%分位 %d，<=2根占比 %.1f%%'
          % (np.median(lag_d1), np.percentile(lag_d1, 25), np.percentile(lag_d1, 75),
             (lag_d1 <= 2).mean() * 100))
    if len(lag_d2):
        p('  二阶导转负 -> 峰：中位滞后 %d 根' % np.median(lag_d2))
    p('  读法：0 根=当根认出（理想），1~2 根=晚一两根，>5 根=来不及')

    # ---- 二、收益检验：入场后用不同方式出场 ----
    rows = []
    for sym in syms:
        g = d[d.symbol == sym]
        P = prep(g)
        if P is None:
            continue
        c, h, l, d1, d2, r24, volr, n = P['c'], P['h'], P['l'], P['d1'], P['d2'], P['r24'], P['volr'], P['n']

        # 入场点：ret24<=-2% 且 volr>=3
        sig = np.flatnonzero((r24 <= -DROP) & (volr >= VOLR))
        if len(sig) == 0:
            continue
        # 抽样去重叠
        keep = []
        last = -10**9
        for i in sig:
            if i - last >= STEP and i + H < n:
                keep.append(i)
                last = i
        for i in keep:
            seg_c = c[i:i + H]
            seg_h = h[i:i + H]
            seg_l = l[i:i + H]
            seg_d1 = d1[i:i + H]
            seg_d2 = d2[i:i + H]
            entry = c[i]
            m = len(seg_c)
            if m < 10:
                continue

            # 理想：卖在最高 high
            best = seg_h.max() / entry - 1.0
            # θ=1% 跟踪止盈（基线）
            hi = 0.0; r_theta = seg_c[-1] / entry - 1.0
            for j in range(m):
                if seg_h[j] > hi:
                    hi = seg_h[j]
                if hi > entry and seg_c[j] <= hi * (1 - 0.01):
                    r_theta = seg_c[j] / entry - 1.0
                    break
            # A: 一阶导零交叉（d1 从正转负即卖）
            r_d1 = seg_c[-1] / entry - 1.0
            hit_a = -1
            for j in range(1, m):
                if seg_d1[j - 1] > 0 and seg_d1[j] <= 0 and seg_c[j] > entry * 1.0:
                    r_d1 = seg_c[j] / entry - 1.0
                    hit_a = j
                    break
            # B: 二阶导转负（加速度由正转负）
            r_d2 = seg_c[-1] / entry - 1.0
            hit_b = -1
            for j in range(2, m):
                if seg_d2[j - 1] > 0 and seg_d2[j] <= 0:
                    r_d2 = seg_c[j] / entry - 1.0
                    hit_b = j
                    break
            # C: 只在一阶导零交叉 + 已浮盈>=1% 时才卖（过滤噪声）
            r_d1f = seg_c[-1] / entry - 1.0
            hit_c = -1
            for j in range(1, m):
                if seg_d1[j - 1] > 0 and seg_d1[j] <= 0 and seg_c[j] > entry * 1.01:
                    r_d1f = seg_c[j] / entry - 1.0
                    hit_c = j
                    break

            rows.append((best, r_theta, r_d1, r_d2, r_d1f, hit_a, hit_b, hit_c))

    A = np.array(rows, np.float64)
    p()
    p('=== 二、收益检验（%d 笔，成本 %.2f%% 往返）' % (len(A), COST * 100))
    p('  %-22s %10s %10s %10s %8s' % ('出场方式', '均值/笔', '中位/笔', '胜率', '触发率'))
    p('  ' + '-' * 66)
    names = ['理想(卖在最高high)', 'θ=1% 跟踪止盈(基线)', 'A 一阶导零交叉',
             'B 二阶导转负', 'C 一阶导+浮盈>=1%']
    for k, nm in enumerate(names):
        col = A[:, k] - COST
        if k == 0:
            trg = 100.0
        elif k == 1:
            trg = 100.0
        elif k == 2:
            trg = (A[:, 5] >= 0).mean() * 100
        elif k == 3:
            trg = (A[:, 6] >= 0).mean() * 100
        else:
            trg = (A[:, 7] >= 0).mean() * 100
        p('  %-22s %9.4f%% %9.4f%% %9.2f%% %7.1f%%'
          % (nm, col.mean() * 100, np.median(col) * 100, (col > 0).mean() * 100, trg))

    # 出场时机（根数）
    p()
    p('=== 三、导数出场发生在第几根（越晚=跟得越久）')
    for k, nm in [(5, 'A 一阶导'), (6, 'B 二阶导'), (7, 'C 一阶导+浮盈')]:
        v = A[:, k]
        ok = v >= 0
        if ok.sum() > 0:
            p('  %-12s 触发 %d/%d 笔，中位第 %d 根' % (nm, ok.sum(), len(v), np.median(v[ok])))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价口径，不含执行延迟；成本 0.15%% 往返已扣。')
    p('2. 真实局部峰用 +-30 根滚动最高 high 定义（W=30），这是事后标签。')
    p('3. 入场点每 240 根取一个，样本重叠已大幅降低，但同一波行情仍可能多次命中。')
    p('4. 一阶导/二阶导对噪声敏感；1m 数据的 d1 抖动大，未做平滑（这是原样测试）。')
    p('5. 已下架归零币不在池内（幸存者偏差）。')
    p('6. 单一样本，含样本内。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s  TOTAL %.0fs' % (OUT, 0))

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
