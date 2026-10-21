# -*- coding: utf-8 -*-
"""compare_online_vs_lookahead.py — 同一批入场点，两种出场实现对比

背景：best_curve_REAL.txt 报出 +1.0822%/笔、胜率 67.14%。
      但它的出场实现（analyze_best_curve.py）是：
          k = theta_k(P, i)                      # 先算 θ 触发点
          for j in range(H):
              if r[j] >= 0.02: k = j; break      # 8h 内任何一根到 +2% 就覆盖 k
      也就是说：θ 早就触发了，只要后面某根 close 到过 +2%，就按那根结算。
      —— 这用了未来的价格，实盘不可能。

本脚本做两件事：
  A 复刻 best_curve（含上述覆盖逻辑）
  B 同一批入场点，改成「谁先触发谁生效」的在线版
两者唯一差别就是这段前视。差值 = 前视贡献了多少收益。

诚实边界：零滑点、成本 0.04% 单边；仍是按币独立结算；全历史含样本内。
"""
import numpy as np
import pandas as pd
import time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/online_vs_lookahead.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR, TP, THETA = -0.02, 3.0, 0.02, 0.01

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

t0 = time.time()
log('=' * 96)
log('在线 vs 前视 · 同一批入场点、两种出场实现   %s' % time.strftime('%Y-%m-%d %H:%M'))
log('=' * 96)

df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
paths = {}; entries = []
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
    paths[sym] = dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)
    for i in range(W + 300, n - H - 1, GAP):
        if ret24[i] <= DIP and volr[i] >= VOLR:
            entries.append((int(t[i]), sym, i))
del df
entries.sort()
log('入场点 %d   耗时 %.0fs' % (len(entries), time.time() - t0))


def theta_k(P, i, th=0.01):
    """复刻 analyze_best_curve.theta_k"""
    h, l = P['h'], P['l']
    eh = h[i:i + H] / P['c'][i] - 1.0
    el = l[i:i + H] / P['c'][i] - 1.0
    hi = 0.0; s = H - 1
    for j in range(H):
        if eh[j] > hi:
            hi = eh[j]
        if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
            s = j
            break
    return s


def exit_online(P, i):
    """在线：逐根推进，先判 TP 再判 θ，谁先触发谁生效"""
    c, h, l = P['c'], P['h'], P['l']
    entry = float(c[i]); peak = entry
    for j in range(1, H):
        k = i + j
        if float(h[k]) > peak:
            peak = float(h[k])
        if float(c[k]) >= entry * (1 + TP):
            return j, 'tp'
        if float(l[k]) <= peak * (1 - THETA):
            return j, 'theta'
    return H - 1, 'timeout'


rows_a = []; rows_b = []
for (tms, sym, i) in entries:
    P = paths[sym]
    c = P['c']
    r = c[i:i + H] / c[i] - 1.0
    # A 前视版
    k = theta_k(P, i)
    ka = k
    for j in range(H):
        if r[j] >= 0.02:
            ka = j; break
    rows_a.append((tms, float(r[ka]) - COST, 'both', ka))
    # B 在线版
    jb, why = exit_online(P, i)
    rows_b.append((tms, float(r[jb]) - COST, why, jb))


def stat(rows, name):
    nets = np.array([x[1] for x in rows], dtype=np.float64)
    n = len(nets)
    mean = nets.mean(); med = float(np.median(nets))
    win = float((nets > 0).mean())
    eq = 1.0; peak = 1.0; mdd = 0.0; to15 = None
    rows_s = sorted(rows, key=lambda x: x[0])
    for idx, x in enumerate(rows_s, 1):
        eq *= (1 + x[1])
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        if to15 is None and eq >= 1.5:
            to15 = idx
    log('%-10s 笔数 %5d  均值 %+8.4f%%  中位 %+7.4f%%  胜率 %5.1f%%  '
        '复利终值 %s  最大回撤 %6.2f%%  到1.5x %s'
        % (name, n, mean * 100, med * 100, win * 100,
           ('%.2fx' % eq) if eq < 1e6 else ('%.2e' % eq), mdd * 100,
           ('%d 笔' % to15) if to15 else '未达'))
    return dict(n=n, mean=mean, med=med, win=win, eq=eq, mdd=mdd, to15=to15)


log('')
log('=' * 96)
sa = stat(rows_a, 'A 前视版')
sb = stat(rows_b, 'B 在线版')
log('=' * 96)
log('前视贡献：均值 %+.4fpp（%.4f%% → %.4f%%）  胜率 %+.1fpp  到1.5x %s → %s'
    % ((sa['mean'] - sb['mean']) * 100, sa['mean'] * 100, sb['mean'] * 100,
       (sa['win'] - sb['win']) * 100,
       ('%d 笔' % sa['to15']) if sa['to15'] else '未达',
       ('%d 笔' % sb['to15']) if sb['to15'] else '未达'))
log('')
cnt = {}
for x in rows_b:
    cnt[x[2]] = cnt.get(x[2], 0) + 1
log('B 在线版出场构成：' + '  '.join('%s=%d(%.1f%%)' % (k, v, v / len(rows_b) * 100)
                                 for k, v in sorted(cnt.items(), key=lambda z: -z[1])))
log('')
log('诚实边界：零滑点、成本 0.04%% 单边；按币独立结算；全历史含样本内；')
log('         A 版本与 analyze_best_curve.py 逐字同逻辑（已核对 theta_k 与覆盖顺序）。')
txt = '\n'.join(L)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(txt + '\n')
log('已写 %s  TOTAL %.0fs' % (OUT, time.time() - t0))
