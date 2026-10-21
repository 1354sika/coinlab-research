# -*- coding: utf-8 -*-
"""
问题：模型能不能同时管入场和出场？
- 入场：固定一个"模型式"信号（ret24<=-2% 且 volr>=3.0，即 exit_peak 用的那套）
- 出场：A 固定持 96 根 / B 单笔跟踪止盈 θ=1% / C 不加单笔出场、只用组合层 L3（净值到 1.5x 后峰值回撤 5% 清）
- 对比：同一入场，四种组合的逐笔期望与组合终值
输出 coinlab/evidence/entry_exit_combo.txt
"""
import numpy as np, pandas as pd, time

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/entry_exit_combo.txt'
H = 96; STEP = 24; COST = 0.0008; THETA = 0.01

D = pd.read_csv(CSV, usecols=['open_time','symbol','open','high','low','close','volume'])
print('rows', len(D), '%.1fs' % (time.time()-t0), flush=True)
for k in ('open','high','low','close','volume'):
    D[k] = D[k].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

rows = []
for sym, g in D.groupby('symbol', sort=False):
    o = g.open.to_numpy(); h = g.high.to_numpy(); c = g.close.to_numpy(); v = g.volume.to_numpy()
    t = g.open_time.to_numpy(); n = len(o)
    if n < 300 + H + 2:
        continue
    c_ret = pd.Series(c).pct_change(288).to_numpy()
    v_ma = pd.Series(v).rolling(288, min_periods=288).mean().to_numpy()
    volr = np.where(v_ma > 0, v / v_ma, 1.0)
    ok = np.isfinite(c_ret) & (c_ret <= -0.02) & (volr >= 3.0)
    idx = np.flatnonzero(ok)
    idx = idx[(idx >= 300) & (idx + H + 1 < n)]
    if len(idx) == 0:
        continue
    idx = idx[::STEP]
    for e in idx:
        ep = float(o[e+1])
        if ep <= 0:
            continue
        hi = h[e+1:e+1+H]; cl = float(c[e+1+H-1])
        rA = cl / ep - 1 - COST
        # B: 单笔跟踪止盈 θ
        peak = np.maximum.accumulate(hi)
        prev_peak = np.concatenate(([hi[0]], peak[:-1]))
        thr = prev_peak * (1 - THETA)
        hit = np.flatnonzero(hi <= thr)
        if len(hit) > 0 and hit[0] > 0 and prev_peak[hit[0]] > ep:
            rB = float(prev_peak[hit[0]] * (1 - THETA) / ep - 1) - COST
        else:
            rB = cl / ep - 1 - COST
        rows.append((int(t[e]), sym, rA, rB))
R = pd.DataFrame(rows, columns=['t','sym','A','B']).sort_values('t').reset_index(drop=True)
print('笔数', len(R), '%.1fs' % (time.time()-t0), flush=True)


def l3(eq, f=0.05, target=1.5):
    """净值到 target 后，从峰值回撤 f 清仓；返回(锁定值, 是否触发)"""
    if not (eq >= target).any():
        return float(eq[-1]), False
    j0 = int(np.argmax(eq >= target))
    peak = eq[j0]
    for k in range(j0, len(eq)):
        peak = max(peak, eq[k])
        if eq[k] <= peak * (1 - f):
            return float(eq[k]), True
    return float(eq[-1]), False


L = []
L.append('入场+出场能否同一个模型做？—— 同一入场、四种出场组合   %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('入场信号：ret24<=-2%% 且 量比>=3.0（模型式筛子；每 24 根取一个，无前视）')
L.append('出场：A 持满 96 根收盘 / B 单笔跟踪止盈 θ=1%% / C 单笔不做，改用组合层 L3（净值到 1.5x 后峰值回撤 5%% 清）')
L.append('成本：往返 0.08%%；数据 60 币 / 3 年 / 5m；笔数 %d' % len(R))
L.append('')
L.append('=== 一、逐笔')
for col, name in (('A','A 持满8h'), ('B','B 跟踪止盈θ=1%%')):
    a = R[col].to_numpy()
    L.append('  %-16s 均值 %+7.3f%%  中位 %+7.3f%%  胜率 %5.1f%%' % (
        name, a.mean()*100, np.median(a)*100, (a>0).mean()*100))
L.append('')
L.append('=== 二、组合净值（1.0 起，逐笔满仓复利）')
for col, name in (('A','A 持满8h'), ('B','B 跟踪止盈θ=1%%')):
    a = np.clip(R[col].to_numpy(), -0.99, None)
    eq = np.cumprod(1 + a)
    v, hitL3 = l3(eq.copy())
    L.append('  %-16s 峰值 %8.3fx  终值 %8.4fx  |  +L3(回撤5%%)：%s，锁定 %8.3fx' % (
        name, eq.max(), eq[-1], '触发' if hitL3 else '未触发', v))
L.append('')
L.append('=== 三、读法')
L.append('  若 B 相对 A 明显改善 => 单笔出场模型有用；若 B≈A => 单笔出场学不到东西（与 gambler_exit 一致）。')
L.append('  若 A+L3、B+L3 都把终值从 ~0 抬到 >1 => 出场该放在组合层，不该放在单笔层。')
L.append('')
L.append('诚实边界：')
L.append('  1. 入场信号是"筛子"不是训练出来的模型；真实 A2/V20 入场点不在这张表里。')
L.append('  2. 零滑点；B 的跟踪止盈按 peak*(1-θ) 成交，实盘需该价位有买单。')
L.append('  3. 满仓单序列复利，未模拟多币并发持仓；样本重叠（每 24 根取一个）。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
