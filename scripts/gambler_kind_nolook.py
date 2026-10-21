# -*- coding: utf-8 -*-
"""gambler_kind_nolook.py — 消除前视的「群体 vs 个体」检验（v1 无前视版）

旧版 two-kinds 的前视：用事件表 start_time 当入场点 —— 那是事后回溯定义的。
本版：每一刻只用「当时可见」的信息。
  群体度 n_up[t] = 过去 24 根(2h) 内「量比≥2 且 单根涨≥3%」的全市场币数，再取 24 根均值
  入场：币 i 在 t 满足「过去 12 根(1h) 涨幅 ≥5%」且「当根量比 ≥2」
  入场价：t 收盘价；出场：+50% 止盈 / −15% 止损 / 24h 超时
  分组：按 n_up[t] 的 33/67 分位 → 个体(低) / 中间 / 群体(高)
  对照：随机时刻入场（同出场规则）
"""
import os, gc, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
t0 = time.time()
df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'])
df = df.sort_values(['symbol', 'open_time'])
CL = df.pivot(index='open_time', columns='symbol', values='close').astype('float32').values
HI = df.pivot(index='open_time', columns='symbol', values='high').astype('float32').values
LO = df.pivot(index='open_time', columns='symbol', values='low').astype('float32').values
QV = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32').values
del df; gc.collect()
T, N = CL.shape
print('matrix %d x %d  %.0fs' % (T, N, time.time() - t0), flush=True)

# ===== 只用到「过去」的特征 =====
ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CL[1:] / CL[:-1] - 1.0
med = pd.DataFrame(QV).rolling(288, min_periods=60).median().values
med = np.where(med > 0, med, np.nan)
volr = (QV / med).astype('float32')
r12 = np.full((T, N), np.nan, np.float32); r12[12:] = CL[12:] / CL[:-12] - 1.0
print('特征算完 %.0fs' % (time.time() - t0), flush=True)

OK = np.isfinite(volr) & np.isfinite(ret1)
M = OK & (volr >= 2) & (ret1 >= 0.03)
n_up = M.sum(axis=1).astype('float32')
n_up_s = pd.Series(n_up).rolling(24, min_periods=6).mean().values
print('n_up(2h 异动币数) 中位 %.1f | p33 %.1f | p67 %.1f | p90 %.1f | max %.0f'
      % (np.nanmedian(n_up_s), np.nanpercentile(n_up_s, 33), np.nanpercentile(n_up_s, 67),
         np.nanpercentile(n_up_s, 90), np.nanmax(n_up_s)), flush=True)

entry = np.isfinite(r12) & np.isfinite(volr) & (r12 >= 0.05) & (volr >= 2)
print('入场点总数 %d' % int(entry.sum()), flush=True)

K = 288          # 24h 持有窗口
def sim(mask, nup_arr, stride=6, cap=300000):
    rs, ns = [], []
    idxs = np.argwhere(mask)
    idxs = idxs[::stride]
    if len(idxs) > cap:
        sel = np.random.RandomState(0).choice(len(idxs), cap, replace=False)
        idxs = idxs[sel]
    for t, j in idxs:
        if t + K >= T:
            continue
        e = CL[t, j]
        if not np.isfinite(e) or e <= 0:
            continue
        sh = HI[t + 1:t + 1 + K, j]; sl = LO[t + 1:t + 1 + K, j]
        tp = np.where(sh >= e * 1.5)[0]; slx = np.where(sl <= e * 0.85)[0]
        i_tp = tp[0] if len(tp) else 10 ** 9
        i_sl = slx[0] if len(slx) else 10 ** 9
        if i_tp == i_sl == 10 ** 9:
            r = CL[t + K, j] / e - 1.0
        elif i_tp < i_sl:
            r = 0.50
        else:
            r = -0.15
        rs.append(r); ns.append(nup_arr[t] if np.isfinite(nup_arr[t]) else np.nan)
    return np.array(rs), np.array(ns)

r, nu = sim(entry, n_up_s)
print('\n' + '=' * 92)
print('【无前视：入场 = 过去1h涨≥5%% 且 量比≥2】（笔数 %d，抽样 1/%d）' % (len(r), 6))
print('=' * 92)
ok = np.isfinite(nu)
q1, q2 = np.nanpercentile(nu[ok], [33, 67])
for lbl, m in [('个体型 n_up低', nu <= q1), ('中间', (nu > q1) & (nu <= q2)), ('群体型 n_up高', nu > q2)]:
    x = r[m & ok]
    if not len(x):
        continue
    print('  %-14s n=%7d  期望净 %+7.2f%%  中位 %+7.2f%%  胜率 %5.1f%%  ≥+50%%占 %5.1f%%  ≤-15%%占 %5.1f%%'
          % (lbl, len(x), 100 * x.mean(), 100 * np.median(x), 100 * (x > 0).mean(),
             100 * (x >= 0.5).mean(), 100 * (x <= -0.15).mean()))
print('  %-14s n=%7d  期望净 %+7.2f%%  中位 %+7.2f%%  胜率 %5.1f%%' % ('【全部】', len(r[ok]), 100 * r[ok].mean(), 100 * np.median(r[ok]), 100 * (r[ok] > 0).mean()))

# 对照：随机时刻
print('\n' + '=' * 92)
print('【对照】随机时刻入场（同样出场规则）')
print('=' * 92)
rnd = np.zeros_like(entry)
rs = np.random.RandomState(1)
rows = rs.randint(K, T - K, size=min(400000, T * N // 400))
cols = rs.randint(0, N, size=len(rows))
rnd[rows, cols] = True
r2, nu2 = sim(rnd, n_up_s, stride=1, cap=300000)
ok2 = np.isfinite(nu2)
print('  n=%7d  期望净 %+7.2f%%  中位 %+7.2f%%  胜率 %5.1f%%'
      % (len(r2), 100 * r2.mean(), 100 * np.median(r2), 100 * (r2 > 0).mean()))

pd.DataFrame({'net': r, 'n_up': nu}).to_csv(os.path.join(E, 'gambler_kind_nolook.csv'), index=False, encoding='utf-8-sig')
print('\nTOTAL %.0fs' % (time.time() - t0))
