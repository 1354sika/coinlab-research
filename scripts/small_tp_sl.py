# -*- coding: utf-8 -*-
"""小止盈 × 小止损 在线网格 —— 直接回答「能不能做到一天 2 次、每次 +2%」。

口径（全部无前视）：
  · 入场：当根 1m 收盘时，ret24 <= -2% 且 volr >= 3（只用当根及之前）
  · 出场：之后每根 1m 的 **收盘价** 判断，收盘 >= entry*(1+TP) 或 <= entry*(1-SL) 才成交
          （不用 high/low 触价，避免「卖在影线」这种不存在的假设）
  · 池 20：同时最多 20 个仓位，每笔占 1/20 资金
  · 成本：往返 0.15%
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'
FN = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
COST = 0.0015
POOL = 20
DROP = 0.02
VOLR = 3.0
MAXH = 1440  # 最长持 24h（1m 数据）

t0 = time.time()
df = pd.read_csv(os.path.join(DATA, FN),
                 usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
df = df.dropna(subset=['open_time'])
df['open_time'] = df['open_time'].astype('int64')
for c in ('open', 'high', 'low', 'close', 'volume'):
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
df = df.dropna()
print('rows %d  %.1fs' % (len(df), time.time() - t0))

# 找入场点
entries = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    c = g['close'].to_numpy()
    v = g['volume'].to_numpy()
    t = g['open_time'].to_numpy()
    h = g['high'].to_numpy()
    l = g['low'].to_numpy()
    n = len(c)
    if n < 3000:
        continue
    vma = pd.Series(v).rolling(1440).mean().to_numpy() * 1440  # 24h 均量
    ret24 = np.full(n, np.nan)
    ret24[1440:] = c[1440:] / c[:-1440] - 1.0
    volr = np.where(vma > 0, v * 1440 / vma, 0.0)
    mask = (ret24 <= -DROP) & (volr >= VOLR)
    idx = np.where(mask)[0]
    for i in idx:
        if i + MAXH >= n:
            continue
        entries.append((int(t[i]), sym, c, i))
print('entries %d  %.1fs' % (len(entries), time.time() - t0))

TPs = [0.010, 0.015, 0.020, 0.025, 0.030]
SLs = [0.010, 0.015, 0.020, 0.025, 0.030]

rows = []
for TP in TPs:
    for SL in SLs:
        trades = []
        for (tms, sym, c, i) in entries:
            ep = c[i]
            up = ep * (1 + TP)
            dn = ep * (1 - SL)
            end = min(i + MAXH, len(c) - 1)
            seg = c[i + 1:end + 1]
            # 找第一次收盘触线
            hit_up = np.where(seg >= up)[0]
            hit_dn = np.where(seg <= dn)[0]
            ju = hit_up[0] if len(hit_up) else 10**9
            jd = hit_dn[0] if len(hit_dn) else 10**9
            j = min(ju, jd)
            if j == 10**9:
                j = len(seg) - 1
                px = seg[j]
                why = 'timeout'
            elif ju <= jd:
                px = up
                why = 'tp'
            else:
                px = dn
                why = 'sl'
            r = px / ep - 1.0 - COST
            trades.append((tms + (j + 1) * 60000, r, why))
        if not trades:
            continue
        tr = pd.DataFrame(trades, columns=['exit_ms', 'r', 'why']).sort_values('exit_ms')
        r = tr['r'].to_numpy()
        # 池 20：按时间顺序，最多 20 并发（简化：等权 r/POOL 累乘）
        eq = 1.0
        pk = 1.0
        mdd = 0.0
        for x in r:
            eq *= (1 + x / POOL)
            if eq > pk:
                pk = eq
            mdd = min(mdd, eq / pk - 1)
        span_d = (tr['exit_ms'].max() - tr['exit_ms'].min()) / 86400000.0
        per_day = len(tr) / span_d if span_d > 0 else 0
        rows.append(dict(TP=TP, SL=SL, n=len(tr), per_day=per_day,
                         win=(r > 0).mean(), mean=r.mean(),
                         tp_rate=(tr['why'] == 'tp').mean(),
                         final=eq, mdd=mdd,
                         span_d=span_d))

R = pd.DataFrame(rows).sort_values('final', ascending=False)
L = []
L.append('小止盈 × 小止损　在线网格（收盘确认，无前视）  file=%s' % FN)
L.append('=' * 108)
L.append('入场 ret24<=-%.0f%% & volr>=%.1f | 成本 %.2f%% 往返 | 池 %d | 最长持 24h' % (
    DROP * 100, VOLR, COST * 100, POOL))
L.append('样本跨度 %.1f 天' % (R['span_d'].iloc[0] if len(R) else 0))
L.append('')
L.append('%-7s %-7s %8s %9s %8s %10s %8s %9s %9s' % (
    'TP', 'SL', '笔数', '笔/天', '胜率', '单笔均值', 'TP占比', '终值', '回撤'))
L.append('-' * 108)
for _, x in R.iterrows():
    L.append('%-7.1f%% %-7.1f%% %8d %9.2f %7.1f%% %9.4f%% %7.1f%% %9.3f %8.1f%%' % (
        x['TP'] * 100, x['SL'] * 100, x['n'], x['per_day'], x['win'] * 100,
        x['mean'] * 100, x['tp_rate'] * 100, x['final'], x['mdd'] * 100))
L.append('')
L.append('--- 目标对照：一天 2 笔、每笔 +2%% 需要什么 ---')
need = 1.02 ** 2  # 一天两次各 +2%
L.append('  若每笔净 +2.00%%、一天 2 笔 => 每天 %.2f%% => 一年 %.3e 倍' % (
    (need - 1) * 100, need ** 365))
best = R.iloc[0] if len(R) else None
if best is not None:
    L.append('  本表最好：TP=%.1f%% SL=%.1f%% 单笔 %+.4f%% 笔/天 %.2f => 每天 %+.4f%% => 一年 %.3fx' % (
        best['TP'] * 100, best['SL'] * 100, best['mean'] * 100, best['per_day'],
        best['mean'] * best['per_day'] * 100,
        (1 + best['mean']) ** (best['per_day'] * 365)))
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘确认，不用 high/low 触价；实盘还需排队与滑点，这里只扣 0.15% 固定成本。')
L.append('2. 池 20 用「等权 r/20 累乘」近似，未模拟同币并发与资金占用上限。')
L.append('3. 单一数据集，含样本内；未做时间切分外推。')
L.append('4. 「一天 2 笔」是笔数统计，实盘信号密度受波动率影响，会时多时少。')

out = os.path.join(EVI, 'small_tp_sl_online.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
