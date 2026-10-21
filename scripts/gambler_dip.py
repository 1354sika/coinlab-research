# -*- coding: utf-8 -*-
"""gambler_dip.py — 策略实测：「低谷买入 + 到达期望收益就撤」（用户口径）

【用户思路】
    不追大局，就捕捉细微波动：在低谷买入，到达期望收益就撤（不贪）。

【本脚本把它拆成可检验的口径】
    入场（t 时刻全部可观测，无前视）：
        ① 位置低      pos <= P     pos = (close - 近288根最低)/(近288根最高 - 近288根最低)
        ② 未拉完      r12 <= 30%   （避免追在半山腰）
        ③ 量能        volr24 >= V  （有资金在动；V 可设 0 = 不看）
    出场（先到先出）：
        ① 止盈 +X%
        ② 止损 -Y%（Y=0 表示不设止损）
        ③ 超时 T 小时
    成本：maker 双边 0.04%

【对照组】
    Z_rand   随机入场（同样出场规则）—— 基准
    Z_nofix  「低谷」入场但不止盈（一直拿到超时）—— 检验「止盈」本身有没有用
    Z_pump   「已大涨」入场（over>=50%）—— 反例

【产出】
    butterfly/dip_all.csv       全部 (P, X, Y, T) 组合的期望/中位/胜率/笔数
    butterfly/dip_summary.txt   汇总 + 判读

跑法：
    cd /d D:\\中信证券\\coinlab\\scripts && python -X utf8 gambler_dip.py
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D   = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')

WDAY = 288            # 一根 5m = 5 分钟；288 根 = 24h
BARS_PER_H = 12
MAKER = 0.0004        # 双边 maker 成本（0.04%）

# 参数网格
PS  = [0.10, 0.20, 0.30]          # pos 阈值（越低越"谷底"）
XS  = [2, 3, 5, 8, 15]            # 止盈 %
YS  = [0, 3, 5]                   # 止损 %（0 = 不设）
TS  = [4, 12, 24]                 # 超时 小时
R12_MAX = 0.30                    # 入场附加条件：r12 <= 30%

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_dip · 低谷买入 + 到达期望收益就撤（用户口径）')
log('=' * 100)

# ================= 1. 读数据 =================
RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym,
                    g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'),
                    g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0:.0f}s')

# ================= 2. 逐币算特征 + 模拟 =================
def simulate_entry(hi, lo, cl, entry_idx, X, Y, T, max_hold):
    """从 entry_idx 次日开始模拟：先触及 +X% 还是 -Y% 还是超时。返回净收益（已扣成本）。"""
    p0 = cl[entry_idx]
    up = p0 * (1 + X / 100.0)
    dn = p0 * (1 - Y / 100.0) if Y > 0 else -np.inf
    end = min(len(cl), entry_idx + max_hold + 1)
    for j in range(entry_idx + 1, end):
        if hi[j] >= up:
            return X / 100.0 - 2 * MAKER          # 止盈
        if lo[j] <= dn:
            return -Y / 100.0 - 2 * MAKER         # 止损
    # 超时：按最后一根收盘价
    return cl[end-1] / p0 - 1.0 - 2 * MAKER

rows = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 6:
        continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    pos = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / c[:-12] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values

    valid = np.isfinite(pos) & np.isfinite(r12) & np.isfinite(volr24)
    # 为避免样本爆炸：每 24 根（2 小时）最多取一个入场点
    cand_all = np.flatnonzero(valid)
    for P in PS:
        cand = cand_all[(pos[cand_all] <= P) & (r12[cand_all] <= R12_MAX)]
        if len(cand) == 0:
            continue
        cand = cand[::24] if len(cand) > 400 else cand          # 抽样限流
        for X in XS:
            for Y in YS:
                for T in TS:
                    mh = int(T * BARS_PER_H)
                    rets = np.array([simulate_entry(h, l, c, i, X, Y, T, mh) for i in cand])
                    if len(rets) < 30:
                        continue
                    rows.append((sym, P, X, Y, T, len(rets),
                                 float(rets.mean()), float(np.median(rets)),
                                 float((rets > 0).mean()),
                                 float((rets >= X/100.0 - 2*MAKER - 1e-9).mean())))
    del hs, ls, qs; gc.collect()

log(f'模拟完成 {len(rows)} 行   耗时 {time.time()-t0:.0f}s')

df = pd.DataFrame(rows, columns=['sym', 'P', 'X', 'Y', 'T', 'n',
                                 'exp_net', 'med_net', 'winrate', 'tp_rate'])
agg = (df.groupby(['P', 'X', 'Y', 'T'])
         .apply(lambda g: pd.Series({
             'n': int(g['n'].sum()),
             'exp_net': float((g['exp_net'] * g['n']).sum() / g['n'].sum()),
             'med_net': float(g['med_net'].median()),
             'winrate': float((g['winrate'] * g['n']).sum() / g['n'].sum()),
             'tp_rate': float((g['tp_rate'] * g['n']).sum() / g['n'].sum()),
         }), include_groups=False)
         .reset_index())
agg = agg[agg['n'] >= 200].sort_values('exp_net', ascending=False)
agg.to_csv(os.path.join(OUT, 'dip_all.csv'), index=False, encoding='utf-8-sig')

log('=' * 100)
log('【结果】按期望净收益排序 top-25（已扣 maker 0.04%）')
log(f"{'pos<=':>6} {'止盈%':>6} {'止损%':>6} {'超时h':>6} {'笔数':>8} {'期望净%':>9} {'中位%':>8} {'胜率%':>7} {'止盈占比%':>9}")
for _, r in agg.head(25).iterrows():
    log(f"{r['P']:>6.2f} {r['X']:>6.0f} {r['Y']:>6.0f} {r['T']:>6.0f} {int(r['n']):>8,} "
        f"{r['exp_net']*100:>9.3f} {r['med_net']*100:>8.3f} {r['winrate']*100:>7.1f} {r['tp_rate']*100:>9.1f}")

log('=' * 100)
log('【判读】')
log('  · 期望净 > 0 且 笔数足够 ⇒ 「低谷买入 + 到点就撤」可交易')
log('  · 若「不止损」那一档期望最高 ⇒ 小波动里止损是负贡献（被打掉的比救回来的多）')
log('  · 若止盈越小期望越高 ⇒ 说明小波动本来就长不大，"到点就撤"是对的')
log('  · 用 random 对照（另见 butterfly/dip_rand.txt）比较才知道是不是"低谷"本身在起作用')

log(f'TOTAL {time.time()-t0:.0f}s')
with open(os.path.join(OUT, 'dip_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log('报告已写 ' + os.path.join(OUT, 'dip_summary.txt'))
