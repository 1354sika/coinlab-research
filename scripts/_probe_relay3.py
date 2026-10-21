# -*- coding: utf-8 -*-
"""接力 · 把「事件扎堆」和「真接力」分开

问题：接力对多，可能只是因为"事件在时间上扎堆"（市场成波）。
     要证明"接力是币之间的真实关联"，必须**保持时间结构不变、只打乱币的身份**。

零模型 = 把 1163 个事件的**币标签随机重排**（时间完全不动）。

比较：
  A 有序币对的**重复出现率**：同一对 (A,B) 出现 ≥2 次的比例
     实际 > 零模型 ⇒ 某些币对**反复接力**（真关联）
  B 自接力（A==B）占比
  C A 端币集合 与 B 端币集合 的重叠度（同一批币还是两批币）
"""
import numpy as np, pandas as pd
from collections import Counter
EV = r'D:\中信证券\coinlab\evidence'
ev = pd.read_csv(EV + r'\pump_hunt_events.csv')
ev['st'] = pd.to_datetime(ev['start_time'])
ev['pk'] = pd.to_datetime(ev['peak_time'])
ev = ev.sort_values('st').reset_index(drop=True)
ts = ev['st'].astype('int64').values // 60000000
tp = ev['pk'].astype('int64').values // 60000000
sym = ev['symbol'].values
n = len(ev)
print('events %d | coins %d' % (n, len(set(sym))))

def pairs_of(ts, tp, sym):
    out = []
    for j in range(len(ts)):
        lo = np.searchsorted(ts, ts[j], 'right')
        hi = np.searchsorted(ts, tp[j], 'right')
        for i in range(lo, hi):
            if i != j:
                out.append((sym[j], sym[i]))
    return out

def stats(prs):
    c = Counter(prs)
    rep = sum(1 for v in c.values() if v > 1) / max(len(c), 1)
    selfr = sum(1 for a, b in prs if a == b) / max(len(prs), 1)
    a_set = set(a for a, b in prs)
    b_set = set(b for a, b in prs)
    both = len(a_set & b_set) / max(len(a_set | b_set), 1)
    return len(prs), len(c), rep, selfr, both

real = pairs_of(ts, tp, sym)
nr, nu, rep, selfr, both = stats(real)
print('\n【实际】叠盖对 %d ｜ 唯一币对 %d ｜ 重复率 %.1f%% ｜ 自接力 %.2f%% ｜ A/B端重叠 %.1f%%'
      % (nr, nu, 100 * rep, 100 * selfr, 100 * both))

rng = np.random.default_rng(0)
R, S, B = [], [], []
for _ in range(30):
    sh = rng.permutation(sym)
    p = pairs_of(ts, tp, sh)
    _, _, r_, s_, b_ = stats(p)
    R.append(r_); S.append(s_); B.append(b_)
print('【零模型（币标签随机重排，时间不动，30 次）】')
print('  重复率 %.1f%%±%.1f ｜ 自接力 %.2f%%±%.2f ｜ A/B端重叠 %.1f%%±%.1f'
      % (100 * np.mean(R), 100 * np.std(R), 100 * np.mean(S), 100 * np.std(S),
         100 * np.mean(B), 100 * np.std(B)))
print('\n判定：')
print('  重复率：实际 %.1f%% vs 随机 %.1f%% ⇒ %s'
      % (100 * rep, 100 * np.mean(R),
         '币对确实反复接力（真关联）' if rep > np.mean(R) else '与随机无异（只是时间扎堆）'))
print('  自接力：实际 %.2f%% vs 随机 %.2f%%' % (100 * selfr, 100 * np.mean(S)))

# 时间间隔的"人尺度"检验
gaps = []
for j in range(n):
    lo = np.searchsorted(ts, ts[j], 'right')
    if lo < n:
        gaps.append(int(ts[lo] - tp[j]))
g = np.array(gaps, float)
print('\n【间隔分布（最近的下一个事件）】n=%d 中位 %.0f 分钟' % (len(g), np.median(g)))
for nm, a, b in (('叠盖(负)', -10**9, 0), ('0~15min', 0, 15), ('15~60min', 15, 60),
                 ('1~4h', 60, 240), ('4~24h', 240, 1440), ('>24h', 1440, 10**9)):
    m = (g > a) & (g <= b)
    print('  %-10s %4d (%.1f%%)' % (nm, int(m.sum()), 100 * m.mean()))
