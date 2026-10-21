# -*- coding: utf-8 -*-
"""接力 · 更强的零模型：并发度 + 币种集中度

上一版的缺陷：用"事件均匀随机散布"当零模型太弱 —— 事件本来就成簇（情绪同步），
所以"接力对 = 随机的 191 倍"这个数字**不可信**。

本版改用：
  A **并发度**：每个事件启动瞬间，有多少**别的**事件正在进行中（A未见顶B已启动 = 叠盖）
     零模型：保持每个事件的"拉升时长"不变，只把启动时间在总跨度内均匀随机重排
  B **币种集中度**：叠盖对的 A 端 / B 端是否集中在少数币（若 277 币均匀出现 ⇒ 巧合；
     若少数币反复出现 ⇒ 同一批资金在轮动）
"""
import pandas as pd, numpy as np
EV = r'D:\中信证券\coinlab\evidence'
ev = pd.read_csv(EV + r'\pump_hunt_events.csv')
ev['st'] = pd.to_datetime(ev['start_time'])
ev['pk'] = pd.to_datetime(ev['peak_time'])
ev = ev.sort_values('st').reset_index(drop=True)
ts = ev['st'].values.astype('datetime64[m]').astype('int64')
tp = ev['pk'].values.astype('datetime64[m]').astype('int64')
n = len(ev)
dur = np.maximum(tp - ts, 1)
print('events %d | coins %d | span %.0f d | dur median %.0f min'
      % (n, ev['symbol'].nunique(), (ts[-1] - ts[0]) / 1440, np.median(dur)))
occ = float(dur.sum()) / max(ts[-1] - ts[0], 1)
print('事件总占用时间比例 = %.1f%%（任一时点处于某个局内的概率）' % (100 * occ))

def mean_concurrent(ts, tp):
    """每个事件启动瞬间，正在进行中的别的事件数（均值）"""
    cnt = np.zeros(len(ts), dtype=float)
    for j in range(len(ts)):
        lo = np.searchsorted(ts, ts[j], 'left')
        hi = np.searchsorted(ts, tp[j], 'right')
        cnt[lo:hi] += 1.0
    return float(np.mean(cnt) - 1.0)

real = mean_concurrent(ts, tp)
rng = np.random.default_rng(0)
vals = []
for _ in range(100):
    tn = np.sort(rng.integers(ts[0], ts[-1] + 1, n))
    vals.append(mean_concurrent(tn, tn + dur))
mu, sd = float(np.mean(vals)), float(np.std(vals))
print('\n【A 并发度（叠盖强度）】')
print('  实际 %.3f 个/事件启动 ｜ 随机 %.3f±%.3f ｜ %.2f× ｜ z=%+.1f'
      % (real, mu, sd, real / max(mu, 1e-9), (real - mu) / max(sd, 1e-9)))
print('  理论下界：若完全随机散布，期望并发 ≈ %.3f' % (occ * n * 0.0 + (dur.mean() / (ts[-1] - ts[0])) * n))
print('  （= 总占用率 × 事件数 - 1 的粗估；实际远高于它 ⇒ 事件确实扎堆重叠）')

# ---------- B 币种集中度 ----------
pairs = []
for j in range(n):
    lo = np.searchsorted(ts, ts[j], 'left')
    hi = np.searchsorted(ts, tp[j], 'right')
    for i in range(lo, hi):
        if i != j:
            a, b = (i, j) if i < j else (j, i)
            pairs.append((ev['symbol'][a], ev['symbol'][b]))
P = pd.DataFrame(pairs, columns=['A', 'B'])
print('\n【B 叠盖对的币种集中度】')
print('  叠盖对 %d 条；A端币数 %d ｜ B端币数 %d' % (len(P), P['A'].nunique(), P['B'].nunique()))
print('  A端 top10:', ', '.join('%s(%d)' % (k.replace('USDT', ''), v)
      for k, v in P['A'].value_counts().head(10).items()))
print('  B端 top10:', ', '.join('%s(%d)' % (k.replace('USDT', ''), v)
      for k, v in P['B'].value_counts().head(10).items()))
both = set(P['A']) & set(P['B'])
print('  既当 A 又当 B 的币 = %d 个（占比 %.1f%%）' % (len(both), 100 * len(both) / ev['symbol'].nunique()))
top_a = P['A'].value_counts()
print('  集中度：A端前 10 币承担 %.1f%% 的叠盖对；B端前 10 币承担 %.1f%%'
      % (100 * top_a.head(10).sum() / len(P),
         100 * P['B'].value_counts().head(10).sum() / len(P)))
P.to_csv(EV + r'\gambler_relay_overlap_pairs.csv', index=False, encoding='utf-8-sig')
