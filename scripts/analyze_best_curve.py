# -*- coding: utf-8 -*-
"""analyze_best_curve.py — 全项目最好的那条规则，补上资金曲线与最大回撤

已知（tp_verify_REAL.txt）：入场 ret24<=-2% & volr>=3.0，出场「首次 >=+2% 卖，否则 θ=1% 跟踪止盈兜底」
  → 单笔均值 +1.064%，中位 1.964%，9/9 季正，去重 72h 后 +1.041%（1866 笔），59/60 币为正。
这是全项目样本外最强的一条，但从未算过资金曲线与最大回撤（用户硬判据 <20% 未验证）。

本脚本只做一件事：按时间顺序逐笔复利，报
  · 终值、最大回撤、到 1.5x 需要多少笔、年化笔数
  · 对照：同一批笔随机打乱顺序（看路径依赖有多强）
  · 对照：不复利（每笔固定本金）的均值/累计

诚实边界：顺序复利 = 上界（假设前一笔记完才开下一笔，实际多币并发）；
          零滑点、成本 0.04% 单边、TP 用 high/close 触价假设；未计容量与成交率。
"""
import numpy as np
import pandas as pd
import time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/best_curve_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)

t0 = time.time()
log('=' * 96)
log(f'best_curve · 全项目最强规则的资金曲线与回撤   {time.strftime("%Y-%m-%d %H:%M")}')
log('=' * 96)
log('规则：入场 ret24<=-2% 且 volr>=3.0；出场 首次>=+2% 卖，否则 θ=1% 跟踪止盈兜底；成本 0.04% 单边')
log('')

df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H:
        return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None:
        continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
log(f'入场点 {len(entries)}  耗时 {time.time()-t0:.0f}s')


def theta_k(P, i, th=0.01):
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


rows = []
for (tms, sym, i) in entries:
    P = paths[sym]; c = P['c']
    r = c[i:i + H] / c[i] - 1.0
    k = theta_k(P, i)
    for j in range(H):
        if r[j] >= 0.02:
            k = j
            break
    rows.append((tms, sym, float(r[k]) - COST, int(k)))
del paths
R = pd.DataFrame(rows, columns=['t', 'sym', 'net', 'hold'])
log(f'结算完成 {len(R)} 笔  耗时 {time.time()-t0:.0f}s')
log('')

net = R.net.to_numpy()
log('=' * 96)
log('【一】单笔分布（复算，应与 tp_verify 一致）')
log('=' * 96)
log(f'均值 {net.mean():.4%}  中位 {np.median(net):.4%}  胜率 {(net>0).mean():.2%}  '
    f'p5 {np.percentile(net,5):.2%}  p95 {np.percentile(net,95):.2%}')
hold_min = R.hold.to_numpy() * 5.0
log(f'持有时间（分钟）：均值 {hold_min.mean():.1f}  中位 {np.median(hold_min):.0f}  '
    f'p90 {np.percentile(hold_min,90):.0f}  最长 {hold_min.max():.0f}')
log(f'单账号顺序（一次一笔）：一天最多 {1440/max(np.median(hold_min),1):.1f} 笔（按中位持有时间算）')
log(f'                                    一年最多 {365*1440/hold_min.mean():.0f} 笔（按均值算）')
log('')

log('=' * 96)
log('【二】顺序复利（按时间）：终值、最大回撤、到 1.5x 需要多少笔')
log('=' * 96)
eq = np.cumprod(1 + net)
peak = np.maximum.accumulate(eq)
dd = eq / peak - 1.0
hit = np.argmax(eq >= 1.5) if (eq >= 1.5).any() else None
log(f'终值            {eq[-1]:.4f}x')
log(f'最大回撤        {dd.min():.2%}')
log(f'到 1.5x 需要    {hit+1 if hit is not None else "未达"} 笔')
log(f'峰值            {peak[-1]:.4f}x')
# 年化
span_y = (R.t.iloc[-1] - R.t.iloc[0]) / (365.25 * 86400 * 1000)
log(f'时间跨度        {span_y:.2f} 年  笔/年 {len(net)/span_y:.0f}')
if eq[-1] > 0:
    log(f'年化（复利）    {(eq[-1] ** (1/span_y) - 1) if eq[-1] > 0 else float("nan"):.2%}')
log('')

log('=' * 96)
log('【三】路径依赖对照：同样这批笔，随机打乱顺序 200 次')
log('=' * 96)
rng = np.random.default_rng(7)
ends, dds, hits = [], [], []
for _ in range(200):
    p = rng.permutation(net)
    e = np.cumprod(1 + p)
    pk = np.maximum.accumulate(e)
    ends.append(e[-1]); dds.append((e / pk - 1.0).min())
    hh = np.argmax(e >= 1.5) if (e >= 1.5).any() else -1
    hits.append(hh + 1 if hh >= 0 else None)
log(f'终值      中位 {np.median(ends):.4f}x  p5 {np.percentile(ends,5):.4f}x  p95 {np.percentile(ends,95):.4f}x')
log(f'最大回撤  中位 {np.median(dds):.2%}  最好 {max(dds):.2%}  最差 {min(dds):.2%}')
ok = [h for h in hits if h]
log(f'到 1.5x   成功 {len(ok)}/200 次，中位 {int(np.median(ok)) if ok else "未达"} 笔')
log('')

log('=' * 96)
log('【四】不复利对照（每笔固定本金，累计而非复利）')
log('=' * 96)
log(f'累计和 {net.sum():.2%}（{len(net)} 笔），等价于每笔固定仓位、不滚动')
log(f'平均每笔 {net.mean():.4%}；若按 100 笔计，约 {net.mean()*100:.2f}% 累计')
log('')
log('诚实边界：')
log('  1. 顺序复利 = 上界：假设前一笔记完才开下一笔，实际多币并发，真实回撤可能更大或更小。')
log('  2. 零滑点；成本 0.04% 单边，若往返 0.08%，全部均值再减 0.04pp。')
log('  3. TP 用 close 触价，实盘需限价单成交，成交率未计。')
log('  4. 未计容量：小市值币无法容纳大资金。')
log('  5. 全样本（3 年）；tp_verify 的 9/9 季正是同一批数据的分季结果。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
log('')
log(f'已写 {OUT}  TOTAL {time.time()-t0:.0f}s')
