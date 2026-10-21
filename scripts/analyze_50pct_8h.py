"""
用户纠正后的关键检验：他说的是「**8 小时上限**内，**一半笔数曾到过 +50%**」，且**只看 K 线**。

我上一轮测的是 30 天窗口，答错了题。这次严格按他的口径：
  窗口 = 入场后 8 小时（96 根 5m）
  指标 = 窗口内最大浮盈（max high / entry - 1）
  问题 = 哪类入场条件下，「≥50% 的比例」能达到一半？

候选入场（全部只用 K 线，无任何其它数据）：
  A 随机基准（每 96 根取一点）
  B 急跌 ret24<=-2% & 量比>=3（现有入场）
  C 1 小时涨 >=10%
  D 1 小时涨 >=20%
  E 1 小时涨 >=30%
  F 创 96 根新高 & 量比>=3
  G 量比 >=5 & 1 小时涨 >=5%
  H 1 小时涨 >=50%（已经在暴拉的）
每种都报：次数、每月频率、8h 内 >=50% 的比例、中位最大浮盈、8h 结束收益。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/50pct_8h.txt'
H = 96
COST = 0.0004
DIP, VOLR = -0.02, 3.0

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 1200 + H: return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    ret12 = np.zeros(n); ret12[12:] = c[12:] / c[:-12] - 1.0
    hi96 = pd.Series(h).rolling(96, min_periods=96).max().to_numpy()
    # 未来 96 根（不含当根）的最高/最低
    futmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    futmin = pd.Series(l).rolling(H, min_periods=H).min().shift(-H).to_numpy()
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24, ret12=ret12,
                hi96=hi96, futmax=futmax, futmin=futmin)


conds = {
    'A 随机基准': lambda P, i: True,
    'B 急跌2%+量比3': lambda P, i: P['ret24'][i] <= DIP and P['volr'][i] >= VOLR,
    'C 1h涨10%': lambda P, i: P['ret12'][i] >= 0.10,
    'D 1h涨20%': lambda P, i: P['ret12'][i] >= 0.20,
    'E 1h涨30%': lambda P, i: P['ret12'][i] >= 0.30,
    'F 新高96+量比3': lambda P, i: P['h'][i] >= P['hi96'][i] - 1e-9 and P['volr'][i] >= 3.0,
    'G 量比5+1h涨5%': lambda P, i: P['volr'][i] >= 5.0 and P['ret12'][i] >= 0.05,
    'H 1h涨50%': lambda P, i: P['ret12'][i] >= 0.50,
}

res = {k: {'n': 0, 'ge50': 0, 'maxret': [], 'endret': [], 'months': set()} for k in conds}

for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    n = P['n']
    step = 96 if True else 1
    idxs = range(300, n - H - 1, 96)   # 每 8 小时取一个决策点（去重）
    for i in idxs:
        fm = P['futmax'][i]
        if not np.isfinite(fm): continue
        e0 = float(P['c'][i])
        mr = float(fm / e0 - 1.0)
        er = float(P['c'][min(i + H, n - 1)] / e0 - 1.0)
        mo = pd.Timestamp(int(P['t'][i]) // 10 ** 6, unit='ms').strftime('%Y-%m')
        for k, fn in conds.items():
            try:
                ok = fn(P, i)
            except Exception:
                ok = False
            if ok:
                d = res[k]
                d['n'] += 1
                if mr >= 0.5:
                    d['ge50'] += 1
                d['maxret'].append(mr)
                d['endret'].append(er)
                d['months'].add(mo)

del df
print('扫描完成 %.1fs' % (time.time() - t0), flush=True)

L = []
L.append('8 小时内"到过 +50%%"的条件普查（只用 K 线）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('窗口 = 入场后 8 小时（96 根 5m）；指标 = 窗口内最大浮盈；60 币 / 3 年')
L.append('决策点每 8 小时取一个（STEP=96），避免重叠')
L.append('')
L.append('  %-16s %9s %10s %10s %12s %12s %10s' % ('入场条件', '样本数', '次/月', '>=50%比例', '中位最大浮盈', '中位8h收益', '均值8h收益'))
for k in conds:
    d = res[k]
    if d['n'] == 0:
        L.append('  %-16s %9d %10s' % (k, 0, '-'))
        continue
    nm = len(d['months'])
    L.append('  %-16s %9d %9.1f %9.1f%% %11.1f%% %11.1f%% %9.2f%%'
             % (k, d['n'], d['n'] / max(nm, 1),
                d['ge50'] / d['n'] * 100,
                np.median(d['maxret']) * 100,
                np.median(d['endret']) * 100,
                np.mean(d['endret']) * 100))
L.append('')
L.append('读法：')
L.append('  ">=50%比例" 就是"曾经到过 50% 的笔数占比"。用户主张一半以上，看哪一列接近 50%。')
L.append('  "次/月" 是该条件下每月的决策点数量（60 币合计，已按 8h 去重）。')
L.append('')
L.append('诚实边界：')
L.append('  1. 决策点每 96 根取一个，实际可以让更多，这里只做条件普查，不追求最大次数。')
L.append('  2. 最大浮盈用 high 计算，实盘能否在那根 high 卖出取决于挂单与流动性，未计滑点。')
L.append('  3. 未计成本（此表只看浮盈，不构成收益结论）。')
L.append('  4. 60 币、3 年；不含更早期与更小市值币。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
