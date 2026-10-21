"""
按用户口径重做：8 小时上限内「曾到过 +50%」的条件普查（全点位，不抽样）。

上一版每 8 小时只取一个点（STEP=96），样本太小不可信。本版扫描**每一个 5m 点**，
用向量化统计，回答：在 60 币 / 3 年 / 只用 K 线的条件下，
哪一类入场条件能让「8h 内最大浮盈 >= 50%」的比例接近一半？
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/50pct_8h_full.txt'
H = 96
DIP, VOLR = -0.02, 3.0

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])

conds = {
    'A 全部决策点': lambda P: np.ones(P['n'], bool),
    'B 急跌2%+量比3': lambda P: (P['ret24'] <= DIP) & (P['volr'] >= VOLR),
    'C 1h涨>=10%': lambda P: P['ret12'] >= 0.10,
    'D 1h涨>=20%': lambda P: P['ret12'] >= 0.20,
    'E 1h涨>=30%': lambda P: P['ret12'] >= 0.30,
    'F 1h涨>=50%': lambda P: P['ret12'] >= 0.50,
    'G 24h涨>=50%': lambda P: P['ret288'] >= 0.50,
    'H 24h涨>=100%': lambda P: P['ret288'] >= 1.00,
    'I 新高96+量比3': lambda P: (P['h'] >= P['hi96'] - 1e-9) & (P['volr'] >= 3.0),
    'J 量比>=5 且 1h涨>=5%': lambda P: (P['volr'] >= 5.0) & (P['ret12'] >= 0.05),
    'K 量比>=10 且 1h涨>=10%': lambda P: (P['volr'] >= 10.0) & (P['ret12'] >= 0.10),
}

agg = {k: {'n': 0, 'ge50': 0, 'mr': [], 'months': set()} for k in conds}
per_sym = {}

for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); c = g.close.to_numpy(np.float32)
    v = g.volume.to_numpy(np.float32); t = g.open_time.to_numpy(); n = len(c)
    if n < 1200 + H:
        continue
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    ret12 = np.zeros(n); ret12[12:] = c[12:] / c[:-12] - 1.0
    ret288 = np.zeros(n); ret288[288:] = c[288:] / c[:-288] - 1.0
    hi96 = pd.Series(h).rolling(96, min_periods=96).max().to_numpy()
    futmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    P = dict(h=h, c=c, n=n, volr=volr, ret24=ret24, ret12=ret12, ret288=ret288, hi96=hi96)
    valid = np.isfinite(futmax) & (np.arange(n) >= 300)
    mr_all = np.where(valid, futmax / np.maximum(c, 1e-12) - 1.0, np.nan)
    mos = pd.to_datetime(t, unit='ms').to_period('M').astype(str)

    for k, fn in conds.items():
        try:
            m = fn(P) & valid
        except Exception:
            continue
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        mr = mr_all[idx]
        d = agg[k]
        d['n'] += len(idx)
        d['ge50'] += int((mr >= 0.5).sum())
        if len(mr) <= 40000:
            d['mr'].append(mr)
        d['months'].update(set(mos[idx]))
        per_sym.setdefault(k, {}).setdefault(sym, [0, 0])
        per_sym[k][sym][0] += len(idx)
        per_sym[k][sym][1] += int((mr >= 0.5).sum())

del df
print('扫描完成 %.1fs' % (time.time() - t0), flush=True)

L = []
L.append('8 小时内"到过 +50%%"条件普查（全点位，不抽样）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('60 币 / 5m / 3 年 | 窗口=入场后 8h | 指标=窗口内 max(high)/entry-1 | 只用 K 线')
L.append('')
L.append('  %-24s %9s %9s %10s %12s' % ('入场条件', '样本数', '次/月', '>=50%比例', '中位最大浮盈'))
for k in conds:
    d = agg[k]
    if d['n'] == 0:
        L.append('  %-24s %9d %9s' % (k, 0, '-'))
        continue
    mr = np.concatenate(d['mr']) if d['mr'] else np.array([])
    nm = max(len(d['months']), 1)
    L.append('  %-24s %9d %9.0f %9.1f%% %11.1f%%'
             % (k, d['n'], d['n'] / nm, d['ge50'] / d['n'] * 100,
                np.median(mr) * 100 if len(mr) else float('nan')))
L.append('')
L.append('=== 分币：哪些币在"1h涨>=30%%"条件下 8h 内到过 50%%（前 10）')
ps = per_sym.get('E 1h涨>=30%', {})
rank = sorted(ps.items(), key=lambda kv: -kv[1][1])[:10]
for sym, (n_, g_) in rank:
    L.append('  %-12s 触发 %4d 次，其中到过 50%% 的 %3d 次 (%.0f%%)'
             % (sym, n_, g_, g_ / max(n_, 1) * 100))
L.append('')
L.append('诚实边界：')
L.append('  1. 用 high 计算最大浮盈，能否在该价位成交取决于挂单与流动性；未计滑点。')
L.append('  2. 60 币是主流币池，天然缺"一天翻倍"的小市值币；用户的币池可能不同。')
L.append('  3. 未计成本（本表只看浮盈，不是收益结论）。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
