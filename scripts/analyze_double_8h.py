"""
按用户最新口径重测：**本金的 50% 翻倍**。
即：仓位约为本金的 50%，价格翻倍（+100%），本金浮盈 = 50%。
=> 关键指标是「入场后 8 小时内，价格是否曾翻倍（max(high)/entry-1 >= 100%）」。

同时给出 >=50% / >=100% / >=200% 三档，便于看清"一半笔数"落在哪一档。
数据：klines_5m_502.csv（304 币，最接近他"挑着看"的池子）+ 60 币主池对照。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_502.csv'
OUT = r'D:/中信证券/coinlab/evidence/double_8h_304.txt'
H = 96

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('读入 %d 行 %.1fs' % (len(df), time.time() - t0), flush=True)
df = df.sort_values(['symbol', 'open_time'])

conds = {
    'A 全部决策点': lambda P: np.ones(P['n'], bool),
    'B 1h涨>=20%': lambda P: P['ret12'] >= 0.20,
    'C 1h涨>=50%': lambda P: P['ret12'] >= 0.50,
    'D 24h涨>=50%': lambda P: P['ret288'] >= 0.50,
    'E 24h涨>=100%': lambda P: P['ret288'] >= 1.00,
    'F 24h涨>=200%': lambda P: P['ret288'] >= 2.00,
    'G 量比>=10 且 1h涨>=20%': lambda P: (P['volr'] >= 10.0) & (P['ret12'] >= 0.20),
}
agg = {k: {'n': 0, 'g50': 0, 'g100': 0, 'g200': 0, 'mset': set()} for k in conds}
per_sym = {}
nsym = 0

for sym, g in df.groupby('symbol', sort=False):
    h = g.high.to_numpy(np.float32); c = g.close.to_numpy(np.float32)
    v = g.volume.to_numpy(np.float32); t = g.open_time.to_numpy(); n = len(c)
    if n < 1200 + H:
        continue
    nsym += 1
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret12 = np.zeros(n); ret12[12:] = c[12:] / c[:-12] - 1.0
    ret288 = np.zeros(n)
    if n > 288:
        ret288[288:] = c[288:] / c[:-288] - 1.0
    futmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    valid = np.isfinite(futmax) & (np.arange(n) >= 300)
    mr = np.where(valid, futmax / np.maximum(c, 1e-12) - 1.0, np.nan)
    mos = pd.to_datetime(t, unit='ms').to_period('M').astype(str)
    P = dict(n=n, ret12=ret12, ret288=ret288, volr=volr)
    for k, fn in conds.items():
        m = fn(P) & valid
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        x = mr[idx]
        d = agg[k]
        d['n'] += len(idx)
        d['g50'] += int((x >= 0.5).sum())
        d['g100'] += int((x >= 1.0).sum())
        d['g200'] += int((x >= 2.0).sum())
        d['mset'].update(set(mos[idx]))
        per_sym.setdefault(sym, {}).setdefault(k, [0, 0])
        per_sym[sym][k][0] += len(idx)
        per_sym[sym][k][1] += int((x >= 1.0).sum())

del df
print('扫描完成 %.1fs' % (time.time() - t0), flush=True)

L = []
L.append('8 小时内"到过 +50%% / +100%%"普查（投入 100 赚 150 => 需 +50%%）  ' + time.strftime('%Y-%m-%d %H:%M'))
L.append('数据 klines_5m_502.csv（%d 币）| 5m | 只用 K 线 | 窗口=入场后 8h' % nsym)
L.append('')
L.append('  %-24s %10s %9s %9s %10s %10s' % ('入场条件', '样本数', '>=50%', '>=100%', '>=200%', '次/月'))
for k in conds:
    d = agg[k]
    if d['n'] == 0:
        continue
    nm = max(len(d['mset']), 1)
    L.append('  %-24s %10d %8.1f%% %8.2f%% %9.2f%% %9.0f'
             % (k, d['n'], d['g50'] / d['n'] * 100, d['g100'] / d['n'] * 100,
                d['g200'] / d['n'] * 100, d['n'] / nm))
L.append('')
L.append('=== 单币：8h 内价格翻倍(>=100%)的比例（条件 B: 1h涨>=20%，样本>=5）')
rows = []
for sym, d in per_sym.items():
    n_, g_ = d.get('B 1h涨>=20%', (0, 0))
    if n_ >= 5:
        rows.append((g_ / n_, n_, g_, sym))
rows.sort(reverse=True)
for r, n_, g_, sym in rows[:15]:
    L.append('  %-14s 触发 %4d 次，翻倍 %3d 次 (%.0f%%)' % (sym, n_, g_, r * 100))
L.append('')
L.append('诚实边界：max(high) 成交假设；304 币多为小市值，流动性差，实际难以在最高价成交；')
L.append('  未计滑点/成本；样本重叠。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
