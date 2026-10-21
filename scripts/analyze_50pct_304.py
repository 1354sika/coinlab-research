"""
换一个币池验证用户的主张：他"不看所有币"——很可能看的是小市值/新上币。
klines_5m_502.csv 实际有 304 个币（比 60 币池多 5 倍，含大量小币）。
用同一口径再扫一遍：8 小时内「max(high)/entry-1 >= 50%」的比例。

若 304 币池里这个比例显著高于 60 币池（例如 20%~50%），
则说明：他的经验成立，但前提是「选币池」而非「K 线形态」。
若仍然很低，则说明「一半到过 50%」在他的口径下另有含义。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_502.csv'
OUT = r'D:/中信证券/coinlab/evidence/50pct_8h_304.txt'
H = 96

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('读入 %d 行 %.1fs' % (len(df), time.time() - t0), flush=True)
df = df.sort_values(['symbol', 'open_time'])

conds = {
    'A 全部决策点': lambda P: np.ones(P['n'], bool),
    'B 1h涨>=20%': lambda P: P['ret12'] >= 0.20,
    'C 1h涨>=30%': lambda P: P['ret12'] >= 0.30,
    'D 1h涨>=50%': lambda P: P['ret12'] >= 0.50,
    'E 24h涨>=50%': lambda P: P['ret288'] >= 0.50,
    'F 24h涨>=100%': lambda P: P['ret288'] >= 1.00,
    'G 24h涨>=200%': lambda P: P['ret288'] >= 2.00,
}
agg = {k: {'n': 0, 'ge50': 0, 'mr': [], 'mset': set()} for k in conds}
per_sym = {}
nsym = 0

for sym, g in df.groupby('symbol', sort=False):
    h = g.high.to_numpy(np.float32); c = g.close.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 1200 + H:
        continue
    nsym += 1
    ret12 = np.zeros(n); ret12[12:] = c[12:] / c[:-12] - 1.0
    ret288 = np.zeros(n)
    if n > 288:
        ret288[288:] = c[288:] / c[:-288] - 1.0
    futmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    valid = np.isfinite(futmax) & (np.arange(n) >= 300)
    mr_all = np.where(valid, futmax / np.maximum(c, 1e-12) - 1.0, np.nan)
    mos = pd.to_datetime(t, unit='ms').to_period('M').astype(str)
    P = dict(n=n, ret12=ret12, ret288=ret288)
    for k, fn in conds.items():
        m = fn(P) & valid
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        mr = mr_all[idx]
        d = agg[k]
        d['n'] += len(idx)
        d['ge50'] += int((mr >= 0.5).sum())
        if len(mr) <= 20000:
            d['mr'].append(mr)
        d['mset'].update(set(mos[idx]))
        per_sym.setdefault(sym, {}).setdefault(k, [0, 0])
        per_sym[sym][k][0] += len(idx)
        per_sym[sym][k][1] += int((mr >= 0.5).sum())

del df
print('扫描完成 %.1fs' % (time.time() - t0), flush=True)

L = []
L.append('304 币池：8 小时内"到过 +50%%"普查  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据 klines_5m_502.csv（实际 %d 币有足够数据）| 5m | 只用 K 线' % nsym)
L.append('指标 = 入场后 8h 内 max(high)/entry - 1 >= 50%% 的比例')
L.append('')
L.append('  %-20s %10s %9s %10s %12s' % ('入场条件', '样本数', '次/月', '>=50%比例', '中位最大浮盈'))
for k in conds:
    d = agg[k]
    if d['n'] == 0:
        continue
    mr = np.concatenate(d['mr']) if d['mr'] else np.array([])
    nm = max(len(d['mset']), 1)
    L.append('  %-20s %10d %8.0f %9.1f%% %11.1f%%'
             % (k, d['n'], d['n'] / nm, d['ge50'] / d['n'] * 100,
                np.median(mr) * 100 if len(mr) else float('nan')))
L.append('')
L.append('=== 单币排名（条件 D: 1h涨>=50% 时，8h 内到过 50% 的比例，样本>=5）')
rows = []
for sym, d in per_sym.items():
    n_, g_ = d.get('D 1h涨>=50%', (0, 0))
    if n_ >= 5:
        rows.append((g_ / n_, n_, sym))
rows.sort(reverse=True)
for r, n_, sym in rows[:15]:
    L.append('  %-14s 触发 %4d 次，到过 50%% 的 %3d 次 (%.0f%%)' % (sym, n_, int(r * n_), r * 100))
L.append('')
L.append('对照（60 币池，同一口径）：1h涨>=50%% 时 11.4%%；24h涨>=100%% 时 9.1%%。')
L.append('诚实边界：high 价成交假设；未计滑点/成本；304 币多为小市值，流动性差，实际成交价可能远低于 high。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
