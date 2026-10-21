"""data/live 覆盖探查：到底有多少天、多少币、多长、什么采样密度。
只读统计，不改数据。
"""
import pandas as pd, numpy as np, os, time, glob

BASE = r'D:/中信证券/coinlab/data/live'
OUT = r'D:/中信证券/coinlab/evidence/live_coverage.txt'
L = []
L.append('data/live 覆盖探查  %s' % time.strftime('%Y-%m-%d %H:%M'))

for day in sorted(os.listdir(BASE)):
    d = os.path.join(BASE, day)
    if not os.path.isdir(d):
        continue
    L.append('')
    L.append('=' * 70)
    L.append('日期目录 %s' % day)
    for fn in ('klines_1s.csv', 'depth20.csv', 'aggTrades.csv'):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            L.append('  %-16s 缺失' % fn)
            continue
        t0 = time.time()
        if fn == 'aggTrades.csv':
            # 大文件：分块统计
            nsym = {}; tmin = None; tmax = None; nrow = 0
            for ch in pd.read_csv(p, usecols=['symbol', 'time'], chunksize=2_000_000):
                nrow += len(ch)
                vc = ch.symbol.value_counts()
                for k, v in vc.items():
                    nsym[k] = nsym.get(k, 0) + int(v)
                a, b = int(ch.time.min()), int(ch.time.max())
                tmin = a if tmin is None else min(tmin, a)
                tmax = b if tmax is None else max(tmax, b)
            L.append('  %-16s 行=%9d  币=%3d  时间 %s ~ %s  跨度 %.1fh  %.1fs'
                     % (fn, nrow, len(nsym),
                        pd.to_datetime(tmin, unit='ms'), pd.to_datetime(tmax, unit='ms'),
                        (tmax - tmin) / 3.6e6, time.time() - t0))
        else:
            with open(p, encoding='utf-8') as f:
                hdr = f.readline().strip().lstrip('\ufeff').split(',')
                row = f.readline().strip().split(',')
            hdr = [str(h).strip() for h in hdr]
            if len(row) == len(hdr) + 1:
                hdr = hdr + ['extra_%d' % len(hdr)]
            df = pd.read_csv(p, names=hdr, header=0, index_col=False)
            tcol = 'open_time' if 'open_time' in df.columns else ('ts_ms' if 'ts_ms' in df.columns else 'time')
            g = df.groupby('symbol').size()
            L.append('  %-16s 行=%9d  币=%3d  时间 %s ~ %s  跨度 %.1fh  %.1fs'
                     % (fn, len(df), df.symbol.nunique(),
                        pd.to_datetime(df[tcol].min(), unit='ms'),
                        pd.to_datetime(df[tcol].max(), unit='ms'),
                        (df[tcol].max() - df[tcol].min()) / 3.6e6, time.time() - t0))
            L.append('      每币行数 min=%d 中位=%d max=%d  | 列: %s'
                     % (g.min(), int(g.median()), g.max(), ','.join(df.columns)))
            if fn == 'depth20.csv':
                sub = df[df.symbol == df.symbol.iloc[0]]
                ts = np.sort(sub.ts_ms.unique())
                if len(ts) > 3:
                    L.append('      盘口快照: 每币快照数中位=%d, 快照间隔中位=%.0f ms'
                             % (int(df.groupby(['symbol', 'ts_ms']).size().groupby('symbol').size().median()),
                                float(np.median(np.diff(ts)))))
                L.append('      盘口档数: %d 档 (side=%s)'
                         % (df[df.symbol == df.symbol.iloc[0]].level.max() + 1,
                            sorted(df.side.unique())[:2]))
            if fn == 'klines_1s.csv':
                # 每秒完整度
                sub = df[df.symbol == df.symbol.iloc[0]].sort_values(tcol)
                exp = (sub[tcol].max() - sub[tcol].min()) / 1000 + 1
                L.append('      抽样(%s): 应有 %.0f 秒, 实有 %d 行, 完整度 %.1f%%'
                         % (df.symbol.iloc[0], exp, len(sub), len(sub) / exp * 100))

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
