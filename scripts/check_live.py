"""data/live 质量体检：脏时间戳、按小时覆盖、盘口采样密度。
目的：判断这份数据能不能支撑"峰前 1 分钟认峰"，而不是先假设能/不能。
"""
import pandas as pd, numpy as np, os, time

BASE = r'D:/中信证券/coinlab/data/live'
OUT = r'D:/中信证券/coinlab/evidence/live_quality.txt'
L = []
L.append('data/live 质量体检  %s' % time.strftime('%Y-%m-%d %H:%M'))


def smart_read(p):
    with open(p, encoding='utf-8') as f:
        hdr = f.readline().strip().lstrip('\ufeff').split(',')
        row = f.readline().strip().split(',')
    hdr = [str(h).strip() for h in hdr]
    if len(row) == len(hdr) + 1:
        hdr = hdr + ['extra_%d' % len(hdr)]
    return pd.read_csv(p, names=hdr, header=0, index_col=False)


for day in sorted(os.listdir(BASE)):
    d = os.path.join(BASE, day)
    if not os.path.isdir(d):
        continue
    L.append('')
    L.append('=' * 72)
    L.append('日期 %s' % day)

    # --- klines_1s ---
    p = os.path.join(d, 'klines_1s.csv')
    if os.path.exists(p):
        df = smart_read(p)
        df['ts'] = pd.to_datetime(df.open_time, unit='ms')
        L.append('-- klines_1s: %d 行, %d 币' % (len(df), df.symbol.nunique()))
        early = df[df.open_time < 1_700_000_000_000]
        L.append('   脏时间戳(<2023-11) %d 行 (%.2f%%), 值样例: %s'
                 % (len(early), len(early) / len(df) * 100,
                    sorted(early.open_time.unique())[:5] if len(early) else '-'))
        clean = df[df.open_time >= 1_700_000_000_000]
        if len(clean):
            L.append('   干净段 %s ~ %s' % (pd.to_datetime(clean.open_time.min(), unit='ms'),
                                            pd.to_datetime(clean.open_time.max(), unit='ms')))
            h = clean.groupby([clean.ts.dt.floor('h'), 'symbol']).size().unstack(fill_value=0)
            L.append('   按小时 × 币的秒数（每格满格=3600）:')
            L.append('     ' + ' '.join('%s' % t.strftime('%m-%d %H') for t in h.index))
            L.append('     每币中位: ' + ' '.join('%d' % int(v) for v in h.median(axis=1).values))
            L.append('     有数据的币数: ' + ' '.join('%d' % int((h.loc[t] > 0).sum()) for t in h.index))
        # 秒级连续性：同一币相邻行时间差
        sub = clean[clean.symbol == clean.symbol.iloc[0]].sort_values('open_time')
        gaps = np.diff(sub.open_time.values) // 1000
        if len(gaps):
            L.append('   %s 相邻秒差 中位=%.0fs P90=%.0fs 最大=%.0fs'
                     % (clean.symbol.iloc[0], np.median(gaps), np.percentile(gaps, 90), gaps.max()))

    # --- depth20 ---
    p = os.path.join(d, 'depth20.csv')
    if os.path.exists(p):
        df = smart_read(p)
        snaps = df.groupby(['symbol', 'ts_ms']).size().groupby('symbol').size()
        L.append('-- depth20: %d 行, %d 币, 每币快照数 中位=%d 最小=%d 最大=%d'
                 % (len(df), df.symbol.nunique(), int(snaps.median()), snaps.min(), snaps.max()))
        ts = np.sort(df.ts_ms.unique())
        gaps = np.diff(ts) / 1000.0
        if len(gaps):
            L.append('   快照间隔(s) 中位=%.0f P10=%.0f P90=%.0f 最大=%.0f'
                     % (np.median(gaps), np.percentile(gaps, 10), np.percentile(gaps, 90), gaps.max()))
        L.append('   注: 20 档盘口，若间隔≈300s 则只有"每5分钟一张快照"，不能做事前盘口微结构。')

    # --- aggTrades 抽样（只读 10-18 的按小时分布） ---
    p = os.path.join(d, 'aggTrades.csv')
    if os.path.exists(p) and day.endswith('10-18'):
        nrow = 0; cnt = {}
        for ch in pd.read_csv(p, usecols=['time', 'symbol'], chunksize=3_000_000):
            nrow += len(ch)
            hh = pd.to_datetime(ch.time, unit='ms').dt.floor('h').astype(str)
            for k, v in hh.value_counts().items():
                cnt[k] = cnt.get(k, 0) + int(v)
        L.append('-- aggTrades: %d 行, 按小时:' % nrow)
        for k in sorted(cnt):
            L.append('     %s  %8d 行' % (k, cnt[k]))

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
