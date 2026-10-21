# -*- coding: utf-8 -*-
"""gambler_micro.py — 用「3 年从没用过的字段」找操盘者痕迹
数据里有但从没碰过的列：trades（成交笔数）、taker_base/taker_quote（主动买量/额）
派生：主动买占比、单笔均额、影线结构、收盘位置 —— 这些是「盘中谁在动手」的直接痕迹
"""
import os, numpy as np, pandas as pd
try:
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except Exception:
    HAS_SK = False

D = r"D:\中信证券\coinlab\data"; E = r"D:\中信证券\coinlab\evidence"
COLS = ['symbol','open_time','open','high','low','close','volume','quote_volume','trades','taker_base','taker_quote']

d = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'), usecols=COLS)
d['open_time'] = pd.to_datetime(d['open_time'], unit='ms', utc=True)
d = d.sort_values(['symbol','open_time'], kind='stable').reset_index(drop=True)
print('rows %d   coins %d   时间 %s ~ %s' % (len(d), d.symbol.nunique(), d.open_time.min(), d.open_time.max()))

g = d.groupby('symbol', sort=False)
d['tbr'] = (d['taker_base'] / d['volume'].replace(0, np.nan)).astype('float32')
d['ats'] = (d['quote_volume'] / d['trades'].replace(0, np.nan)).astype('float32')
rng = (d['high'] - d['low']).replace(0, np.nan)
oc_max = d[['open','close']].max(axis=1); oc_min = d[['open','close']].min(axis=1)
d['cpos'] = ((d['close'] - d['low']) / rng).astype('float32')
d['ush'] = ((d['high'] - oc_max) / rng).astype('float32')
d['dsh'] = ((oc_min - d['low']) / rng).astype('float32')
d['ret'] = g['close'].pct_change().astype('float32')

for c, out in [('volume','volr'), ('ats','atsr'), ('trades','trdr')]:
    med = g[c].transform(lambda s: s.rolling(288, min_periods=50).median().shift(1))
    d[out] = (d[c] / med).astype('float32')

print('主动买占比 中位 %.4f   单笔均额 中位 $%.1f   笔数 中位 %.0f' % (
    d['tbr'].median(), d['ats'].median(), d['trades'].median()))

ev = pd.read_csv(os.path.join(E, 'pump_hunt_events.csv'))
ev['start_time'] = pd.to_datetime(ev['start_time'], utc=True)
ev['peak_time'] = pd.to_datetime(ev['peak_time'], utc=True)

idx = {s: gg.reset_index(drop=True) for s, gg in d.groupby('symbol', sort=False)}
aligned = []
for _, e in ev.iterrows():
    gg = idx.get(e['symbol'])
    if gg is None:
        continue
    i0 = int(np.searchsorted(gg['open_time'].values, e['start_time'].value))
    i1 = int(np.searchsorted(gg['open_time'].values, e['peak_time'].value))
    if i0 < 1 or i1 >= len(gg) or i1 <= i0:
        continue
    aligned.append((e['symbol'], i0, i1))
print('事件表 %d 个，成功对齐 %d 个' % (len(ev), len(aligned)))

def stat(x):
    return (len(x), np.nanmean(x['tbr']), np.nanmedian(x['ats']),
            np.nanmedian(x['volr']), np.nanmedian(x['atsr']), np.nanmedian(x['trdr']),
            np.nanmean(x['cpos']), np.nanmean(x['ush']), np.nanmean(x['dsh']))

def show(name, r):
    print('%-24s %9d %9.4f %11.1f %8.2f %8.2f %8.2f %9.3f %9.3f' % (
        name, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))

HDR = '%-24s %9s %9s %11s %8s %8s %8s %9s %9s' % ('段','n','主动买%','单笔额$','量比','单笔比','笔数比','收盘位','上影F')
print('\n' + HDR); print('-' * len(HDR))
show('【全样本基准】', stat(d))
for name, sp in [('拉升段 start->peak', [(s,i0,i1) for s,i0,i1 in aligned]),
                 ('前24h 吸筹期',       [(s,max(0,i0-288),i0) for s,i0,i1 in aligned]),
                 ('见顶后4h 派发期',    [(s,i1,min(len(idx[s]),i1+48)) for s,i0,i1 in aligned])]:
    parts = [idx[s].iloc[a:b] for s, a, b in sp if b > a]
    if parts:
        show(name, stat(pd.concat(parts)))

print('\n【事件启动时刻 UTC 小时分布】（均匀基准=4.17%/h）')
for k, v in ev['start_time'].dt.hour.value_counts().sort_index().items():
    p = 100.0 * v / len(ev)
    print('  %02d时 %5.2f%% %s' % (k, p, '#' * int(p * 2)))

if HAS_SK:
    pos = pd.DataFrame([idx[s].iloc[i0-1] for s, i0, i1 in aligned])
    neg = d.sample(min(200000, len(d)), random_state=0)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    print('\n【E 早期可识别性】事件启动「前一根」vs 随机根（单特征 AUC）')
    for c in ['tbr','volr','atsr','trdr','cpos','ush','dsh','ret']:
        a = np.r_[pos[c].values.astype('float64'), neg[c].values.astype('float64')]
        m = ~np.isnan(a)
        pm = np.nanmedian(pos[c].values); nm = np.nanmedian(neg[c].values)
        if m.sum() < 500 or np.unique(y[m]).size < 2:
            continue
        print('  %-6s AUC=%.4f   正样本中位=%+.4g   随机中位=%+.4g' % (c, roc_auc_score(y[m], a[m]), pm, nm))
else:
    print('sklearn 不可用，跳过 AUC')

print('\nTOTAL done')
