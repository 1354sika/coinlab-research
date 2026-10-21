# -*- coding: utf-8 -*-
"""relay 逻辑快速验证（不读大文件，只用事件表，秒级）"""
import pandas as pd, numpy as np
EV = r'D:\中信证券\coinlab\evidence'
ev = pd.read_csv(EV + r'\pump_hunt_events.csv')
ev['st'] = pd.to_datetime(ev['start_time'])
ev['pk'] = pd.to_datetime(ev['peak_time'])
ev = ev.sort_values('st').reset_index(drop=True)
ts = ev['st'].values.astype('datetime64[m]').astype('int64')
tp = ev['pk'].values.astype('datetime64[m]').astype('int64')
n = len(ev)
print('events', n, '| coins', ev['symbol'].nunique(), '| span_days %.0f' % ((ts[-1] - ts[0]) / 1440))
print('dur median(min) %.0f' % np.median(np.maximum(tp - ts, 1)))

def rc(ts, tp, H):
    c = 0
    for i in range(len(ts)):
        lo = np.searchsorted(ts, tp[i], 'left')
        hi = np.searchsorted(ts, tp[i] + H, 'right')
        c += max(0, hi - max(lo, i + 1))
    return c

def ov(ts, tp):
    c = 0
    for i in range(len(ts)):
        lo = np.searchsorted(ts, ts[i], 'right')
        hi = np.searchsorted(ts, tp[i], 'right')
        c += max(0, hi - lo)
    return c

print('relay pairs 1h/4h/24h:', rc(ts, tp, 60), rc(ts, tp, 240), rc(ts, tp, 1440))
print('OVERLAP pairs (A未完成B已启动):', ov(ts, tp))
rng = np.random.default_rng(0)
dur = np.maximum(tp - ts, 1)
v = []
for _ in range(50):
    tn = np.sort(rng.integers(ts[0], ts[-1] + 1, n))
    v.append(rc(tn, tn + dur, 240))
print('null model 4h mean %.1f' % np.mean(v))
gaps = []
for i in range(n):
    lo = np.searchsorted(ts, ts[i], 'right')
    gaps.append(int(ts[lo] - tp[i]) if lo < n else 0)
gaps = np.array(gaps, float)
print('gap<0 (overlap)  %d' % int((gaps < 0).sum()))
print('gap 0~15min      %d' % int(((gaps >= 0) & (gaps <= 15)).sum()))
print('gap 15min~1h     %d' % int(((gaps > 15) & (gaps <= 60)).sum()))
print('gap 1~4h         %d' % int(((gaps > 60) & (gaps <= 240)).sum()))
print('gap 4~24h        %d' % int(((gaps > 240) & (gaps <= 1440)).sum()))
print('gap >24h         %d' % int((gaps > 1440).sum()))
