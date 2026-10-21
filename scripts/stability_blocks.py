# -*- coding: utf-8 -*-
"""机制分时段稳定性（random_control 的 C 部分，独立跑，修 DatetimeIndex bug）
把 365 天按 2 个月一段切成 6 段，逐个机制看每段是否仍然有效。
判据：若某机制只在 1-2 段有效、其余为负 -> 是「特殊时期的产物」，不是普遍规律。
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, DQ, FEE_M = 60, 0.90, 0.0002
t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
QV = qv.values.astype(np.float64)
qv60 = qv.rolling(60).mean().values
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
R60 = (px / px.shift(60) - 1.0).values.astype(np.float64)
MA60 = px.rolling(60).mean().values
madev = P / MA60 - 1.0
FWD = np.full((T, N), np.nan)
FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)

# 月标签（DatetimeIndex -> str）
idx = pd.to_datetime(t, unit='ms', utc=True)
mo = idx.strftime('%Y-%m').values           # numpy array of str
months = sorted(set(mo))
blocks = [months[i:i + 2] for i in range(0, len(months), 2)]
say('月份 %d 个 -> 时段 %d 段: %s' % (len(months), len(blocks),
                                    ' | '.join('/'.join(b) for b in blocks)))

r40 = np.nanquantile(R60[np.isfinite(R60)], 0.4)
m20 = np.nanquantile(madev[np.isfinite(madev)], 0.2)

MECHS = {
    'M1 急跌+volr>10(做多)': ((streak_ret < -0.05) & (volr > 10) & sel, +1),
    'M1b 急跌+volr>10+段长>=3': ((streak_ret < -0.05) & (volr > 10) & (seglen >= 3) & sel, +1),
    'M2 急涨+volr>10(做多)': ((streak_ret > +0.05) & (volr > 10) & sel, +1),
    'M2b 急涨+volr>10(做空)': ((streak_ret > +0.05) & (volr > 10) & sel, -1),
    'M3 急跌+volr>10(做空)': ((streak_ret < -0.05) & (volr > 10) & sel, -1),
    'M4 低位+远离均线(做多)': ((R60 <= r40) & (madev <= m20) & sel, +1),
    'M5 急跌+volr>5(做多)': ((streak_ret < -0.05) & (volr > 5) & sel, +1),
}

say('\n=== 分时段：各机制的平均 60 分钟收益（已乘方向；% ）===')
say('  %-28s' % '机制' + ''.join('%16s' % ('/'.join(b)) for b in blocks))
res = {}
for nm, (m, sign) in MECHS.items():
    row = '  %-28s' % nm
    vals = []
    for b in blocks:
        mm = np.isin(mo, b)[:, None] & np.ones((1, N), bool)
        y = FWD[m & mm]; y = y[np.isfinite(y)] * sign
        if len(y) >= 20:
            vals.append(100 * y.mean())
            row += '%16s' % ('%+.2f%%(n%d)' % (100 * y.mean(), len(y)))
        else:
            vals.append(np.nan)
            row += '%16s' % '-'
    say(row)
    res[nm] = vals

say('\n=== 汇总：每段为负的段数 / 有数据段数（越少越稳）===')
for nm, vals in res.items():
    v = np.array([x for x in vals if np.isfinite(x)])
    say('  %-28s 有效段=%d  为正=%d  为负=%d  段间极差=%.2fpp  均值=%+.3f%%' % (
        nm, len(v), int((v > 0).sum()), int((v < 0).sum()),
        (v.max() - v.min()) if len(v) else np.nan, v.mean() if len(v) else np.nan))

with open(os.path.join(EV, 'stability_blocks.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
