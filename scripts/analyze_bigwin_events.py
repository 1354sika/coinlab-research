# -*- coding: utf-8 -*-
"""对 bigwin 发现的爆发点做「事件级去重 + walk-forward」检验。
条件：分化时刻(disp>P90) 且 R15 < -5% 且 VolRatio > 2
问题：4653 个「分钟x币」样本是否只是少数几次事件的重复计数？
做法：同一 symbol 的连续触发（间隔<=5分钟）合并为一个事件，只取首次触发时点，
      事件级重算 P(未来5min>2%) 与净收益；再切前半/后半。
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV   = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d' % px.shape)

r = px.pct_change()
disp = r.std(axis=1)
thr = float(disp.quantile(0.90))
selrow = (disp > thr).values

R15 = (px / px.shift(15) - 1.0)
VR  = qv / qv.rolling(60).mean()
FWD = px.shift(-5) / px - 1.0

cond = (R15 < -0.05) & (VR > 2.0)
cond = cond & pd.DataFrame(np.repeat(selrow[:, None], px.shape[1], axis=1),
                           index=px.index, columns=px.columns)
say('原始触发（分钟x币）= %d' % int(cond.values.sum()))

# ---- 事件合并：同一 symbol 连续触发（gap<=5min）归为一个事件，取首次 ----
times = px.index.values
events = []
mask = cond.values
fwd = FWD.values
r15v = R15.values
for j in range(mask.shape[1]):
    col = mask[:, j]
    if not col.any():
        continue
    ii = np.flatnonzero(col)
    gaps = np.diff(ii)
    starts = np.r_[0, np.flatnonzero(gaps > 5) + 1]
    for s in starts:
        k = ii[s]                      # 事件首发分钟（行号）
        f = fwd[k, j]
        if not np.isfinite(f):
            continue
        events.append((times[k], str(px.columns[j]), r15v[k, j], f))
ev = pd.DataFrame(events, columns=['t', 'sym', 'r15', 'fwd5'])
say('事件数（去重后）= %d   压缩比 %.1fx' % (len(ev), int(cond.values.sum()) / max(len(ev), 1)))

fee_t, fee_m = 0.0020, 0.0004
for nm, d in [('全样本', ev),
              ('前半', ev[ev.t < np.median(ev.t)]),
              ('后半', ev[ev.t >= np.median(ev.t)])]:
    if len(d) == 0:
        continue
    y = d.fwd5.values
    say('\n[%s] n=%d' % (nm, len(d)))
    say('  P(>2%%)=%.2f%%  P(>1%%)=%.2f%%  P(>0.2%%)=%.2f%%  均值=%+.4f%%  中位=%+.4f%%' % (
        100*(y > .02).mean(), 100*(y > .01).mean(), 100*(y > .002).mean(),
        100*y.mean(), 100*np.median(y)))
    say('  净收益: taker=%+.4f%%   maker=%+.4f%%' % (
        100*(y.mean()-fee_t), 100*(y.mean()-fee_m)))
    say('  胜率=%.1f%%   最大单笔=%+.2f%%   最差单笔=%+.2f%%' % (
        100*(y > 0).mean(), 100*y.max(), 100*y.min()))

# 事件时间分布
say('\n=== 事件的时间分布（是否集中在少数几天）===')
ev['day'] = pd.to_datetime(ev.t, unit='ms', utc=True).dt.date
byday = ev.groupby('day').size().sort_values(ascending=False)
say('  覆盖天数=%d / 总天数=%d' % (len(byday), (pd.to_datetime(ev.t.max(), unit='ms') - pd.to_datetime(ev.t.min(), unit='ms')).days + 1))
say('  事件最多的 5 天: ' + ', '.join('%s:%d' % (k, v) for k, v in byday.head(5).items()))

# 币分布
syms = ev.groupby('sym').size().sort_values(ascending=False)
say('\n=== 事件涉及的币（top10）===')
say('  ' + ', '.join('%s:%d' % (k, v) for k, v in syms.head(10).items()))
say('  涉及币数=%d / %d' % (len(syms), px.shape[1]))

with open(os.path.join(EV, 'bigwin_events.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.1fs' % (time.time() - t0))
