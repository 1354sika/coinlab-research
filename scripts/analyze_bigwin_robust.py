# -*- coding: utf-8 -*-
"""对爆发点信号做稳健性检验：剔除极端日、按天等权、剔除热门币、随机抽半。
信号：分化时刻(disp>P90) 且 R15<-5% 且 VolRatio>2 -> 做多，持有 5 分钟。
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
r = px.pct_change()
disp = r.std(axis=1)
thr = float(disp.quantile(0.90))
selrow = (disp > thr).values
R15 = px / px.shift(15) - 1.0
VR  = qv / qv.rolling(60).mean()
FWD = px.shift(-5) / px - 1.0

cond = (R15 < -0.05) & (VR > 2.0)
cond = cond & pd.DataFrame(np.repeat(selrow[:, None], px.shape[1], axis=1),
                           index=px.index, columns=px.columns)
times = px.index.values
mask, fwd, r15v = cond.values, FWD.values, R15.values
ev = []
for j in range(mask.shape[1]):
    col = mask[:, j]
    if not col.any():
        continue
    ii = np.flatnonzero(col)
    starts = np.r_[0, np.flatnonzero(np.diff(ii) > 5) + 1]
    for s in starts:
        k = ii[s]
        if np.isfinite(fwd[k, j]):
            ev.append((times[k], str(px.columns[j]), r15v[k, j], fwd[k, j]))
ev = pd.DataFrame(ev, columns=['t', 'sym', 'r15', 'fwd5'])
ev['day'] = pd.to_datetime(ev.t, unit='ms', utc=True).dt.date
ev.to_csv(os.path.join(EV, 'bigwin_events.csv'), index=False, encoding='utf-8')
say('事件数=%d  存 evidence/bigwin_events.csv' % len(ev))

fee_t, fee_m = 0.0020, 0.0004
def rep(nm, y):
    if len(y) < 50:
        say('  %-24s n=%d (太少)' % (nm, len(y))); return
    say('  %-24s n=%6d  均值=%+.4f%%  中位=%+.4f%%  胜率=%.1f%%  净taker=%+.4f%%  净maker=%+.4f%%' % (
        nm, len(y), 100*y.mean(), 100*np.median(y), 100*(y > 0).mean(),
        100*(y.mean()-fee_t), 100*(y.mean()-fee_m)))

say('\n=== 稳健性检验（信号: 分化+R15<-5%+量比>2, 持5分钟）===')
rep('全样本', ev.fwd5.values)

top = ev.groupby('day').size().sort_values(ascending=False)
say('\n  事件最多的天数: ' + ', '.join('%s:%d' % (k, v) for k, v in top.head(8).items()))
for k in (1, 3, 5, 10):
    drop = set(top.head(k).index)
    rep('剔除 top%d 个事件日' % k, ev[~ev.day.isin(drop)].fwd5.values)

say('\n=== 按天等权（每天先算均值，再对天求平均）===')
daily = ev.groupby('day').fwd5.mean()
say('  天数=%d  日均值的中位=%+.4f%%  日均值的均值=%+.4f%%  正收益天占比=%.1f%%' % (
    len(daily), 100*daily.median(), 100*daily.mean(), 100*(daily > 0).mean()))

say('\n=== 剔除事件最多的币 ===')
topsym = ev.groupby('sym').size().sort_values(ascending=False)
for k in (5, 10, 20):
    drop = set(topsym.head(k).index)
    rep('剔除 top%d 个币' % k, ev[~ev.sym.isin(drop)].fwd5.values)

say('\n=== 随机抽一半（10 次）===')
rs = np.random.RandomState(42)
mus = [ev.fwd5.sample(frac=0.5, random_state=rs).mean() for _ in range(10)]
say('  10 次抽半均值: %s' % ' '.join('%+.3f%%' % (100*m) for m in mus))
say('  抽半均值范围: %+.4f%% ~ %+.4f%%（全为负则不稳）' % (100*min(mus), 100*max(mus)))

say('\n=== 对比：不在分化时刻的同样条件 ===')
cond_all = (R15 < -0.05) & (VR > 2.0)
non_sel = (~pd.Series(selrow, index=px.index)).values
cond_ns = cond_all & pd.DataFrame(np.repeat(non_sel[:, None], px.shape[1], axis=1),
                                  index=px.index, columns=px.columns)
m2, f2 = cond_ns.values, FWD.values
yy = []
for j in range(m2.shape[1]):
    ii = np.flatnonzero(m2[:, j])
    for k in ii:
        if np.isfinite(f2[k, j]):
            yy.append(f2[k, j])
rep('非分化时刻同样条件', np.array(yy))

with open(os.path.join(EV, 'bigwin_robust.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.1fs' % (time.time() - t0))
