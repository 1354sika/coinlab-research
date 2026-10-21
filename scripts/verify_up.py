# -*- coding: utf-8 -*-
"""预判涨-2：对扫描出的最强候选做「事件级去重复核 + 分时期稳定性」
候选（来自 scan_up）：
  低位 + 远离均线：R60 处于最低 40% 且 MA60dev 处于最负的 20%  -> 做多，持 60 分钟
检查三件事：
  ① 去重前 vs 事件级去重后（防样本重叠虚高——前面踩过的坑）
  ② 6 个时期各自表现（是不是只靠某一段行情）
  ③ 门槛敏感性：R60 分位 / MA60dev 分位 各自扫描，看是否单调
另对照：同样的条件在「高位 + 远离均线上方」时的反向表现
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, FEE_M, FEE_T, K = 60, 0.0002, 0.0010, 6
t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
del df
P = px.values.astype(np.float64)
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

R60 = (px / px.shift(60) - 1.0).values.astype(np.float64)
MA60 = px.rolling(60).mean().values
madev = P / MA60 - 1.0
FWD60 = np.full((T, N), np.nan)
FWD60[:-60] = P[60:] / P[:-60] - 1.0
mkt = px.pct_change(60).median(axis=1).values
MKT = np.repeat(mkt[:, None], N, axis=1)
edges = np.linspace(0, T, K + 1).astype(int)

qs60 = np.nanquantile(R60[np.isfinite(R60)], [0.2, 0.4, 0.6, 0.8])
qm = np.nanquantile(madev[np.isfinite(madev)], [0.2, 0.4, 0.6, 0.8])
say('R60 分位: ' + ' '.join('%.4f' % x for x in qs60))
say('MA60dev 分位: ' + ' '.join('%.4f' % x for x in qm))


def stats(sig, tag):
    y = FWD60[sig]; y = y[np.isfinite(y)]
    mr = MKT[sig]; mr = mr[np.isfinite(mr)]
    if len(y) < 200:
        say('  %-34s n=%d 太少' % (tag, len(y))); return
    rel = y[:len(mr)] - mr[:len(y)]
    say('  %-34s n=%6d  P(涨)=%5.2f%%  P(跑赢)=%5.2f%%  均值=%+.4f%%  净maker=%+.4f%%  净taker=%+.4f%%' % (
        tag, len(y), 100*(y > 0).mean(), 100*(rel > 0).mean(),
        100*y.mean(), 100*(y.mean()-FEE_M), 100*(y.mean()-FEE_T)))


say('\n=== 候选信号：R60 最低40% 且 MA60dev 最负20%（做多，持60m）===')
LOW = R60 <= qs60[1]
DEEP = madev <= qm[0]
SIG = LOW & DEEP
say('  信号（分钟x币，未去重）= %d' % int(SIG.sum()))
stats(SIG, '① 全样本（未去重）')

# 事件级去重
opens = []
for j in range(N):
    ii = np.flatnonzero(SIG[:, j])
    if len(ii) == 0:
        continue
    last = -1
    for k in ii:
        if k <= last:
            continue
        if np.isfinite(FWD60[k, j]):
            opens.append((k, j))
        last = k + HOLD - 1
opens = np.array(opens, int)
mo = np.zeros((T, N), bool)
for k, j in opens:
    mo[k, j] = True
say('  信号（事件级去重）= %d' % len(opens))
stats(mo, '② 事件级去重后')

say('\n=== ③ 逐期（事件级去重口径）===')
per = []
for p in range(K):
    a, b = edges[p], edges[p + 1]
    seg = np.zeros((T, N), bool)
    seg[a:b] = mo[a:b]
    y = FWD60[seg]; y = y[np.isfinite(y)]
    if len(y) == 0:
        continue
    nav = np.prod(1 + (y - FEE_M))
    d0 = pd.to_datetime(t[a], unit='ms', utc=True).strftime('%m-%d')
    d1 = pd.to_datetime(t[min(b, T - 1)], unit='ms', utc=True).strftime('%m-%d')
    say('  期%d %s~%s  n=%5d  毛均值=%+.4f%%  等权净值=%+.2f%%  胜率=%.1f%%' % (
        p+1, d0, d1, len(y), 100*y.mean(), 100*(nav-1), 100*(y > 0).mean()))
    per.append(y.mean())
per = np.array(per)
say('  → 正收益期 %d/%d' % ((per > 0).sum(), len(per)))

say('\n=== ④ 门槛敏感性：MA60dev 分位 x R60 分位（净maker%）===')
say('  %-16s' % 'MA\\R60' + ''.join('%14s' % ('<%d%%' % int(q*100)) for q in (0.2, 0.4, 0.6, 0.8)))
mpcts = [0.05, 0.10, 0.20, 0.40]
rpcts = [0.20, 0.40, 0.60, 0.80]
for mp in mpcts:
    thm = np.nanquantile(madev[np.isfinite(madev)], mp)
    row = '  ma<%3.0f%%(=%.3f)' % (mp*100, thm)
    for rp in rpcts:
        thr = np.nanquantile(R60[np.isfinite(R60)], rp)
        s = (madev <= thm) & (R60 <= thr)
        y = FWD60[s]; y = y[np.isfinite(y)]
        row += '%14s' % ('%+.3f%%(n%d)' % (100*(y.mean()-FEE_M), len(y)) if len(y) > 200 else '-')
    say(row)

say('\n=== ⑤ 反向对照：R60 最高20% 且 MA60dev 最正20%（做空视角，收益已取负）===')
HIGH = R60 >= qs60[3]
UP = madev >= qm[3]
SIGR = HIGH & UP
say('  信号（未去重）= %d' % int(SIGR.sum()))
stats(SIGR, '高位+远离上方（做多=顺势）')
SIGR2 = SIGR.copy()
stats(SIGR2, '同上（做空需取负，见均值符号）')

with open(os.path.join(EV, 'verify_up.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
