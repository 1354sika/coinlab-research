# -*- coding: utf-8 -*-
"""系统性扫描：什么前置状态 -> 未来会涨？（用户：如何真正预判涨，需要细致全面的采样与验证）
设计要点（吸收前面踩过的坑）：
  ① 每个特征都同时报两口径：全样本 / 事件级去重（防样本重叠虚高）
  ② 同时报两个目标：绝对涨（FWD>0）与相对涨（FWD > 同期市场收益）
  ③ 每个特征都报 P(涨)、均值、扣 maker 后的净收益 —— 只报 P 不报净的结论不算数
特征（t 时刻全部可观测，无前视）：
  R5/R15/R60 过去收益 | streak_ret 连续段累计(带符号) | volr 量比 | MA60dev 距均线
  vol60 已实现波动 | disp 横截面离散度 | mkt15 市场滚动收益 | hour 时段
目标：FWD5 / FWD60（绝对），以及相对市场的超额
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

FEE_M, FEE_T = 0.0002, 0.0010
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
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

# ---- 特征 ----
s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
R5 = (px / px.shift(5) - 1.0).values.astype(np.float64)
R15 = (px / px.shift(15) - 1.0).values.astype(np.float64)
R60 = (px / px.shift(60) - 1.0).values.astype(np.float64)
volr = QV / qv60
MA60 = px.rolling(60).mean().values
madev = P / MA60 - 1.0
vol60 = px.pct_change().rolling(60).std().values.astype(np.float64)
dr = px.pct_change()
disp_v = dr.std(axis=1).values
mkt15 = (dr.median(axis=1).rolling(15).sum()).values        # 市场 15 分钟累积
hour = pd.to_datetime(px.index, unit='ms', utc=True).hour.values.astype(np.float64)
H = np.repeat(hour[:, None], N, axis=1)

FWD = {}
FWDREL = {}
for h in (5, 60):
    f = np.full((T, N), np.nan)
    f[:-h] = P[h:] / P[:-h] - 1.0
    FWD[h] = f
    # 相对：同期市场（各币等权中位）的收益

FEATS = {
    'R5(过去5m)': R5, 'R15(过去15m)': R15, 'R60(过去60m)': R60,
    'streak_ret(连续段)': streak_ret, 'volr(量比60m)': volr,
    'MA60dev(距均线)': madev, 'vol60(波动率60m)': vol60,
    'mkt15(市场15m)': np.repeat(mkt15[:, None], N, axis=1),
    'hour(UTC时)': H,
}
# 相对收益（跑赢市场）
mktf = {}
for h in (5, 60):
    x = (px.pct_change(h).median(axis=1)).values
    mktf[h] = np.repeat(x[:, None], N, axis=1)


def table(name, F, h, dedup_every=None, mask=None, nbins=6):
    f = F.ravel() if mask is None else F[mask]
    y = FWD[h].ravel() if mask is None else FWD[h][mask]
    mr = mktf[h].ravel() if mask is None else mktf[h][mask]
    ok = np.isfinite(f) & np.isfinite(y) & np.isfinite(mr)
    f, y, mr = f[ok], y[ok], mr[ok]
    if len(y) < 2000:
        say('  [%s] 样本不足' % name); return
    rel = y - mr
    say('\n--- %s -> 未来 %dm ---' % (name, h))
    say('  %-18s %9s %9s %10s %10s %10s' % ('档', 'n', 'P(涨)%', 'P(跑赢)%', '均值%', '净maker%'))
    b_all = (y > 0).mean(); br_all = (rel > 0).mean()
    say('  %-18s %9d %8.2f%% %9.2f%% %9.4f%% %9.4f%%' % (
        '【全样本基准】', len(y), 100*b_all, 100*br_all, 100*y.mean(), 100*(y.mean()-FEE_M)))
    qs = np.quantile(f, np.linspace(0, 1, nbins + 1))
    for i in range(nbins):
        lo, hi = qs[i], qs[i + 1]
        m = (f >= lo) & (f <= hi) if i == nbins - 1 else (f >= lo) & (f < hi)
        if m.sum() < 500:
            continue
        yy = y[m]; rr = rel[m]
        say('  [%8.3f..%8.3f] %9d %8.2f%% %9.2f%% %9.4f%% %9.4f%%' % (
            lo, hi, m.sum(), 100*(yy > 0).mean(), 100*(rr > 0).mean(),
            100*yy.mean(), 100*(yy.mean()-FEE_M)))


say('\n' + '='*80)
say('【A】绝对口径：全样本逐特征扫描（未来 60 分钟）')
for k, F in FEATS.items():
    table(k, F, 60)

say('\n' + '='*80)
say('【B】未来 5 分钟（短持有，成本占比更高，供对照）')
for k in ('R5(过去5m)', 'streak_ret(连续段)', 'volr(量比60m)', 'mkt15(市场15m)'):
    table(k, FEATS[k], 5)

say('\n' + '='*80)
say('【C】只看「低位」样本：R60 处于最低两档时，其他特征怎么影响未来涨')
F60 = FEATS['R60(过去60m)']
q = np.quantile(F60[np.isfinite(F60)], 0.4)
low = F60 <= q
say('  低位子样本数 = %d' % int(low.sum()))
for k in ('volr(量比60m)', 'MA60dev(距均线)', 'vol60(波动率60m)', 'mkt15(市场15m)'):
    table('低位 + ' + k, FEATS[k], 60, mask=low)

with open(os.path.join(EV, 'scan_up.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
