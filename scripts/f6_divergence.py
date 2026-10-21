# -*- coding: utf-8 -*-
"""F6 验证：量价背离度 —— 唯一「只需 K 线」的操盘手指纹
定义：价格接近 60 分钟新高 + 量能萎缩（背离）→ 出货嫌疑
验证：该条件下未来 5/15/60 分钟的期望收益（做空视角 / 做多视角）
对照：同口径下的无条件基准
"""
import os, numpy as np, pandas as pd

DATA = r'D:\中信证券\coinlab\data'
FEE_M = 0.0002          # maker 单边
FEE_T = 0.0010

def say(*a): print(*a, flush=True)

say('=' * 100)
say('F6 验证：量价背离度（价格近新高 + 量能萎缩）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'])
df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=5)
qv    = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
T, N = close.shape
say('matrix %d x %d' % (T, N))

# ---- 指纹 ----
hi60  = close.rolling(60, min_periods=30).max()
near_hi = (close / hi60).astype('float32')          # 越接近 1 越靠近 60 分钟高点
v5    = qv.rolling(5,  min_periods=3).mean()
v60   = qv.rolling(60, min_periods=30).mean()
volr  = (v5 / v60).astype('float32')                # <1 = 量能萎缩

# ---- 未来收益 ----
FW = {5: None, 15: None, 60: None}
for H in FW:
    FW[H] = (close.shift(-H) / close - 1.0).astype('float32')

say('\n【一】基准（无条件）：未来 5/15/60 分钟的平均收益')
say('=' * 100)
for H in sorted(FW):
    r = FW[H].values.ravel()
    r = r[np.isfinite(r)]
    say('  未来 %2d 分钟：均值 %+.4f%%  中位 %+.4f%%  P(涨) %.2f%%'
        % (H, r.mean() * 100, np.median(r) * 100, (r > 0).mean() * 100))

say('\n【二】F6 背离条件：近 60 分钟高点（>=P） 且 量能萎缩（volr <= V）')
say('=' * 100)
say('  条件                                       样本     未来5m均   未来15m均   未来60m均')
nh = near_hi.values.ravel()
vr = volr.values.ravel()
for P, V in [(0.999, 0.8), (0.999, 1.0), (0.995, 0.8), (0.98, 0.8), (0.999, 1.2)]:
    m = (nh >= P) & (vr <= V) & np.isfinite(nh) & np.isfinite(vr)
    if m.sum() < 500:
        say('  近高点>=%.3f & volr<=%.1f                     样本不足' % (P, V))
        continue
    row = []
    for H in sorted(FW):
        r = FW[H].values.ravel()
        mm = m & np.isfinite(r)
        row.append(r[mm].mean() * 100 if mm.sum() > 100 else float('nan'))
    say('  近60m高点>=%.3f & volr<=%.1f        %9s  %+8.4f%%  %+9.4f%%  %+10.4f%%'
        % (P, V, format(int(m.sum()), ','), row[0], row[1], row[2]))

say('\n【三】反向对照：近高点 + 量能放大（volr >= 1.5）—— 若是真突破，应比背离好')
say('=' * 100)
for P, V in [(0.999, 1.5), (0.999, 2.0), (0.999, 3.0)]:
    m = (nh >= P) & (vr >= V) & np.isfinite(nh) & np.isfinite(vr)
    if m.sum() < 500: continue
    row = []
    for H in sorted(FW):
        r = FW[H].values.ravel()
        mm = m & np.isfinite(r)
        row.append(r[mm].mean() * 100 if mm.sum() > 100 else float('nan'))
    say('  近60m高点>=%.3f & volr>=%.1f        %9s  %+8.4f%%  %+9.4f%%  %+10.4f%%'
        % (P, V, format(int(m.sum()), ','), row[0], row[1], row[2]))

say('\n【四】做空口径（收益取负，扣 maker 双边 %.2f%%）' % (2 * FEE_M * 100))
say('=' * 100)
say('  条件                                       样本   空5m净    空15m净    空60m净')
for P, V in [(0.999, 0.8), (0.999, 1.0), (0.995, 0.8), (0.98, 0.8)]:
    m = (nh >= P) & (vr <= V) & np.isfinite(nh) & np.isfinite(vr)
    if m.sum() < 500: continue
    row = []
    for H in sorted(FW):
        r = FW[H].values.ravel()
        mm = m & np.isfinite(r)
        row.append(-r[mm].mean() * 100 - 2 * FEE_M * 100 if mm.sum() > 100 else float('nan'))
    say('  近60m高点>=%.3f & volr<=%.1f        %9s  %+7.4f%%  %+8.4f%%  %+9.4f%%'
        % (P, V, format(int(m.sum()), ','), row[0], row[1], row[2]))

say('\n【五】分月稳定性（最优背离档，做空 60m 净收益）')
say('=' * 100)
m = (nh >= 0.999) & (vr <= 1.0) & np.isfinite(nh) & np.isfinite(vr)
r60 = FW[60].values.ravel()
ok = m & np.isfinite(r60)
if ok.sum() > 1000:
    dts = np.repeat(close.index.values, N)[ok]
    mo = pd.Series(-r60[ok] * 100 - 2 * FEE_M * 100, index=pd.DatetimeIndex(dts)).groupby(
        pd.DatetimeIndex(dts).strftime('%Y-%m')).agg(['count', 'mean', 'sum'])
    pos = 0
    for k, row in mo.iterrows():
        if row['mean'] > 0: pos += 1
        say('  %s  n=%6d  均值 %+.4f%%  月合计 %+7.3f%%' % (k, int(row['count']), row['mean'], row['sum']))
    say('  -> 正收益月 = %d/%d' % (pos, len(mo)))

say('\nTOTAL done')
