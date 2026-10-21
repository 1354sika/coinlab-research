# -*- coding: utf-8 -*-
"""从「先挖后验」改成「先想后验」：给机制写出因果假设，再逐条检验推论
机制（唯一通过两把尺子的）：极端急跌 + 巨量 -> 60 分钟反弹
因果假设 H0：这是【强平/清算踩踏后的流动性回补】，不是"超卖反弹"这种含糊说法。
若 H0 为真，必须能推出以下【事前可检验】的推论（全部用现有 365 天数据）：
  T1 系统性去杠杆（多币同时被砸）应比单币事件更强
  T2 在市场整体暴跌背景下应更强（全市场去杠杆 vs 个股事件）
  T3 跌得越快（段内速率高）应更强 —— 强平是加速的
  T4 反弹应在很短时间内发生（流动性回补快）—— 已知 1 分钟为负、3 分钟转正
  T5 与"跌得深"应是两件不同的事：控制住速度后，深度本身应不再重要
反面推论（若成立则 H0 被削弱）：
  N1 若"多币同时被砸"反而更弱 -> 说明是普跌拖累，不是强平回补
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

DQ, VOLR, DEP, FEE_M = 0.90, 10.0, 0.05, 0.0002
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
mkt_r = px.pct_change().median(axis=1).values          # 市场分钟收益
mkt15 = pd.Series(mkt_r).rolling(15).sum().values      # 市场 15 分钟累积
T, N = P.shape
t = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
speed = streak_ret / seglen                          # 每分钟平均（负=下跌速率）
volr = QV / qv60
FWD = {}
for h in (3, 5, 15, 60):
    f = np.full((T, N), np.nan)
    f[:-h] = P[h:] / P[:-h] - 1.0
    FWD[h] = f
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
SIG = (streak_ret < -DEP) & (volr > VOLR) & sel
say('信号（分钟x币，未去重）= %d，涉及分钟数 = %d' % (
    int(SIG.sum()), int((SIG.sum(axis=1) > 0).sum())))

# 同时触发信号的其他币数（系统性去杠杆代理）
concur = SIG.sum(axis=1).astype(np.float64)[:, None] * np.ones((1, N))
MKT15 = mkt15[:, None] * np.ones((1, N))


def tab(name, F, h=60, bins=((-1e9, 1.5, '0-1个'), (1.5, 3.5, '2-3个'), (3.5, 7.5, '4-7个'),
                             (7.5, 1e9, '8+个'))):
    y = FWD[h][SIG]
    f = F[SIG]
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    if len(y) < 50:
        say('  [%s] 太少' % name); return
    say('\n--- %s（未来 %d 分钟）---' % (name, h))
    say('  %-14s %7s %11s %11s %11s' % ('档', 'n', '均值%', '中位%', 'P(涨)%'))
    base = y.mean()
    for lo, hi, lb in bins:
        m = (f >= lo) & (f < hi)
        if m.sum() < 15:
            continue
        yy = y[m]
        say('  %-14s %7d %10.3f%% %10.3f%% %10.1f%%' % (
            lb, m.sum(), 100*yy.mean(), 100*np.median(yy), 100*(yy > 0).mean()))
    say('  基准（全信号）: n=%d 均值=%+.3f%%' % (len(y), 100*base))


say('\n' + '=' * 78)
say('【T1】系统性去杠杆：同时触发信号的币数越多，反弹应该越强？')
tab('同分钟触发数', concur)
say('\n【T2】市场背景：市场 15 分钟越跌，反弹应该越强？')
tab('市场15m', MKT15, bins=((-1e9, -0.02, '跌>2%'), (-0.02, -0.005, '跌0.5-2%'),
                            (-0.005, 0.005, '±0.5%'), (0.005, 1e9, '涨>0.5%')))

say('\n【T3】下跌速率：跌得越快，反弹应该越强？')
tab('段内速率', speed, bins=((-1e9, -0.03, '>3%/分'), (-0.03, -0.02, '2-3%/分'),
                             (-0.02, -0.01, '1-2%/分'), (-0.01, 0, '<1%/分')))

say('\n【T4】反弹的时间结构：应该很快（流动性回补）')
y3 = FWD[3][SIG]; y5 = FWD[5][SIG]; y15 = FWD[15][SIG]; y60 = FWD[60][SIG]
for lb, y in (('3 分钟', y3), ('5 分钟', y5), ('15 分钟', y15), ('60 分钟', y60)):
    y = y[np.isfinite(y)]
    say('  %-8s n=%4d  均值=%+.3f%%  中位=%+.3f%%  P(涨)=%.1f%%' % (
        lb, len(y), 100*y.mean(), 100*np.median(y), 100*(y > 0).mean()))

say('\n【T5】控制速率后，深度本身是否还有增量？（二因素）')
say('  %-12s' % '速率\\深度' + ''.join('%22s' % d for d in ('跌5-8%', '跌8-12%', '跌>12%')))
for lo, hi, lb in ((-1e9, -0.03, '>3%/分'), (-0.03, -0.02, '2-3%/分'),
                   (-0.02, -0.01, '1-2%/分'), (-0.01, 0, '<1%/分')):
    row = '  %-12s' % lb
    for dlo, dhi in ((-0.08, -0.05), (-0.12, -0.08), (-1e9, -0.12)):
        m = (speed >= lo) & (speed < hi) & (streak_ret >= dlo) & (streak_ret < dhi) & SIG
        y = FWD[60][m]; y = y[np.isfinite(y)]
        row += '%22s' % ('n=%d %+.2f%%' % (len(y), 100*y.mean()) if len(y) >= 15 else '-')
    say(row)

say('\n【N1】反面检验：若"多币同时被砸"更弱，则 H0（强平回补）被削弱')
c = concur[SIG]; y = FWD[60][SIG]
ok = np.isfinite(c) & np.isfinite(y)
if ok.sum() > 30:
    say('  同分钟触发数 与 未来60m收益 的相关 = %+.3f' % np.corrcoef(c[ok], y[ok])[0, 1])

with open(os.path.join(EV, 'hypothesis_test.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
