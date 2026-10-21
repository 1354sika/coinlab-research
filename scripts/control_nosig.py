# -*- coding: utf-8 -*-
"""回答用户三问的量化部分（365 天 / 60 币）
问① 短期与长期策略是否相反 -> 按月拆开，看两个方向如何随行情切换
问② 没被信号覆盖的时刻（"预判不了"的样本）市场在干什么 -> 有信号 vs 无信号 vs 全体
做法：
  S1 有信号 = 分化(disp>P90) & |连续段幅度|>=5% & 量比>10（双向都算）
  S2 无信号 = 其余所有「分钟x币」
  S3 全体
  对每个集合，统计未来 60 分钟收益：均值、中位、P(涨)、以及按月走势
  另：把 S1 拆成「跌型」和「涨型」，看它们的未来收益是否互为镜像
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

HOLD, DQ, VOLR, DEP = 60, 0.90, 10.0, 0.05
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
say('matrix %d x %d (%.0f 天)  %.0fs' % (T, N, (t[-1]-t[0])/86400000, time.time()-t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
volr = QV / qv60
FWD60 = np.full((T, N), np.nan)
FWD60[:-60] = P[60:] / P[:-60] - 1.0
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)

DOWN = (streak_ret < -DEP) & (volr > VOLR) & sel
UP = (streak_ret > +DEP) & (volr > VOLR) & sel
SIG = DOWN | UP
NOSIG = np.isfinite(FWD60) & (~SIG)
ALL = np.isfinite(FWD60)


def desc(mask, tag):
    y = FWD60[mask]; y = y[np.isfinite(y)]
    if len(y) < 100:
        say('  %-26s n=%d 太少' % (tag, len(y))); return
    say('  %-26s n=%9d  均值=%+.4f%%  中位=%+.4f%%  P(涨)=%.2f%%  P(跌>1%%)=%.2f%%' % (
        tag, len(y), 100*y.mean(), 100*np.median(y), 100*(y > 0).mean(), 100*(y < -0.01).mean()))


say('\n=== ① 有信号 vs 无信号 vs 全体（未来 60 分钟）===')
desc(DOWN, 'S1a 有信号·跌型（做多候选）')
desc(UP, 'S1b 有信号·涨型（做空候选）')
desc(SIG, 'S1 有信号（合计）')
desc(NOSIG, 'S2 无信号（其余全部）')
desc(ALL, 'S3 全体')

say('\n=== ② 按月：有信号 vs 无信号 vs 全体（均值 %）===')
idx = pd.to_datetime(t, unit='ms', utc=True)
mo = idx.to_period('M')
say('  %-9s %10s %10s %10s %10s %10s' % ('月份', '信号·跌', '信号·涨', '有信号', '无信号', '全体'))
for p in mo.unique():
    m = np.asarray(mo == p)
    def mu(mask):
        v = FWD60[mask & np.isfinite(FWD60)]
        return 100*v.mean() if len(v) >= 30 else np.nan
    say('  %-9s %9.4f%% %9.4f%% %9.4f%% %9.4f%% %9.4f%%' % (
        str(p), mu(DOWN), mu(UP), mu(SIG), mu(NOSIG), mu(ALL)))

say('\n=== ③ 「无信号」样本自身按未来方向分解（回答"预判不了的币涨了还是跌了"）===')
y = FWD60[NOSIG]; y = y[np.isfinite(y)]
say('  无信号样本 n=%d' % len(y))
for lo, hi, lb in [(-99, -0.05, '<-5%'), (-0.05, -0.02, '-5~-2%'), (-0.02, -0.005, '-2~-0.5%'),
                   (-0.005, 0.005, '±0.5%'), (0.005, 0.02, '+0.5~2%'),
                   (0.02, 0.05, '+2~5%'), (0.05, 99, '>+5%')]:
    m = (y >= lo) & (y < hi)
    say('    未来60m %-10s 占比=%5.2f%%  (n=%d)' % (lb, 100*m.mean(), int(m.sum())))
say('  -> 无信号样本里 涨的有 %.2f%%，跌的有 %.2f%%' % (100*(y > 0).mean(), 100*(y < 0).mean()))

say('\n=== ④ 两个方向的镜像性检验（是否只是"市场整体涨跌"的翻版）===')
say('  如果市场整体上涨，跌型信号应该更容易出现；这里看两者的时间占比相关性')
dcount = DOWN.sum(axis=1).astype(float)
ucount = UP.sum(axis=1).astype(float)
ok = np.isfinite(dcount) & np.isfinite(ucount)
say('  每分钟触发数：跌型 均值=%.3f（有信号的分钟占 %.1f%%）；涨型 均值=%.3f（占 %.1f%%）' % (
    dcount[ok].mean(), 100*(dcount[ok] > 0).mean(), ucount[ok].mean(), 100*(ucount[ok] > 0).mean()))
say('  同日两者相关 = %.3f' % np.corrcoef(dcount[ok], ucount[ok])[0, 1])

with open(os.path.join(EV, 'control_nosig.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
