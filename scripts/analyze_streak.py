# -*- coding: utf-8 -*-
"""持续性与方向：连涨/连跌的时长与累计幅度，能否给出【方向】？
用户三点：
  Q1 「某个时间单位的持续涨 / 持续跌」—— 连续同向的时长 streak，之前未测过
  Q2 「这一单位或连续这一时间单位涨跌了多少」—— 连续段累计幅度 streak_ret
  Q3 「之前只能算绝对值，算不出方向」—— 本脚本直接统计【方向命中率】P(未来涨)
之前测过的：R_5m / R_15m / R_60m（滚动收益 = 涨跌多少）已测；
            「连续涨了几分钟」与「方向命中率」未测 —— 本脚本补上。
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

FEE_T, FEE_M = 0.0020, 0.0004

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

P = px.values
r = px.pct_change()
R = r.values
disp_v = r.std(axis=1).values
VR = (qv / qv.rolling(60).mean()).values
HOLD = (5, 15, 60)
FWD = {h: (px.shift(-h) / px - 1.0).values for h in HOLD}
T, N = P.shape

# ---- 连续性：连涨/连跌几分钟 + 该连续段累计幅度 ----
s = np.sign(R)
s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
streak_len = rows - start + 1
streak = (streak_len * s).astype(np.float32)                 # +3 = 连涨3分钟
p_start = np.take_along_axis(P, start, axis=0)
streak_ret = (P / p_start - 1.0).astype(np.float32)          # 连续段累计幅度
say('streak 分布: 连涨1=%d 连涨2=%d 连涨3=%d 连跌1=%d 连跌2=%d 连跌3=%d' % (
    (streak == 1).sum(), (streak == 2).sum(), (streak == 3).sum(),
    (streak == -1).sum(), (streak == -2).sum(), (streak == -3).sum()))


def tab(name, feat, mask, h, bins=None, labels=None):
    f = feat[mask]; y = FWD[h][mask]
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    if len(y) < 500:
        say('  [%s] 样本不足' % name); return None
    base = (y > 0).mean()
    say('\n--- %s (未来 %d 分钟方向) ---' % (name, h))
    say('  基准: P(涨)=%.2f%%  P(跌)=%.2f%%  均值=%+.4f%%  n=%d' % (
        100*base, 100*(1-base), 100*y.mean(), len(y)))
    out = []
    for lo, hi, lb in bins:
        m = (f >= lo) & (f < hi)
        if m.sum() < 300:
            continue
        yy = y[m]
        pu = (yy > 0).mean()
        say('  %-16s n=%7d  P(涨)=%5.2f%%(%+5.2fpp)  均值=%+.4f%%  净maker=%+.4f%%  净taker=%+.4f%%' % (
            lb, m.sum(), 100*pu, 100*(pu-base), 100*yy.mean(),
            100*(yy.mean()-FEE_M), 100*(yy.mean()-FEE_T)))
        out.append((lb, m.sum(), 100*pu, 100*yy.mean()))
    return base, out


ALL = np.ones((T, N), bool)
say('\n' + '='*74)
say('【A】连涨/连跌时长 streak -> 方向（全样本）')
B1 = [(-99, -10, '连跌>=10'), (-10, -5, '连跌5-9'), (-5, -2, '连跌2-4'),
      (-2, -1, '连跌1'), (-1, 0.5, '持平'), (0.5, 2, '连涨1'),
      (2, 5, '连涨2-4'), (5, 10, '连涨5-9'), (10, 99, '连涨>=10')]
for h in HOLD:
    tab('A-streak h=%dm' % h, streak, ALL, h, B1)

say('\n' + '='*74)
say('【B】连续段累计幅度 streak_ret -> 方向（全样本）')
B2 = [(-99, -0.05, '累计跌>5%'), (-0.05, -0.02, '跌2-5%'), (-0.02, -0.005, '跌0.5-2%'),
      (-0.005, 0, '跌<0.5%'), (0, 0.005, '涨<0.5%'), (0.005, 0.02, '涨0.5-2%'),
      (0.02, 0.05, '涨2-5%'), (0.05, 99, '累计涨>5%')]
for h in HOLD:
    tab('B-streak_ret h=%dm' % h, streak_ret, ALL, h, B2)

say('\n' + '='*74)
say('【C】限定在分化时刻（disp>P90）')
d90 = np.nanpercentile(disp_v, 90)
mkd = (disp_v > d90)[:, None] & np.ones((1, N), bool)
for h in (15, 60):
    tab('C-streak 分化 h=%dm' % h, streak, mkd, h, B1)
    tab('C-streak_ret 分化 h=%dm' % h, streak_ret, mkd, h, B2)

say('\n' + '='*74)
say('【D】已确认的爆发点信号内，用 streak 细分方向（分化+急跌5%+放量2）')
R15 = (px / px.shift(15) - 1.0).values
sig = (R15 < -0.05) & (VR > 2.0) & mkd
say('  信号样本数（分钟x币）= %d' % int(sig.sum()))
for h in (5, 15, 60):
    y = FWD[h][sig]; y = y[np.isfinite(y)]
    if len(y) < 200:
        continue
    say('  信号整体 h=%dm: P(涨)=%.2f%%  均值=%+.4f%%  净maker=%+.4f%%  n=%d' % (
        h, 100*(y > 0).mean(), 100*y.mean(), 100*(y.mean()-FEE_M), len(y)))
tab('D-信号内 streak h=60m', streak, sig, 60, B1)
tab('D-信号内 streak_ret h=60m', streak_ret, sig, 60, B2)
tab('D-信号内 streak h=5m', streak, sig, 5, B1)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
lab, hi_streak = [], []
for lo, hh, lb in B1:
    m = (streak >= lo) & (streak < hh) & np.isfinite(FWD[15])
    if m.sum() < 300:
        continue
    y = FWD[15][m]; y = y[np.isfinite(y)]
    lab.append(lb); hi_streak.append(100*(y > 0).mean())
axs[0].bar(range(len(lab)), hi_streak)
axs[0].axhline(50, ls='--', lw=1, color='gray')
axs[0].set_xticks(range(len(lab))); axs[0].set_xticklabels(lab, rotation=45, fontsize=8)
axs[0].set_title('streak -> P(up) next 15m')
axs[0].set_ylabel('P(up) %')

lab2, hi2 = [], []
for lo, hh, lb in B2:
    m = (streak_ret >= lo) & (streak_ret < hh) & np.isfinite(FWD[15])
    if m.sum() < 300:
        continue
    y = FWD[15][m]; y = y[np.isfinite(y)]
    lab2.append(lb); hi2.append(100*(y > 0).mean())
axs[1].bar(range(len(lab2)), hi2)
axs[1].axhline(50, ls='--', lw=1, color='gray')
axs[1].set_xticks(range(len(lab2))); axs[1].set_xticklabels(lab2, rotation=45, fontsize=8)
axs[1].set_title('streak_ret -> P(up) next 15m')

mu = []
for lo, hh, lb in B2:
    m = (streak_ret >= lo) & (streak_ret < hh) & np.isfinite(FWD[60])
    if m.sum() < 300:
        continue
    y = FWD[60][m]; y = y[np.isfinite(y)]
    mu.append(100*y.mean())
axs[2].bar(range(len(mu)), mu)
axs[2].axhline(0, color='gray', lw=.8)
axs[2].set_xticks(range(len(lab2))); axs[2].set_xticklabels(lab2, rotation=45, fontsize=8)
axs[2].set_title('streak_ret -> mean ret next 60m')
axs[2].set_ylabel('%')
plt.tight_layout()
pp = os.path.join(EV, 'streak.png')
plt.savefig(pp, dpi=125)
say('\nsaved: %s' % pp)

with open(os.path.join(EV, 'streak.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
