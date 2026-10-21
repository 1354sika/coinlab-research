# -*- coding: utf-8 -*-
"""把已确认的机制放到不同时期实验：市场在变化，机制能适应/进化吗？
已确认机制：
  M1 横截面离散度 disp（分化 vs 同步）—— 新的观测维度
  M2 爆发点信号：分化 + 急跌5% + 放量2倍 -> 做多（单边）
  M3 跌幅门槛：3% 无效 / 5% 起效 / 7% 强
  M4 持有期：1 分钟负，3+ 分钟正，60 分钟最优
  M5 止损是负贡献
  M6 做空侧无效
做法：把 60 天切成 K 期，每期重算；
  M2 同时用【全局固定阈值】与【当期自适应阈值(P90)】两种口径 —— 
  若自适应明显更好，说明机制需要随市场重新校准（= 可进化）；
  若两者接近，说明参数稳定、无需进化。
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
K = 6

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float32',
                        'low': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
lo = df.pivot(index='open_time', columns='symbol', values='low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

r = px.pct_change()
disp = r.std(axis=1); disp_v = disp.values
R15 = (px / px.shift(15) - 1.0).values
VR = (qv / qv.rolling(60).mean()).values
C = px.values; L = lo.values
t = px.index.values
HOLD = (5, 15, 30, 60)
FWD = {h: (px.shift(-h) / px - 1.0).values for h in HOLD}
FWDN = {h: (px / px.shift(h) - 1.0).values for h in HOLD}   # 做空用（未来跌=正）

dp = np.nanpercentile(disp_v, [50, 90, 99])
G90, G99 = dp[1], dp[2]
say('全局 disp: 中位=%.4f%%  P90=%.4f%%  P99=%.4f%%' % (dp[0]*100, G90*100, G99*100))

edges = np.linspace(t[0], t[-1], K + 1)
day = lambda ms: pd.to_datetime(ms, unit='ms', utc=True).strftime('%m-%d')


def extract(condmask, selname):
    """事件级：同币间隔<=5min 合并，取首次"""
    out = []
    for j in range(condmask.shape[1]):
        ii = np.flatnonzero(condmask[:, j])
        if len(ii) == 0:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(ii) > 5) + 1]
        for s in starts:
            out.append((ii[s], j))
    return out


rows = []
for p in range(K):
    a, b = edges[p], edges[p + 1]
    mrow = (t >= a) & (t < b)
    dseg = disp_v[mrow & np.isfinite(disp_v)]
    dP90 = np.nanpercentile(dseg, 90)
    base = (R15 < -0.05) & (VR > 2.0) & mrow[:, None] & np.ones_like(C, bool)
    ev_fix = extract(base & (disp_v > G90)[:, None] & np.ones_like(C, bool), 'fix')
    ev_ad = extract(base & (disp_v > dP90)[:, None] & np.ones_like(C, bool), 'ad')

    def stat(ev, h=60):
        yy = np.array([FWD[h][k, j] for k, j in ev])
        yy = yy[np.isfinite(yy)]
        if len(yy) < 20:
            return (len(yy), np.nan, np.nan, np.nan)
        return (len(yy), 100*yy.mean(), 100*(yy.mean()-FEE_T), 100*(yy.mean()-FEE_M))

    nf, mf, tf, kf = stat(ev_fix)
    na, ma, ta, ka = stat(ev_ad)
    # 做空侧
    sb = (R15 > 0.05) & (VR > 2.0) & mrow[:, None] & np.ones_like(C, bool)
    ev_s = extract(sb & (disp_v > G90)[:, None] & np.ones_like(C, bool), 's')
    ys = np.array([-FWD[60][k, j] for k, j in ev_s]); ys = ys[np.isfinite(ys)]
    # 跌幅门槛
    gates = []
    for rthr in (0.03, 0.04, 0.05, 0.07):
        g = (R15 < -rthr) & (VR > 2.0) & mrow[:, None] & np.ones_like(C, bool)
        evg = extract(g & (disp_v > dP90)[:, None] & np.ones_like(C, bool), 'g')
        yg = np.array([FWD[60][k, j] for k, j in evg]); yg = yg[np.isfinite(yg)]
        gates.append((len(yg), 100*yg.mean() if len(yg) >= 20 else np.nan))
    # 持有期
    holds = [100 * np.nanmean([FWD[h][k, j] for k, j in ev_ad]) for h in HOLD]

    say('\n=== 期 %d/%d  %s ~ %s  (%d 分钟) ===' % (
        p+1, K, day(a), day(b), int(mrow.sum())))
    say('  M1 离散度: 中位=%.4f%%  P90=%.4f%%  (全局P90=%.4f%%)  -> 市场状态%s' % (
        np.nanmedian(dseg)*100, dP90*100, G90*100,
        '变了' if abs(dP90-G90)/G90 > 0.10 else '基本不变'))
    say('  M2 固定阈值: n=%4d 毛=%+.3f%% 净taker=%+.3f%% 净maker=%+.3f%%' % (nf, mf, tf, kf))
    say('  M2 自适应  : n=%4d 毛=%+.3f%% 净taker=%+.3f%% 净maker=%+.3f%%' % (na, ma, ta, ka))
    say('  M6 做空侧  : n=%4d 毛(做空)=%+.3f%%' % (len(ys), 100*ys.mean() if len(ys) else np.nan))
    say('  M3 跌幅门槛(自适应): ' + '  '.join(
        '%.0f%%:n=%d,%+.2f%%' % (r*100, n_, m_) for (n_, m_), r in zip(gates, (0.03,0.04,0.05,0.07))))
    say('  M4 持有期(自适应): ' + '  '.join('%dm:%+.2f%%' % (h, v) for h, v in zip(HOLD, holds)))

    rows.append(dict(p=p+1, dmed=np.nanmedian(dseg)*100, dP90=dP90*100,
                     nf=nf, mf=mf, tf=tf, kf=kf, na=na, ma=ma, ta=ta, ka=ka,
                     short=100*ys.mean() if len(ys) else np.nan,
                     g3=gates[0][1], g4=gates[1][1], g5=gates[2][1], g7=gates[3][1],
                     h5=holds[0], h15=holds[1], h30=holds[2], h60=holds[3]))
R = pd.DataFrame(rows)

say('\n\n=== 跨期汇总：固定 vs 自适应（持有60分钟，扣maker）===')
say('  期  离散中位%  P90%   固定n  固定净%  自适应n  自适应净%  差异pp')
for _, x in R.iterrows():
    say('  %d   %7.4f  %6.4f  %6d  %+7.3f  %7d  %+8.3f  %+6.3f' % (
        x.p, x.dmed, x.dP90, x.nf, x.ka, x.na, x.ta, x.ta - x.ka))
say('  均值/合计: 固定净均值=%+.3f%% (n合计=%d)   自适应净均值=%+.3f%% (n合计=%d)' % (
    R.ka.mean(), R.nf.sum(), R.ta.mean(), R.na.sum()))

say('\n=== 机制跨期一致性（正收益期数 / 总期数）===')
say('  M2 做多(固定): %d/%d 期为正   做多(自适应): %d/%d' % (
    (R.ka > 0).sum(), K, (R.ta > 0).sum(), K))
say('  M6 做空侧    : %d/%d 期为正' % ((R.short > 0).sum(), K))
say('  M3 门槛 5%%   : %d/%d 期为正' % ((R.g5 > 0).sum(), K))
say('  M3 门槛 7%%   : %d/%d 期为正' % ((R.g7 > 0).sum(), K))
say('  M4 持有60m   : %d/%d 期为正' % ((R.h60 > 0).sum(), K))
say('  M4 持有5m    : %d/%d 期为正' % ((R.h5 > 0).sum(), K))

say('\n=== M1 市场是否在变化 ===')
say('  disp 中位: 最小 %.4f%%  最大 %.4f%%  极差比 %.2f 倍' % (
    R.dmed.min(), R.dmed.max(), R.dmed.max()/max(R.dmed.min(), 1e-9)))
say('  disp P90 : 最小 %.4f%%  最大 %.4f%%  极差比 %.2f 倍' % (
    R.dP90.min(), R.dP90.max(), R.dP90.max()/max(R.dP90.min(), 1e-9)))
say('  与全局 P90(%.4f%%) 偏离超过 10%% 的期数: %d/%d' % (
    G90*100, (abs(R.dP90-G90)/G90 > 0.10).sum(), K))

# ---- 图 ----
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
ax = axs[0, 0]
ax.plot(R.p, R.dmed, 'o-', label='disp median')
ax.plot(R.p, R.dP90, 's-', label='disp P90')
ax.axhline(G90*100, ls='--', lw=1, label='global P90')
ax.set_title('M1 cross-sectional dispersion by period')
ax.set_xlabel('period'); ax.set_ylabel('%'); ax.legend(); ax.grid(alpha=.3)

ax = axs[0, 1]
w = 0.38
ax.bar(R.p - w/2, R.ka, w, label='fixed P90')
ax.bar(R.p + w/2, R.ta, w, label='adaptive P90')
ax.axhline(0, color='gray', lw=.8)
ax.set_title('M2 dip-bounce signal: net (maker) by period')
ax.set_xlabel('period'); ax.set_ylabel('net %'); ax.legend(); ax.grid(alpha=.3)

ax = axs[1, 0]
for c, lb in [('g3', '-3%'), ('g4', '-4%'), ('g5', '-5%'), ('g7', '-7%')]:
    ax.plot(R.p, R[c], 'o-', label=lb)
ax.axhline(0, color='gray', lw=.8)
ax.set_title('M3 drawdown threshold by period')
ax.set_xlabel('period'); ax.set_ylabel('gross %'); ax.legend(); ax.grid(alpha=.3)

ax = axs[1, 1]
for c, lb in [('h5', 'hold 5m'), ('h15', 'hold 15m'), ('h30', 'hold 30m'), ('h60', 'hold 60m')]:
    ax.plot(R.p, R[c], 'o-', label=lb)
ax.axhline(0, color='gray', lw=.8)
ax.set_title('M4 holding period by period')
ax.set_xlabel('period'); ax.set_ylabel('gross %'); ax.legend(); ax.grid(alpha=.3)

plt.tight_layout()
pp = os.path.join(EV, 'epochs.png')
plt.savefig(pp, dpi=125)
say('\nsaved: %s' % pp)
R.to_csv(os.path.join(EV, 'epochs.csv'), index=False, encoding='utf-8')

with open(os.path.join(EV, 'epochs.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
