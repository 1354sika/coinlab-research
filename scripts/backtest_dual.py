# -*- coding: utf-8 -*-
"""A) 把「连续段」双向信号写成正式算子   B) 组合回测（多空同跑，无前视，扣费）
算子：
  OP_LONG_DIPBOUNCE : disp > P90 且 连续同向段累计跌 > 5%  -> 做多
  OP_SHORT_SPIKEFADE: disp > P90 且 连续同向段累计涨 > 5%  -> 做空
公共：持有 60 分钟；不设止损（已验证止损为负贡献）；费档 taker 0.10%/边、maker 0.02%/边
回测规则（严格无前视）：
  信号在 t 分钟收盘产生 -> t+1 分钟起算收益；
  同一币在持仓期内不重复开仓（新信号只在空仓时生效）；
  组合等权：总资金按当期活跃仓位均分。
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

HOLD = 60
THR = 0.05
DISP_Q = 0.90
FEE = {'taker': 0.0010, 'maker': 0.0002}   # 单边

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
disp_v = px.pct_change().std(axis=1).values
T, N = P.shape
t = px.index.values
d90 = float(np.nanpercentile(disp_v, DISP_Q * 100))

# ---------- 连续同向段累计幅度 ----------
s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
p_start = np.take_along_axis(P, start, axis=0)
streak_ret = P / p_start - 1.0

# ---------- 算子定义 ----------
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)
SIG_L = (streak_ret < -THR) & sel          # OP_LONG_DIPBOUNCE
SIG_S = (streak_ret > +THR) & sel          # OP_SHORT_SPIKEFADE
say('信号（分钟x币，未去重）: 做多=%d  做空=%d' % (SIG_L.sum(), SIG_S.sum()))


def build_pos(sig, sign):
    """空仓才开：线性扫描（按列），返回 pos(T,N) 与 开仓索引"""
    pos = np.zeros((T, N), np.float32)
    opens = []
    for j in range(N):
        col = sig[:, j]
        if not col.any():
            continue
        ii = np.flatnonzero(col)
        last_exit = -1
        for k in ii:
            if k <= last_exit:
                continue
            pos[k:k + HOLD, j] = sign
            opens.append((k, j))
            last_exit = k + HOLD - 1
    return pos, np.array(opens) if opens else np.zeros((0, 2), int)


posL, openL = build_pos(SIG_L, +1.0)
posS, openS = build_pos(SIG_S, -1.0)
say('实际开仓次数: 做多=%d  做空=%d' % (len(openL), len(openS)))

pos = posL + posS
R_next = np.vstack([R[1:], np.full((1, N), np.nan)])
pnl_gross = np.nansum(pos[:-1] * R_next[:-1], axis=1)

active = (np.abs(pos[:-1]) > 0).sum(axis=1).astype(np.float64)
pnl_gross_eq = np.where(active > 0, pnl_gross / np.maximum(active, 1), 0.0)

fee_series = np.zeros(T)
for (k, j) in openL:
    fee_series[k] += FEE['taker']
for (k, j) in openS:
    fee_series[k] += FEE['taker']
fee_series_eq = np.where(active > 0, fee_series[:-1] / np.maximum(active, 1), 0.0)

say('\n=== B) 组合回测（等权，持有 %d 分钟，无前视）===' % HOLD)
say('  活跃持仓分钟数=%d  平均同时持仓=%.1f  最大=%d' % (
    int((active > 0).sum()), active.mean(), active.max()))
say('  覆盖时间 %s ~ %s' % (
    pd.to_datetime(t[0], unit='ms', utc=True).strftime('%Y-%m-%d'),
    pd.to_datetime(t[-1], unit='ms', utc=True).strftime('%Y-%m-%d')))

for tag, fee in (('taker 0.10%/边', FEE['taker']), ('maker 0.02%/边', FEE['maker'])):
    fs = np.zeros(T)
    for (k, j) in openL:
        fs[k] += fee
    for (k, j) in openS:
        fs[k] += fee
    fs_eq = np.where(active > 0, fs[:-1] / np.maximum(active, 1), 0.0)
    net = pnl_gross_eq - fs_eq
    nav = np.cumprod(1.0 + net)
    tot = nav[-1] - 1.0
    ann = (1.0 + tot) ** (365.0 / 60.0) - 1.0 if tot > -1 else -1.0
    sd = net.std()
    sharpe = net.mean() / sd * np.sqrt(525600) if sd > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    win = (net > 0).mean()
    say('\n  【%s】' % tag)
    say('    毛收益(未扣费)=%+.2f%%   总净收益=%+.2f%%   年化(线性外推)=%+.1f%%' % (
        100 * (pnl_gross_eq.sum()), 100 * tot, 100 * ann))
    say('    夏普(年化,分钟级)=%.2f   最大回撤=%.1f%%   分钟胜率=%.1f%%' % (
        sharpe, 100 * mdd, 100 * win))
    say('    交易数=%d (多%d/空%d)' % (len(openL) + len(openS), len(openL), len(openS)))

# 分腿
say('\n=== 分腿对照（maker 档）===')
for nm, p, op in (('做多腿', posL, openL), ('做空腿', posS, openS)):
    pg = np.nansum(p[:-1] * R_next[:-1], axis=1)
    act = (np.abs(p[:-1]) > 0).sum(axis=1).astype(float)
    pge = np.where(act > 0, pg / np.maximum(act, 1), 0.0)
    fs = np.zeros(T)
    for (k, j) in op:
        fs[k] += FEE['maker']
    fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
    net = pge - fse
    nav = np.cumprod(1 + net)
    pk = np.maximum.accumulate(nav)
    say('  %s: 交易=%d  总净=%+.2f%%  年化=%+.1f%%  夏普=%.2f  回撤=%.1f%%' % (
        nm, len(op), 100 * (nav[-1] - 1),
        100 * ((1 + nav[-1] - 1) ** (365.0 / 60.0) - 1) if nav[-1] > 0 else np.nan,
        net.mean() / net.std() * np.sqrt(525600) if net.std() > 0 else np.nan,
        100 * ((nav - pk) / pk).min()))

# 逐周
say('\n=== 逐周（组合，maker 档）===')
fs = np.zeros(T)
for (k, j) in openL:
    fs[k] += FEE['maker']
for (k, j) in openS:
    fs[k] += FEE['maker']
fse = np.where(active > 0, fs[:-1] / np.maximum(active, 1), 0.0)
net = pnl_gross_eq - fse
ts = pd.to_datetime(t[:-1], unit='ms', utc=True)
dfw = pd.DataFrame({'net': net, 'wk': ts.isocalendar().week.values})
gw = dfw.groupby('wk').agg(n=('net', 'size'), s=('net', 'sum'))
gw = gw[gw.n > 500]
say('  周数=%d  正收益周=%d (%.0f%%)' % (len(gw), (gw.s > 0).sum(), 100 * (gw.s > 0).mean()))
say('  各周累计: ' + ' '.join('%d:%+.2f%%' % (w, 100 * r.s) for w, r in gw.iterrows()))

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for tag, fee, col in (('taker 0.10%/边', FEE['taker'], 'tab:red'),
                      ('maker 0.02%/边', FEE['maker'], 'tab:blue')):
    fs = np.zeros(T)
    for (k, j) in openL:
        fs[k] += fee
    for (k, j) in openS:
        fs[k] += fee
    fse = np.where(active > 0, fs[:-1] / np.maximum(active, 1), 0.0)
    nav = np.cumprod(1 + pnl_gross_eq - fse)
    axs[0].plot(ts, nav, label=tag, lw=1.2)
axs[0].set_title('Dual-signal portfolio NAV (no look-ahead)')
axs[0].set_ylabel('NAV'); axs[0].legend(); axs[0].grid(alpha=.3)
axs[0].tick_params(axis='x', rotation=30)

axs[1].plot(ts, np.cumprod(1 + pnl_gross_eq) / np.maximum.accumulate(np.cumprod(1 + pnl_gross_eq)) * 100 - 100,
            lw=1, label='drawdown (gross)')
axs[1].set_title('Drawdown'); axs[1].set_ylabel('%'); axs[1].grid(alpha=.3)
axs[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
pp = os.path.join(EV, 'backtest_dual.png')
plt.savefig(pp, dpi=125)
say('\nsaved: %s' % pp)

with open(os.path.join(EV, 'backtest_dual.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
