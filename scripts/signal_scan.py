# -*- coding: utf-8 -*-
"""信号扩容扫描：能不能收到更多「又大又稳」的机会？

现状痛点：单笔 +1.8% 但 3 年只有 285 笔（一年约 10 次）。
本脚本网格扫描三条门槛，每组都给出：
  样本数 | 单笔净maker | 胜率 | 逐年为正数 | K=3 复利终值(1000元) | 最大回撤
纪律：同一套事件去重 + 无前视 + 扣费 maker；币自身7天跌超20%过滤统一开启
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_5m_3y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

HOLD, FEE_M = 12, 0.0002
D7 = 288 * 7
CAP = 1000.0

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float64', 'quote_volume': 'float64'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P, QV = px.values, qv.values
T, N = P.shape
tms = px.index.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60
s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
FWD = np.full((T, N), np.nan); FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0
coin_7d = P / pd.DataFrame(P).shift(D7).values - 1.0
OK_FWD = np.isfinite(FWD)
say('预计算完成 %.0fs' % (time.time() - t0))


def dedup_idx(S):
    """返回去重后的 (bar, coin) 索引；只在有信号的 bar 上循环"""
    bi, bj = [], []
    last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= HOLD)
        if sel.any():
            js = np.where(sel)[0]
            bi.append(np.full(len(js), i)); bj.append(js)
            last[js] = i
    if not bi:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(bi), np.concatenate(bj)


def compound(ii, jj, net, K=3, cap=CAP):
    """事件驱动 K 份复利"""
    o = np.argsort(ii, kind='stable')
    ii, jj, net = ii[o], jj[o], net[o]
    slot = np.full(K, cap / K); free = np.zeros(K, np.int64)
    hist = [cap]; taken = 0
    for t in range(len(net)):
        i = ii[t]
        av = np.where(free <= i)[0]
        if len(av) == 0:
            continue
        k = av[0]
        slot[k] *= (1 + net[t]); free[k] = i + HOLD
        taken += 1
        hist.append(slot.sum())
    h = np.array(hist)
    dd = ((h - np.maximum.accumulate(h)) / np.maximum.accumulate(h)).min() if len(h) > 1 else 0
    return h[-1], dd, taken


DES = [0.03, 0.05, 0.07, 0.10]
VRS = [3.0, 5.0, 10.0, 20.0]
CCS = [2, 3, 4, 6]

say('\n' + '=' * 108)
say('【网格扫描】跌幅 x 量比 x 多币数（统一开「币7天跌>20%不进场」过滤，K=3 复利，maker）')
say('  %6s %6s %6s | %7s %7s %9s %8s %11s %10s %7s' % (
    '跌幅', '量比', '多币', '信号', '成交', '单笔净%', '胜率%', 'K3终值', '回撤%', '正年数'))
rows_out = []
for dep in DES:
    for vr in VRS:
        for cc in CCS:
            CD = streak_ret < -dep
            CV = volr > vr
            base = CD & CV
            conc = base.sum(axis=1)[:, None] * np.ones((1, N))
            SIG = base & (conc >= cc)
            ii, jj = dedup_idx(SIG)
            if len(ii) < 30:
                continue
            okf = np.isfinite(FWD[ii, jj])
            ii, jj = ii[okf], jj[okf]
            keep = ~(coin_7d[ii, jj] < -0.20)
            ii, jj = ii[keep], jj[keep]
            if len(ii) < 30:
                continue
            y = FWD[ii, jj]
            net = y - 2 * FEE_M
            yr = pd.to_datetime(tms[ii], unit='ms', utc=True).year.values
            yy = sorted(set(yr))
            pos = sum(1 for k in yy if (yr == k).sum() >= 5 and net[yr == k].mean() > 0)
            tot = sum(1 for k in yy if (yr == k).sum() >= 5)
            end, dd, taken = compound(ii, jj, net, K=3)
            rows_out.append((dep, vr, cc, len(ii), taken, 100 * net.mean(),
                             100 * (net > 0).mean(), end, 100 * dd, '%d/%d' % (pos, tot)))
            say('  %5.0f%% %6.0f %6d | %7d %7d %+8.3f%% %7.1f%% %10.2f %9.1f%% %7s' % (
                dep * 100, vr, cc, len(ii), taken, 100 * net.mean(),
                100 * (net > 0).mean(), end, 100 * dd, '%d/%d' % (pos, tot)))

say('\n【排序：按 K=3 复利终值】')
say('  %6s %6s %6s | %7s %9s %8s %11s %10s %7s' % (
    '跌幅', '量比', '多币', '成交', '单笔净%', '胜率%', 'K3终值', '回撤%', '正年数'))
for r in sorted(rows_out, key=lambda x: -x[7])[:15]:
    say('  %5.0f%% %6.0f %6d | %7d %+8.3f%% %7.1f%% %10.2f %9.1f%% %7s' % (
        r[0] * 100, r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9]))

say('\n【排序：按单笔净收益】')
for r in sorted(rows_out, key=lambda x: -x[5])[:10]:
    say('  %5.0f%% %6.0f %6d | 成交%6d  单笔=%+.3f%%  胜率=%.1f%%  K3=%9.2f  回撤=%.1f%%  %s' % (
        r[0] * 100, r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9]))

with open(os.path.join(EV, 'signal_scan.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
