# -*- coding: utf-8 -*-
"""复利（利滚利）真实模拟：币卖了、钱回笼、再全投下一笔

上一轮 week_profit.py 的"全押"是【每周内按顺序复利】，隐含了"钱能同时买多个币"的问题。
本脚本做严格的事件驱动模拟：
  · 资金分成 K 份（K = 允许的最大同时持仓数）
  · 按信号发生的真实时间顺序遍历
  · 信号来时找一个"已平仓、资金空闲"的份；找不到就跳过（不重复用钱）
  · 每份独立复利：卖掉（平仓）后资金回笼，可再买入
  · 同一时刻最多 K 个仓位，每仓用 1/K 资金

对比 K = 1（单仓全押）/ 3 / 5 / 10 / 20，看真实复利上限与回撤
纪律：扣费 maker 双档（taker 另列）；事件去重；无前视；不重复使用同一笔资金
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_5m_3y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

DEP, VOLR, CONCUR, HOLD = 0.05, 10.0, 4, 12
FEE_M, FEE_T = 0.0002, 0.0010
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
CV = volr > VOLR; CD = streak_ret < -DEP
conc = (CD & CV).sum(axis=1)[:, None] * np.ones((1, N))
SIG = CD & CV & (conc >= CONCUR)
coin_7d = P / pd.DataFrame(P).shift(D7).values - 1.0


def dedup(S, hold=HOLD):
    out = np.zeros_like(S); last = np.full(N, -10 ** 9, np.int64)
    for i in np.where(S.any(axis=1))[0]:
        sel = S[i] & ((i - last) >= hold)
        if sel.any():
            out[i] = sel; last[sel] = i
    return out


ii, jj = np.where(dedup(SIG))
y = FWD[ii, jj]; ok = np.isfinite(y)
ii, jj, y = ii[ok], jj[ok], y[ok]
keep = ~(coin_7d[ii, jj] < -0.20)
ii, jj, y = ii[keep], jj[keep], y[keep]
dt = pd.to_datetime(tms[ii], unit='ms', utc=True)
say('候选信号 %d 笔  %s ~ %s' % (len(y), dt.min().date(), dt.max().date()))

# 同时性检查：有多少信号在时间上重叠
o = np.argsort(ii, kind='stable')
ii, jj, y = ii[o], jj[o], y[o]
overlap = 0
for a in range(len(ii)):
    for b in range(a + 1, len(ii)):
        if ii[b] >= ii[a] + HOLD:
            break
        overlap += 1
say('时间重叠的信号对 = %d（若为 0，说明每笔都能独享全部资金）' % overlap)


def compound(net, K, cap=CAP):
    """K 份资金事件驱动复利；返回 (净值序列, 期末资金, 实际成交笔数, 跳过笔数)"""
    slot = np.full(K, cap / K)
    free = np.zeros(K, np.int64)
    hist = [cap]; taken = 0; skipped = 0
    for t in range(len(net)):
        i = ii[t]
        avail = np.where(free <= i)[0]
        if len(avail) == 0:
            skipped += 1
            continue
        k = avail[0]
        slot[k] *= (1 + net[t])
        free[k] = i + HOLD
        taken += 1
        hist.append(slot.sum())
    h = np.array(hist)
    return h, slot.sum(), taken, skipped


say('\n' + '=' * 88)
say('【复利模拟：币卖了钱回笼再全投】1000 元本金，%d 笔候选' % len(y))
say('  %-22s %10s %14s %12s %12s %10s %10s' % (
    '最大同时持仓 K', '实际成交', '跳过(钱不够)', '期末资金', '总收益%', '最大回撤%', '年化%'))
years = (dt.max() - dt.min()).days / 365.25
res = {}
for K in (1, 3, 5, 10, 20, 60):
    for fee_nm, fee in (('maker', 2 * FEE_M), ('taker', 2 * FEE_T)):
        h, end, taken, skip = compound(y - fee, K)
        peak = np.maximum.accumulate(h)
        dd = ((h - peak) / peak).min()
        ann = (end / CAP) ** (1 / years) - 1
        if fee_nm == 'maker':
            res[K] = (h, end, dd, taken, skip, ann)
            say('  K=%-3d %-15s %8d %12d %13.2f %+11.1f%% %11.1f%% %9.1f%%' % (
                K, fee_nm, taken, skip, end, 100 * (end / CAP - 1), 100 * dd, 100 * ann))
        else:
            say('  K=%-3d %-15s %8d %12d %13.2f %+11.1f%% %11.1f%% %9.1f%%' % (
                K, fee_nm, taken, skip, end, 100 * (end / CAP - 1), 100 * dd, 100 * ann))

say('\n【结论判读】')
say('  · K=1 意味着「一次只持一个币、全押」—— 但若信号同时出现，后面的全部被迫跳过')
say('  · K 越大，越能吃满信号，但单笔冲击越小（资金被摊薄）')
say('  · 观察哪一档「期末资金」最高，那才是真实复利上限（不是全押那行数字）')

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for K in (1, 3, 5, 10, 20):
    if K in res:
        ax[0].plot(res[K][0], label='K=%d' % K)
ax[0].set_yscale('log'); ax[0].axhline(CAP, color='k', lw=.8, ls='--')
ax[0].legend(fontsize=8); ax[0].set_title('equity (log), maker fee')
ax[0].set_xlabel('trade #')
Ks = [k for k in (1, 3, 5, 10, 20, 60) if k in res]
ax[1].bar(range(len(Ks)), [100 * (res[k][1] / CAP - 1) for k in Ks])
ax[1].set_xticks(range(len(Ks))); ax[1].set_xticklabels(['K=%d' % k for k in Ks])
ax[1].axhline(0, color='k', lw=.8); ax[1].set_title('total return % (maker)')
ax[2].bar(range(len(Ks)), [100 * res[k][2] for k in Ks])
ax[2].set_xticks(range(len(Ks))); ax[2].set_xticklabels(['K=%d' % k for k in Ks])
ax[2].set_title('max drawdown %')
plt.tight_layout()
p = os.path.join(EV, 'compound_sim.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
say('\nsaved: %s' % p)
open(os.path.join(EV, 'compound_sim.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
