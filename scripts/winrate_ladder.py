# -*- coding: utf-8 -*-
"""胜率阶梯：从「每个 5m 都下注」到「只在极端时刻下注」，胜率是怎么被选出来的

回答用户问题：「全部 5m 平均下注的概率只有 50% 吗？那能不能提升到 70%、和其他机制一样？」

关键区分：
  · 「预测能力」= 单次预测的准确率（≈50%，无法提升）
  · 「选择能力」= 只在有利时刻下注（可以把胜率抬到 70%+，但样本会大幅减少）

做法：逐级加条件，看 ★胜率★ 与 ★样本量★ 的变化
  阶梯0  全部 5m（无条件下注）
  阶梯1  + 横截面分化 disp > P90
  阶梯2  + 段内跌幅 < -5%
  阶梯3  + 量比 > 3
  阶梯4  + 多币同时 >= 2
  阶梯5  + 币自身 7 天跌超 20% 不进场
每级给出：样本数、胜率、单笔均值、扣费后期望、年化机会数
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

HOLD, DEP, VOLR, CONCUR, DQ = 12, 0.05, 3.0, 2, 0.90
FEE_M = 0.0002
D7 = 288 * 7
DAYS = 1070.0

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
say('matrix %d x %d   %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan); ret1[1:] = P[1:] / P[:-1] - 1.0
disp_v = np.nanstd(ret1, axis=1)
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
ones = np.ones((1, N), bool)
d90 = float(np.nanpercentile(disp_v[60:], DQ * 100))
OK = np.isfinite(FWD)

C_disp = (disp_v > d90)[:, None] & ones
C_dep = streak_ret < -DEP
C_vol = volr > VOLR
BASE = C_dep & C_vol
conc = BASE.sum(axis=1)[:, None] * ones
C_conc = conc >= CONCUR
C_coin7 = coin_7d > -0.20

LADDER = [
    ('0 全部 5m（无条件）', np.ones((T, N), bool)),
    ('1 + 横截面分化>P90', C_disp),
    ('2 + 段内跌<-5%', C_disp & C_dep),
    ('3 + 量比>3', C_disp & C_dep & C_vol),
    ('4 + 多币同时>=2', C_disp & C_dep & C_vol & C_conc),
    ('5 + 币7天跌>20%不进场', C_disp & C_dep & C_vol & C_conc & C_coin7),
]

say('\n' + '=' * 100)
say('【胜率阶梯】逐级加条件，看胜率与样本量（持 60 分钟）')
say('  %-24s %12s %9s %10s %10s %11s %10s' % (
    '阶梯', '样本数', '胜率%', '均值%', '中位%', '净maker%', '机会/年'))
rows_out = []
for nm, M in LADDER:
    m = M & OK
    n = int(m.sum())
    if n < 100:
        say('  %-24s 样本太少' % nm); continue
    y = FWD[m]
    net = y - 2 * FEE_M
    opp_year = n / (DAYS / 365.25)
    rows_out.append((nm, n, 100 * (y > 0).mean(), 100 * y.mean(), 100 * np.median(y), 100 * net.mean(), opp_year))
    say('  %-24s %12d %8.2f%% %+9.4f%% %+9.4f%% %+10.4f%% %10.0f' % (
        nm, n, 100 * (y > 0).mean(), 100 * y.mean(), 100 * np.median(y), 100 * net.mean(), opp_year))

say('\n【判读】')
say('  · 胜率从「靠近 50%」抬到「70%+」，靠的不是预测更准，而是【只在有利时刻下注】')
say('  · 代价：样本从 1252 万 -> 几百，机会从「每分钟」-> 「每年几十次」')
say('  · 注意阶梯 2->3 之间：样本减少最多、胜率提升最猛的一段，就是「量比」这道门')

# 单笔均值 vs 胜率的权衡图
if len(rows_out) >= 3:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    xs = range(len(rows_out))
    ax[0].plot(xs, [r[2] for r in rows_out], 'o-')
    ax[0].axhline(50, color='k', ls='--', lw=.8, label='coin flip 50%')
    ax[0].axhline(59, color='k', ls=':', lw=.8, label='maker breakeven ~59%')
    ax[0].set_xticks(xs); ax[0].set_xticklabels([r[0][0] for r in rows_out])
    ax[0].legend(fontsize=8); ax[0].set_ylabel('win rate %'); ax[0].set_title('win rate by ladder step')
    ax[1].bar(xs, [r[1] for r in rows_out]); ax[1].set_yscale('log')
    ax[1].set_xticks(xs); ax[1].set_xticklabels([r[0][0] for r in rows_out])
    ax[1].set_ylabel('samples (log)'); ax[1].set_title('sample count by step')
    ax[2].bar(xs, [r[3] for r in rows_out])
    ax[2].set_xticks(xs); ax[2].set_xticklabels([r[0][0] for r in rows_out])
    ax[2].axhline(0, color='k', lw=.8); ax[2].set_ylabel('mean net %')
    ax[2].set_title('per-trade mean net %')
    plt.tight_layout()
    p = os.path.join(EV, 'winrate_ladder.png'); plt.savefig(p, dpi=125, bbox_inches='tight')
    say('\nsaved: %s' % p)

open(os.path.join(EV, 'winrate_ladder.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
