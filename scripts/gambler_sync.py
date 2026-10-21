# -*- coding: utf-8 -*-
"""gambler · 群体协同检验（直接来自用户的内幕经历）

用户经历：一群人靠内幕消息协同，针对一种币大量买入卖出 → 抬高价格 →**同时撤走收盘**，
         本金翻数倍。

关键点：这不是"单体庄家"，而是**群体同步行动**。那么在数据上应留下**跨币同步**的指纹：
  异动不是零散随机发生的，而是**同一时刻多个币一起异动**。

检验（全部无前视、可实时计算）：
  A 同步异动水平：同一根 5m K 线里，有多少个币同时满足「量比≥5 且 |ret|≥5%」
     零模型（关键）：把每个币的异动标记**循环移位一个随机量** —— 保留该币自身的异动频率
     与聚集性，但破坏跨币同步。实际 vs 零模型，若实际远高 ⇒ **群体同步是真的**
  B 同步拉盘簇（≥5 币同时上涨异动）之后：这些币 2h/8h 怎么走
  C 同步撤走簇（≥3 币同时下跌异动）之后：市场整体怎么走（"一起撤"的痕迹）
  D 撤走时刻的成交量倍率（"同时砸盘"应该伴随极端量）
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler · 群体协同检验（跨币同步：一群人一起动、一起撤）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
close = df.pivot(index='open_time', columns='symbol', values='close').astype('float32')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
del df; gc.collect()
close = close.where(close > 0).ffill(limit=3)
qv = qv.where(qv >= 0)
ts = close.index.values
T, N = close.shape
syms = np.array(list(close.columns))
CV, QV = close.values, qv.values
say('%d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
med = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
med = np.where(med > 0, med, np.nan)
volr = QV / med
BAD = np.isfinite(volr) & (volr > 200)
ret1[BAD] = np.nan
say('极端成交量点（量比>200，视为假数据）：%s' % format(int(np.nansum(BAD)), ','))

M_UP = np.isfinite(volr) & np.isfinite(ret1) & (volr >= 5) & (ret1 >= 0.05)
M_DN = np.isfinite(volr) & np.isfinite(ret1) & (volr >= 5) & (ret1 <= -0.05)
say('单币"放量异动"点：上涨 %s（%.3f%%）/ 下跌 %s（%.3f%%）'
    % (format(int(M_UP.sum()), ','), 100 * M_UP.mean(),
       format(int(M_DN.sum()), ','), 100 * M_DN.mean()))

# ---------- A 实际 vs 零模型 ----------
say('\n' + '=' * 100)
say('【A 同步异动水平：实际 vs 零模型（每币循环移位随机量，保留自身频率、破坏跨币同步）】')
say('=' * 100)
n_up_real = M_UP.sum(axis=1)
n_dn_real = M_DN.sum(axis=1)
rows = np.arange(300, T - 96, 12)                     # 抽样统计，省时间
rng = np.random.default_rng(0)
sh_up = np.zeros_like(M_UP); sh_dn = np.zeros_like(M_DN)
for j in range(N):
    sh_up[:, j] = np.roll(M_UP[:, j], int(rng.integers(1, T)))
    sh_dn[:, j] = np.roll(M_DN[:, j], int(rng.integers(1, T)))
n_up_sh = sh_up.sum(axis=1); n_dn_sh = sh_dn.sum(axis=1)
del sh_up, sh_dn; gc.collect()

def cmp_line(name, a, b, rows):
    a, b = a[rows], b[rows]
    say('  %-8s 平均同时币数 实际=%.3f 零模型=%.3f (%.1f×)  ≥3币 实际=%.2f%% 零模型=%.2f%%  ≥5币 实际=%.2f%% 零模型=%.2f%%  最大 实际=%d 零模型=%d'
        % (name, a.mean(), b.mean(), a.mean() / max(b.mean(), 1e-9),
           100 * (a >= 3).mean(), 100 * (b >= 3).mean(),
           100 * (a >= 5).mean(), 100 * (b >= 5).mean(), a.max(), b.max()))
cmp_line('上涨异动', n_up_real, n_up_sh, rows)
cmp_line('下跌异动', n_dn_real, n_dn_sh, rows)

# ---------- 簇时刻 ----------
rows_all = np.arange(300, T - 96)
up5 = rows_all[n_up_real[rows_all] >= 5]
dn3 = rows_all[n_dn_real[rows_all] >= 3]
say('\n  同步拉盘簇（≥5 币同时上涨异动）时刻数 = %s（占时间 %.2f%%）'
    % (format(len(up5), ','), 100 * len(up5) / len(rows_all)))
say('  同步撤走簇（≥3 币同时下跌异动）时刻数 = %s（占时间 %.2f%%）'
    % (format(len(dn3), ','), 100 * len(dn3) / len(rows_all)))

# ---------- B 拉盘簇之后 ----------
say('\n' + '=' * 100)
say('【B 同步拉盘簇（≥5 币同时）之后：参与币与市场怎么走】')
say('=' * 100)
mkt = np.nanmedian(ret1, axis=1)
for k, hor in ((24, '2h'), (96, '8h'), (288, '24h')):
    fwd = np.full((T, N), np.nan, np.float32)
    fwd[:-k] = CV[k:] / CV[:-k] - 1.0
    part = np.nanmean(np.where(M_UP[up5], fwd[up5], np.nan), axis=1) if len(up5) else np.array([np.nan])
    allm = np.nanmean(fwd[up5], axis=1) if len(up5) else np.array([np.nan])
    say('  %-3s 之后：参与币平均 %+.3f%%（中位 %+.3f%%）｜全市场平均 %+.3f%%'
        % (hor, 100 * np.nanmean(part), 100 * np.nanmedian(part), 100 * np.nanmean(allm)))
if len(up5):
    fwd288 = np.full((T, N), np.nan, np.float32)
    fwd288[:-288] = CV[288:] / CV[:-288] - 1.0
    big = np.where(M_UP[up5], fwd288[up5] >= 1.0, np.nan)
    say('  之后 24h 参与币里 ≥+100%% 的比例：%.2f%%' % (100 * np.nanmean(big)))
else:
    say('  之后 24h：无同步拉盘簇')

# ---------- C/D 撤走簇之后 ----------
say('\n' + '=' * 100)
say('【C/D 同步撤走簇（≥3 币同时下跌异动）之后：市场整体 + 成交量倍率】')
say('=' * 100)
for k, hor in ((24, '2h'), (96, '8h'), (288, '24h')):
    fwd = np.full((T, N), np.nan, np.float32)
    fwd[:-k] = CV[k:] / CV[:-k] - 1.0
    say('  %-3s 之后：全市场平均 %+.3f%%（中位 %+.3f%%）｜同时币平均 %+.3f%%'
        % (hor, 100 * np.nanmean(fwd[dn3]), 100 * np.nanmedian(fwd[dn3]),
           100 * np.nanmean(np.where(M_DN[dn3], fwd[dn3], np.nan))))
say('  撤走时刻的同时币，成交量倍率中位 = %.1f×（量比）'
    % np.nanmedian(np.where(M_DN[dn3], volr[dn3], np.nan)))
say('  撤走时刻的同时币，单根跌幅中位 = %.2f%%'
    % (100 * np.nanmedian(np.where(M_DN[dn3], ret1[dn3], np.nan))))

np.save(os.path.join(EV, 'sync_nup.npy'), n_up_real)
open(os.path.join(EV, 'gambler_sync.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
