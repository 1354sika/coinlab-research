# -*- coding: utf-8 -*-
"""_probe_time.py — 秒级（纯事件表）检验：时段效应 + 整数关口 + 涨幅聚集
A 时段：1163 个局的启动时刻 UTC 小时 / 星期分布（均匀基准 = 1/24、1/7）
B 整数关口：峰值价 peak_high 是否聚集在「1/2/5 × 10^k」这类整数位
C 涨幅聚集：gain 是否聚集在 +50% / +100%（对 A 节的复核，用事件表原始列）
"""
import numpy as np, pandas as pd

E = r'D:\中信证券\coinlab\evidence\pump_hunt_events.csv'
d = pd.read_csv(E)
d['start_time'] = pd.to_datetime(d['start_time'], utc=True)

print('=' * 96)
print('【A 时段效应】局的启动时刻分布（UTC）—— 基准：均匀 = 4.17%%/小时、14.29%%/天')
print('=' * 96)
h = d['start_time'].dt.hour.value_counts().sort_index()
print('  按 UTC 小时：')
for k, v in h.items():
    p = 100.0 * v / len(d)
    flag = '★' if p > 4.17 * 1.4 else ('·' if p < 4.17 * 0.6 else ' ')
    print('    %02d时 %5.2f%% %s %s' % (k, p, '#' * int(p * 1.6), flag))
wd = d['start_time'].dt.dayofweek.value_counts().sort_index()
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
print('  按星期（UTC，基准 14.29%%）：')
for k, v in wd.items():
    p = 100.0 * v / len(d)
    print('    %s %5.2f%% %s' % (names[k], p, '#' * int(p * 1.2)))

# 亚洲/欧洲/美国盘
def bucket(hh):
    if 0 <= hh < 8: return '亚洲盘(00-08 UTC)'
    if 8 <= hh < 16: return '欧洲盘(08-16 UTC)'
    return '美洲盘(16-24 UTC)'
b = d['start_time'].dt.hour.map(bucket).value_counts()
print('  按时段（基准 33.3%%）：')
for k, v in b.items():
    print('    %-20s %5.2f%%' % (k, 100.0 * v / len(d)))

print('\n' + '=' * 96)
print('【B 整数关口】峰值价 peak_high 是否聚集在「1/2/5 × 10^k」（±2%）')
print('=' * 96)
p = d['peak_high'].values.astype(float)
p = p[np.isfinite(p) & (p > 0)]
lg = np.log10(p)
mant = 10 ** (lg - np.floor(lg))            # 尾数 ∈ [1,10)
for tgt in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5):
    lo, hi = tgt * 0.98, tgt * 1.02
    hit = ((mant >= lo) & (mant <= hi)).sum()
    exp = len(p) * (hi - lo) / 9.0          # 对数均匀下的期望
    print('    尾数 ≈ %4.1f (±2%%)：%4d 个（实测 %5.2f%% vs 期望 %5.2f%%）%s'
          % (tgt, hit, 100 * hit / len(p), 100 * exp / len(p),
             '★聚集' if hit > exp * 1.5 else ('·稀少' if hit < exp * 0.5 else '')))

print('\n' + '=' * 96)
print('【C 涨幅聚集】峰值涨幅是否聚集在 +50% / +100% / +150%（±10%）')
print('=' * 96)
g = d['gain'].values.astype(float); g = g[np.isfinite(g) & (g > 0)]
for lv in (0.5, 1.0, 1.5, 2.0, 3.0):
    hit = ((g >= lv * 0.9) & (g <= lv * 1.1)).sum()
    print('    +%4.0f%% (±10%%)：%4d 个（%.1f%%）' % (lv * 100, hit, 100 * hit / len(g)))
print('    涨幅中位 %.2f 倍 ｜ 均值 %.2f 倍' % (np.median(g), g.mean()))

print('\n' + '=' * 96)
print('【D 币种复用 & 链条】同一批币反复出现')
print('=' * 96)
vc = d['symbol'].value_counts()
print('  277 币 / %d 事件 ｜ 均值 %.2f 次/币 ｜ 最高 %d 次' % (len(d), len(d) / len(vc), vc.max()))
print('  Top10：' + ', '.join('%s×%d' % (k, v) for k, v in vc.head(10).items()))
print('\nTOTAL done')
