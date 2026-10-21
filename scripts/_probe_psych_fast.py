# -*- coding: utf-8 -*-
"""_probe_psych_fast.py — 抄近路：只用事件表（秒级）做操盘者心理侧写的 A/B/C/D
不读 10GB K线，只读 1163 行事件表：gain（峰值涨幅）+ min_to_peak（到峰值分钟）
"""
import pandas as pd, numpy as np
d = pd.read_csv(r'D:\中信证券\coinlab\evidence\pump_hunt_events.csv')
g = d['gain'].values.astype(float); m = d['min_to_peak'].values.astype(float)
ok = np.isfinite(g) & (g > 0)
g, m = g[ok], np.maximum(m[ok], 1.0)
print('事件 %d 个（有效 %d）' % (len(d), len(g)))
print('峰值涨幅: 中位 %.2f 倍 | 均值 %.2f | p10 %.2f | p90 %.2f | 最大 %.1f'
      % (np.median(g), g.mean(), np.percentile(g, 10), np.percentile(g, 90), g.max()))

print('\n【A 贪婪·目标位：落点是否聚集在常见目标位（±10%）】')
for lv in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
    hit = ((g >= lv * 0.9) & (g <= lv * 1.1)).sum()
    print('  +%5.0f%% 附近: %4d 个 (%.1f%%)' % (lv * 100, hit, 100 * hit / len(g)))
print('  涨幅对数分箱:')
bins = np.logspace(np.log10(0.5), np.log10(20), 16)
h, _ = np.histogram(np.clip(g, 0.5, 20), bins=bins)
for i in range(len(h)):
    if h[i] > 0:
        print('    %5.2f~%5.2f 倍: %4d %s' % (bins[i], bins[i + 1], h[i], '#' * int(60 * h[i] / h.max())))

print('\n【B 急切·拉升速度（涨幅 ÷ 到峰值小时数）】')
sp = g / (m / 60.0)
print('  速度: 中位 %.2f 倍/h | p10 %.2f | p90 %.2f | 最大 %.1f'
      % (np.median(sp), np.percentile(sp, 10), np.percentile(sp, 90), sp.max()))
q = np.percentile(sp, [33, 67])
for name, lo, hi in [('慢庄', -1, q[0]), ('中庄', q[0], q[1]), ('快庄', q[1], 1e9)]:
    s = (sp > lo) & (sp <= hi)
    print('  %s n=%4d  速度中位 %5.2f 倍/h  涨幅中位 %5.2f 倍  到峰值中位 %5.0f 分'
          % (name, s.sum(), np.median(sp[s]), np.median(g[s]), np.median(m[s])))

print('\n【C 耐心/步调·到峰值耗时分布】')
for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 240), (240, 600), (600, 999999)]:
    s = (m >= lo) & (m < hi)
    if s.sum():
        lbl = '%3d~%5d 分' % (lo, hi if hi < 999999 else 0)
        print('  %s: %4d (%.1f%%) %s' % (lbl, s.sum(), 100 * s.sum() / len(m), '#' * int(50 * s.sum() / len(m))))

print('\n【D 同一批人反复用同一个币？】')
vc = d['symbol'].value_counts()
print('  上榜币 %d 个 / 事件 %d 个 | 平均 %.2f 次/币 | 最高 %d 次' % (len(vc), len(d), len(d) / len(vc), vc.max()))
rep = vc[vc >= 3]
print('  ≥3 次的币 %d 个（占事件 %.1f%%）' % (len(rep), 100 * rep.sum() / len(d)))
print('  Top12: ' + ', '.join('%s×%d' % (k, v) for k, v in vc.head(12).items()))

print('\n【E thr 分层（0.5=小局 / 2.0=大局）】')
for t in sorted(d['thr'].dropna().unique()):
    s = d[d['thr'] == t]
    print('  thr=%.2f n=%4d | 涨幅中位 %5.2f 倍 | 到峰值中位 %5.0f 分 | 币数 %d'
          % (t, len(s), s['gain'].median(), s['min_to_peak'].median(), s['symbol'].nunique()))
print('\nTOTAL done')
