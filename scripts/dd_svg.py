# -*- coding: utf-8 -*-
"""从 dd_curve.json 画真实资金曲线 SVG（标注峰值与谷底）。"""
import json, os

EVI = r'D:\中信证券\coinlab\evidence'
d = json.load(open(os.path.join(EVI, 'dd_curve.json')))
cv = d['curve']; pi = d['peak_i']; ti = d['trough_i']; mdd = d['mdd']

W, H = 900, 420
L, R, T, B = 70, 30, 30, 60
pw, ph = W - L - R, H - T - B
n = len(cv)
ymin, ymax = min(cv), max(cv)
pad = (ymax - ymin) * 0.10
ymin -= pad; ymax += pad

def X(i): return L + pw * i / (n - 1)
def Y(v): return T + ph * (1 - (v - ymin) / (ymax - ymin))

pts = ' '.join('%.1f,%.1f' % (X(i), Y(v)) for i, v in enumerate(cv))

g = []
g.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" style="background:#0f1420">' % (W, H, W, H))
# 网格
for k in range(5):
    v = ymin + (ymax - ymin) * k / 4
    y = Y(v)
    g.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#293040" stroke-width="1"/>' % (L, y, W - R, y))
    g.append('<text x="%d" y="%.1f" fill="#7b8798" font-size="12" text-anchor="end">%.2f</text>' % (L - 8, y + 4, v))
# 1.0 基准线
y1 = Y(1.0)
g.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#5a6578" stroke-width="1" stroke-dasharray="5,4"/>' % (L, y1, W - R, y1))
g.append('<text x="%d" y="%.1f" fill="#5a6578" font-size="11">本金 1.00</text>' % (W - R - 78, y1 - 6))
# 曲线
g.append('<polyline points="%s" fill="none" stroke="#4da3ff" stroke-width="1.6"/>' % pts)
# 峰
g.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c8a34a" stroke-width="1" stroke-dasharray="4,3"/>' % (X(pi), Y(cv[pi]), X(ti), Y(cv[pi])))
g.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#c8a34a"/>' % (X(pi), Y(cv[pi])))
g.append('<text x="%.1f" y="%.1f" fill="#c8a34a" font-size="12">峰值 %.3f</text>' % (X(pi) + 6, Y(cv[pi]) - 8, cv[pi]))
# 谷
g.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#9b7bd4"/>' % (X(ti), Y(cv[ti])))
g.append('<text x="%.1f" y="%.1f" fill="#9b7bd4" font-size="12">谷底 %.3f</text>' % (X(ti) + 6, Y(cv[ti]) + 16, cv[ti]))
# 回撤标注竖线
g.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9b7bd4" stroke-width="1"/>' % (X(ti), Y(cv[pi]), X(ti), Y(cv[ti])))
g.append('<text x="%.1f" y="%.1f" fill="#9b7bd4" font-size="13" text-anchor="middle">回撤 %.1f%%</text>' % (X(ti), (Y(cv[pi]) + Y(cv[ti])) / 2, mdd * 100))
# 轴标题
g.append('<text x="%d" y="%d" fill="#9aa5b6" font-size="13">676 笔逐笔复利（池20，每笔1/20资金）　x=第几笔　y=净值倍数</text>' % (L, H - 18))
g.append('</svg>')

out = os.path.join(EVI, 'drawdown_curve.svg')
open(out, 'w', encoding='utf-8').write('\n'.join(g))
print('written ' + out)
