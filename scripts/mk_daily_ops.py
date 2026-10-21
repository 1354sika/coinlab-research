# -*- coding: utf-8 -*-
"""每天有多少个「操盘局」？—— 回答用户问："每一天都有操盘手，能否锁定他们？"

数据源：evidence/pump_hunt_events.csv（job-14824-3 产出，304 币 3 年 5m）
"""
import pandas as pd
import numpy as np

EV = r'D:\中信证券\coinlab\evidence'
d = pd.read_csv(EV + r'\pump_hunt_events.csv')
d['t'] = pd.to_datetime(d.start_time)
d['day'] = d.t.dt.date
span = (d.t.max() - d.t.min()).days + 1
print('数据跨度 %s ~ %s   共 %d 天（304 币）' % (d.t.min().date(), d.t.max().date(), span))

for thr, lab in ((0.5, '4h 内 ≥+50%'), (1.0, '4h 内 ≥+100%'), (2.0, '4h 内 ≥+200%')):
    s = d[d.thr == thr]
    g = s.groupby('day').size()
    print('\n【%s】事件 %d 个，分布在 %d 天（占 %.0f%% 的天数）'
          % (lab, len(s), len(g), 100 * len(g) / span))
    print('  每天局数：均值 %.2f  中位 %.0f  P75 %.0f  P90 %.0f  最大 %d'
          % (g.mean(), g.median(), np.percentile(g, 75), np.percentile(g, 90), g.max()))
    print('  **一个局都没有的天数 = %d（%.0f%%）**' % (span - len(g), 100 * (span - len(g)) / span))
    print('  每天 ≥1 局的概率 = %.0f%%；≥3 局的概率 = %.0f%%'
          % (100 * len(g) / span, 100 * (g >= 3).mean()))
