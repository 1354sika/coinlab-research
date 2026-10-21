"""
两个问题一起答：
  1. 机械止盈一个月有几次机会？（频率）
  2. 人类入局者说"一半多下注曾到过 50% 收益，因贪没走，最后不亏不赚"——数据上成立吗？
     如果绝大多数笔的最大浮盈都能到 20%~50%，那问题就变成"止盈线设在哪"。

做法：
  A 频率：按自然月统计入场数、TP(2%) 触发数、每币每月
  B 长窗口浮盈：入场后 1d/3d/7d/30d 内的最大浮盈（用 high）与最终收益（用 close）
  C 对照：到 50% 就卖 vs 到 20% 就卖 vs 持有 30 天
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/freq_and_50pct.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
D1, D3, D7, D30 = 288, 864, 2016, 8640

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + D30: return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - D30 - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
mon = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('M')
qs = sorted(set(qm.unique().tolist()))
print('入场 %d（30d 窗口版）  %.1fs' % (len(entries), time.time() - t0), flush=True)

val = [(ei, e) for ei, e in enumerate(entries) if str(qm[ei]) >= str(qs[VALID_FROM_Q])]
print('验证集 %d' % len(val), flush=True)

rows = []
for ei, (tms, sym, i) in val:
    P = paths[sym]; h, c = P['h'], P['c']
    e0 = float(c[i])
    r = {}
    for tag, k in (('1d', D1), ('3d', D3), ('7d', D7), ('30d', D30)):
        seg_h = h[i + 1:i + 1 + k]
        seg_c = c[i + 1:i + 1 + k]
        if len(seg_h) < k:
            break
        r[tag + '_max'] = float(seg_h.max() / e0 - 1.0)
        r[tag + '_end'] = float(seg_c[-1] / e0 - 1.0)
    if '30d_max' not in r:
        continue
    r.update(q=str(qm[ei]), m=str(mon[ei]), sym=sym)
    rows.append(r)

D = pd.DataFrame(rows)
print('有效样本 %d' % len(D), flush=True)

L = []
L.append('频率 + "50%%" 检验  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('入场: ret24<=-2%% & volr>=3.0（急跌2%%+放量3倍）| 5m | 60 币 | 验证自 %s' % qs[VALID_FROM_Q])
L.append('样本 %d 笔（要求入场后至少还有 30 天数据）' % len(D))
L.append('')

L.append('=== 一、频率：一个月几次机会')
pm = D.groupby('m').size()
L.append('  平均每月入场 %.1f 次（60 币合计），中位 %.0f 次，最多 %d 次，最少 %d 次'
         % (pm.mean(), pm.median(), pm.max(), pm.min()))
L.append('  按 60 币摊：每币每月 %.2f 次' % (pm.mean() / 60))
L.append('  每月明细（前 12 个月）:')
for m, n in list(pm.items())[:12]:
    L.append('    %s  %3d 次' % (m, n))
L.append('')

L.append('=== 二、"一半多能到 50%%" 成立吗（入场后 30 天内的最大浮盈）')
for th in (0.10, 0.20, 0.30, 0.50, 1.00):
    L.append('  最大浮盈 >= %5.0f%%  占比 %5.1f%%' % (th * 100, (D['30d_max'] >= th).mean() * 100))
L.append('  最大浮盈 中位 %.1f%%  均值 %.1f%%' % (D['30d_max'].median() * 100, D['30d_max'].mean() * 100))
L.append('  30 天后最终收益 中位 %.1f%%  均值 %.1f%%' % (D['30d_end'].median() * 100, D['30d_end'].mean() * 100))
L.append('')

L.append('=== 三、各时间窗内"曾经到过 50%%"的比例')
for tag in ('1d', '3d', '7d', '30d'):
    col = tag + '_max'
    if col in D:
        L.append('  %-4s 内最大浮盈 >=50%% 的比例 %5.1f%%   中位最大浮盈 %5.1f%%'
                 % (tag, (D[col] >= 0.5).mean() * 100, D[col].median() * 100))
L.append('')

L.append('=== 四、止盈线设在哪：到线就卖 vs 持有')
L.append('  %-16s %10s %10s %8s' % ('策略', '均值/笔', '中位/笔', '季正'))
L.append('  %-16s %9.3f%% %9.3f%% %6d/%d' % ('持有 30 天', D['30d_end'].mean() * 100,
                                              D['30d_end'].median() * 100,
                                              int((D.groupby('q')['30d_end'].mean() > 0).sum()),
                                              D.q.nunique()))
for th in (0.10, 0.20, 0.30, 0.50):
    col = '30d_max'
    net = np.where(D[col] >= th, th, D['30d_end']) - COST
    g = pd.DataFrame({'q': D.q, 'net': net}).groupby('q').net.mean()
    L.append('  %-16s %9.3f%% %9.3f%% %6d/%d' % ('到 %d%% 就卖' % (th * 100), net.mean() * 100,
                                                 np.median(net) * 100, int((g > 0).sum()), len(g)))
L.append('')
L.append('注：到线就卖用"线值"近似成交价（实际上影可能更高），是保守估计；')
L.append('    真实执行需挂限价单，成交率与滑点未计。')
L.append('诚实边界：样本重叠（STEP=24）；成本 0.04% 单边；零滑点；未算资金曲线与回撤。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
