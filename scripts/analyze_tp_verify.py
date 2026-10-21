"""
TP(2%) 稳健性核查 —— 这个 +1.064%/9季正 是真突破还是样本假象？

必须查四件事：
  1. 逐季明细（9/9 是不是靠某一季撑起来的）
  2. 事件去重：同一币 24 小时内只保留一个入场点，看结果是否还在
  3. 分币稳定性：60 个币里有多少币为正
  4. 阈值敏感性：1%~5% 是否都为正（若不敏感，说明不是过拟合某个阈值）
全部零模型，纯规则。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/tp_verify_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H: return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm.unique().tolist()))


def theta_k(P, i, th=0.01):
    h, l, c = P['h'], P['l'], P['c']
    eh = h[i:i + H] / c[i] - 1.0
    el = l[i:i + H] / c[i] - 1.0
    hi = 0.0; s = H - 1
    for j in range(H):
        if eh[j] > hi: hi = eh[j]
        if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
            s = j; break
    return s


def run(x, dedup_h=0):
    """dedup_h>0 时，同一币 dedup_h 小时内只保留第一个入场"""
    out = []
    last = {}
    for ei, (tms, sym, i) in enumerate(entries):
        if str(qm[ei]) < str(qs[VALID_FROM_Q]):
            continue
        if dedup_h:
            if sym in last and tms - last[sym] < dedup_h * 3600_000:
                continue
            last[sym] = tms
        P = paths[sym]; c = P['c']
        r = c[i:i + H] / c[i] - 1.0
        k = theta_k(P, i)
        for j in range(H):
            if r[j] >= x:
                k = j; break
        out.append((str(qm[ei]), sym, float(r[k]) - COST))
    return pd.DataFrame(out, columns=['q', 'sym', 'net'])


L = []
L.append('TP(2%%) 稳健性核查  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('入场 ret24<=-2%% & volr>=3.0 | 首次收益>=x 卖，未达则 θ=1%% 兜底 | 成本 0.04%% 单边')
L.append('')

base = run(0.02)
L.append('=== 一、逐季（TP=2%%，不去重）')
g = base.groupby('q').net.agg(['mean', 'median', 'size'])
for q, row in g.iterrows():
    L.append('  %-8s 均值 %8.3f%%  中位 %8.3f%%  n=%4d'
             % (q, row['mean'] * 100, row['median'] * 100, int(row['size'])))
L.append('  合计: 均值 %.3f%%  中位 %.3f%%  季正 %d/%d'
         % (base.net.mean() * 100, base.net.median() * 100, int((g['mean'] > 0).sum()), len(g)))
L.append('')

L.append('=== 二、事件去重（同一币 N 小时内只留一个入场）')
L.append('  %-10s %8s %10s %10s %8s' % ('去重窗', '笔数', '均值/笔', '中位/笔', '季正'))
for dh in (0, 6, 24, 72):
    d = run(0.02, dh)
    if len(d) == 0:
        continue
    gg = d.groupby('q').net.mean()
    L.append('  %-10s %8d %9.3f%% %9.3f%% %6d/%d'
             % ('%dh' % dh if dh else '不去重', len(d), d.net.mean() * 100, d.net.median() * 100,
                int((gg > 0).sum()), len(gg)))
L.append('')

L.append('=== 三、分币稳定性（TP=2%%，不去重）')
per = base.groupby('sym').net.agg(['mean', 'size'])
L.append('  60 币中为正: %d 个；中位币均值 %.3f%%；最好 %.3f%%；最差 %.3f%%'
         % (int((per['mean'] > 0).sum()), per['mean'].median() * 100,
            per['mean'].max() * 100, per['mean'].min() * 100))
worst = per['mean'].nsmallest(3)
L.append('  最差 3 币: ' + '  '.join('%s %.2f%%(n=%d)' % (k, v * 100, int(per.loc[k, 'size']))
                                  for k, v in worst.items()))
L.append('')

L.append('=== 四、阈值敏感性（不去重）')
L.append('  %-8s %10s %10s %8s %8s' % ('止盈x', '均值/笔', '中位/笔', '季正', '触发率'))
for x in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
    d = run(x)
    gg = d.groupby('q').net.mean()
    L.append('  %-8s %9.3f%% %9.3f%% %6d/%d %7.1f%%'
             % ('%.1f%%' % (x * 100), d.net.mean() * 100, d.net.median() * 100,
                int((gg > 0).sum()), len(gg), 0))
L.append('')
L.append('诚实边界：')
L.append('  1. 成本 0.04%% 单边，若实为往返 0.08%%，全部均值再减 0.04pp。')
L.append('  2. 零滑点、零挂单成交率假设；止盈单假设能以限价成交。')
L.append('  3. 未算资金曲线与最大回撤（用户硬判据 <20% 尚未验证）。')
L.append('  4. 这是"止盈规则"，不含任何模型；与本轮"警觉"无关。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
