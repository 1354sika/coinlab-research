"""
出场·补充：V20 的"跟踪止盈"是否真的在跟随？

上一步测到 θ<=1.0% 时持有时长中位只有 1 根 —— 若属实，V20 实际是"入场即离场"，
并未像报告所说"跟住上涨"。本脚本回答两件事：
  1. θ=1.0% 时出场时刻的真实分布（多少笔在第 1~2 根就走了）
  2. 强制最小持有 minhold 根后再允许触发，收益怎么变（若是"没跟住"，放宽 minhold 应显著改变结果）

入场/数据同 V20。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/exit_peak2_REAL.txt'

H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('rows %d load %.1fs' % (len(df), time.time() - t0))


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H:
        return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None:
        continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('入场 %d 验证自 %s' % (len(entries), qs[VALID_FROM_Q]))

val = []
for ei, (tms, sym, i) in enumerate(entries):
    if str(qm_all[ei]) < str(qs[VALID_FROM_Q]):
        continue
    P = paths[sym]; c, h, l = P['c'], P['h'], P['l']
    val.append(dict(q=str(qm_all[ei]),
                    gp=c[i:i + H] / c[i] - 1.0,
                    eh=h[i:i + H] / c[i] - 1.0,
                    el=l[i:i + H] / c[i] - 1.0,
                    n=H))
print('验证集 %d  %.1fs' % (len(val), time.time() - t0))


def run(theta, minhold=1, cap=None):
    net, sell, cape = [], [], []
    for p in val:
        gp, eh, el = p['gp'], p['eh'], p['el']
        n = p['n']; hi = 0.0; s = n - 1
        for j in range(n):
            if eh[j] > hi:
                hi = eh[j]
            if j >= minhold:
                dd = (1.0 + el[j]) / (1.0 + hi) - 1.0
                if dd <= -theta:
                    s = j
                    break
            if cap is not None and j >= cap:
                s = cap
                break
        net.append(float(gp[s]) - COST)
        sell.append(s)
        cape.append((1.0 + gp[s]) / (1.0 + float(eh.max())) - 1.0)
    return np.array(net), np.array(sell), np.array(cape)


L = []
L.append('出场·补充：V20 跟踪止盈是否真的在"跟随"')
L.append('入场 ret24<=-2%% & volr>=3.0 | H=96根 | 验证自 %s | %d 笔 | 成本0.04%%单边'
         % (qs[VALID_FROM_Q], len(val)))
L.append('')

net, sell, cape = run(0.01, 1)
L.append('=== 一、θ=1.0%% 时出场时刻分布（中位持有 %d 根 = %d 分钟）'
         % (int(np.median(sell)), int(np.median(sell)) * 5))
for lo, hi_, name in ((0, 0, '第1根'), (1, 1, '第2根'), (2, 5, '第3~6根'),
                      (6, 20, '第7~21根'), (21, 95, '第22根以后')):
    m = (sell >= lo) & (sell <= hi_)
    L.append('  %-10s %6d 笔  %5.1f%%  均值 %7.3f%%' % (name, int(m.sum()), m.mean() * 100,
                                                     net[m].mean() * 100 if m.any() else 0))
L.append('')

L.append('=== 二、强制最小持有 minhold 根后才允许 θ=1.0%% 触发')
L.append('  %-8s %10s %10s %8s %12s %10s' % ('minhold', '均值/笔', '中位/笔', '季正', '离峰中位', '出场中位'))
for mh in (1, 2, 3, 6, 12, 24, 48):
    net2, sell2, cape2 = run(0.01, mh)
    qq = pd.Series([p['q'] for p in val])
    g = pd.DataFrame({'q': qq, 'net': net2}).groupby('q').net.mean()
    L.append('  %-9d %9.3f%% %9.3f%% %6d/%d %11.2f%% %9d根'
             % (mh, net2.mean() * 100, np.median(net2) * 100, int((g > 0).sum()), len(g),
                np.median(cape2) * 100, int(np.median(sell2))))
L.append('')

L.append('=== 三、对照：同样的 minhold 但换 θ（看是"θ 问题"还是"minhold 问题"）')
L.append('  %-8s %10s %10s %10s %10s' % ('minhold', 'θ=0.5%', 'θ=1.0%', 'θ=2.0%', 'θ=4.0%'))
for mh in (1, 6, 12, 24):
    vals = [run(th, mh)[0].mean() * 100 for th in (0.005, 0.01, 0.02, 0.04)]
    L.append('  %-9d %9.3f%% %9.3f%% %9.3f%% %9.3f%%' % (mh, *vals))
L.append('')
L.append('读法：若 minhold 从 1 加到 12/24 收益不升反降，说明"没跟住"不是主因，')
L.append('      θ=1%% 的 +0.053%% 本来就来自那些"刚入场就反转"的少数笔。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time() - t0))
