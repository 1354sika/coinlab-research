"""
出场 V20：跟踪止盈（trailing stop）——不预测顶部，等顶部形成后从最高点回撤 θ 就走。
为什么换这条路：V19 证明"预测未来最优"标签 90% 正、模型学不到任何条件信息（REAL≈SHUF）。
但 E5 oracle 3.574% vs E3 -0.048% 说明"出场时机"本身价值 3.6pp。
跟踪止盈不需要预测——它承认顶部不可知，只负责"顶部形成后不错过、也不恋战"。

θ 扫描 + 动态 θ（按入场时波动率缩放）+ 对比 E3/E1/oracle。
"""
import numpy as np, pandas as pd, time, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v20_REAL.txt'

H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))

def prep(g):
    g = g.sort_values('open_time')
    o = g.open.to_numpy(np.float32); h = g.high.to_numpy(np.float32)
    l = g.low.to_numpy(np.float32); c = g.close.to_numpy(np.float32)
    v = g.volume.to_numpy(np.float32); t = g.open_time.to_numpy()
    n = len(c)
    if n < 600 + H: return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    lr1 = np.zeros(n); lr1[1:] = np.log(c[1:] / c[:-1])
    sd24 = pd.Series(lr1).rolling(24, min_periods=24).std().to_numpy()
    return dict(o=o, h=h, l=l, c=c, v=v, t=t, n=n, volr=volr, ret24=ret24, lr1=lr1, sd24=sd24)

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
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('好入场 %d  验证自 %s  %.1fs' % (len(entries), qs[VALID_FROM_Q], time.time()-t0))

# 预计算每条路径
P_list = []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    c, h, l, sd24 = P['c'], P['h'], P['l'], P['sd24']
    gp = c[i:i+H] / c[i] - 1.0
    eh = h[i:i+H] / c[i] - 1.0
    el = l[i:i+H] / c[i] - 1.0
    runmax = np.maximum.accumulate(eh)
    sd0 = float(sd24[i]) if np.isfinite(sd24[i]) else 0.01
    P_list.append(dict(ei=ei, q=str(qm_all[ei]), gp=gp, eh=eh, el=el,
                       runmax=runmax, sd0=sd0, n=len(gp)))

def trailing(theta, mode='fixed', k=2.0):
    """从最高点回撤 theta 就走。mode='fixed' 用固定 θ；'vol' 用 k*sd0。"""
    res = []
    for p in P_list:
        if p['q'] < str(qs[VALID_FROM_Q]): continue
        gp, eh, el, runmax, sd0 = p['gp'], p['eh'], p['el'], p['runmax'], p['sd0']
        th = theta if mode == 'fixed' else max(0.003, k * sd0 * 24**0.5)
        # 逐根：先更新高点，再看回撤（用 low 判断触发）
        hi = 0.0; sell = len(gp) - 1
        for j in range(len(gp)):
            if eh[j] > hi: hi = eh[j]
            dd = (1.0 + el[j]) / (1.0 + hi) - 1.0
            if j >= 1 and dd <= -th:
                sell = j; break
        res.append((p['q'], float(gp[sell]) - COST, sell))
    return pd.DataFrame(res, columns=['q','net','off'])

def fixed(hold):
    out = []
    for p in P_list:
        if p['q'] < str(qs[VALID_FROM_Q]): continue
        off = min(hold - 1, p['n'] - 1)
        out.append((p['q'], float(p['gp'][off]) - COST))
    return pd.DataFrame(out, columns=['q','net'])

orc = []
for p in P_list:
    if p['q'] < str(qs[VALID_FROM_Q]): continue
    orc.append((p['q'], float(p['eh'].max()) - COST))
E5 = pd.DataFrame(orc, columns=['q','net'])
E3 = fixed(12); E1 = fixed(96)

variants = []
for th in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
    variants.append(('V20 跟踪止盈 θ=%.1f%%' % (th*100), trailing(th, 'fixed')))
for k in (2.0, 3.0, 4.0):
    variants.append(('V20 动态 θ=%.1f×σ' % k, trailing(k, 'vol', k)))

lines = ['出场 V20：跟踪止盈（不预测顶部，从最高点回撤 θ 就走）', '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  成本 %.2f%%' % (DIP*100, VOLR, qs[VALID_FROM_Q], COST*100))
lines.append('')
om = E5.net.mean()
lines.append('%-28s %10s %10s %10s %8s' % ('方式','均值/笔','中位/笔','捕获率','季正'))
for name, d_ in [('E3 固定持 12 根', E3), ('E1 固定持 96 根', E1), ('E5 oracle 上界', E5)] + variants:
    g = d_.groupby(d_.q).net.mean()
    lines.append('%-28s %9.3f%% %9.3f%% %9.1f%% %7d/%d' %
                 (name, d_.net.mean()*100, d_.net.median()*100,
                  d_.net.mean()/om*100 if om else 0, int((g>0).sum()), len(g)))

# 逐季：取最好的固定 θ
best = max([(v[1].net.mean(), v[0], v[1]) for v in variants if 'θ=%.1f%%' % 0 in v[0] or True],
           key=lambda x: x[0])
lines.append('')
lines.append('=== 逐季（%s）' % best[1])
for qq, row in best[2].groupby('q').net.agg(['mean','median','size']).iterrows():
    lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
