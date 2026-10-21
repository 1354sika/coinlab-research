"""
出场 V18：表示 C（多尺度 18 维，AUC 0.660）+ 决策标签（现在卖是否优于 12 根后卖）。
V15 用 GRU + 表示 A 做这个标签失败（AUC 0.532）；V17 用表示 C + LR 预测回撤成功但决策亏。
V18 = 表示 C + LR + 决策标签：标签与决策对齐，且表示对了。

标签: g_now > g_{j+12}（两个都可实现的动作，非 oracle）
决策: p > thr 就卖，否则持到窗口末
"""
import numpy as np, pandas as pd, time, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v18_REAL.txt'

H, GAP, COST, REF, W, STEP = 96, 24, 0.0004, 12, 48, 3
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

# ---------- 表示 C + 决策标签 ----------
Xl, yl, ml = [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c = P['h'], P['l'], P['c']
    lr1, sd24, volr = P['lr1'], P['sd24'], P['volr']
    for j in range(i, i + H - REF, STEP):
        g_now = c[j]/c[i] - 1.0
        g_ref = c[j+REF]/c[i] - 1.0
        yl.append(1.0 if g_now > g_ref else 0.0)
        def agg(w_):
            r = lr1[j-w_+1:j+1]
            return float(r.sum()), float(r.std()), float(np.mean(np.abs(r)))
        r5, s5, a5 = agg(12)
        r4h, s4h, a4h = agg(48)
        Xl.append(np.array([
            r5, s5, a5, r5, s5, a5, r4h, s4h, a4h,
            r5/(s5+1e-6), r4h/(s4h+1e-6),
            float(c[j]/c[j-48]-1.0), float(volr[j]), float(sd24[j]),
            float((c[j]-h[j-48:j+1].max())/h[j-48:j+1].max()),
            float((c[j]-l[j-48:j+1].min())/l[j-48:j+1].min()),
            float(l[j]/c[j-1]-1.0), float(h[j]/c[j-1]-1.0),
        ], dtype=np.float32))
        ml.append((str(qm_all[ei]), ei, j - i, float(g_now)))
X = np.stack(Xl); Y = np.array(yl, dtype=np.float32)
MQ = np.array([m[0] for m in ml])
tr = MQ < str(qs[VALID_FROM_Q]); te = ~tr
print('samples %d  train %d  test %d  posrate %.1f%%  %.1fs' %
      (len(X), tr.sum(), te.sum(), 100*Y.mean(), time.time()-t0))

sc = StandardScaler().fit(X[tr])
lr = LogisticRegression(max_iter=800, C=1.0)
lr.fit(sc.transform(X[tr]), Y[tr])
p_te = lr.predict_proba(sc.transform(X[te]))[:, 1]
print('AUC test %.3f' % roc_auc_score(Y[te], p_te))

dec = defaultdict(list)
for n, k in enumerate(np.where(te)[0]):
    q, ei, off, g = ml[k]
    dec[ei].append((off, g, float(p_te[n])))

def run_dec(thr):
    res = []
    for ei, lst in dec.items():
        lst.sort()
        sold = False
        for off, g, p in lst:
            if p > thr:
                res.append((str(qm_all[ei]), g - COST)); sold = True; break
        if not sold:
            res.append((str(qm_all[ei]), lst[-1][1] - COST))
    return pd.DataFrame(res, columns=['q','net'])

def fixed(hold):
    out = []
    for ei, (tms, sym, i) in enumerate(entries):
        q = qm_all[ei]
        if q < qs[VALID_FROM_Q]: continue
        P = paths[sym]
        out.append((str(q), P['c'][i+hold-1]/P['c'][i] - 1.0 - COST))
    return pd.DataFrame(out, columns=['q','net'])

E3 = fixed(12); E1 = fixed(96)
orc = []
for ei, (tms, sym, i) in enumerate(entries):
    q = qm_all[ei]
    if q < qs[VALID_FROM_Q]: continue
    P = paths[sym]
    orc.append((str(q), P['h'][i:i+H].max()/P['c'][i] - 1.0 - COST))
E5 = pd.DataFrame(orc, columns=['q','net'])

runs = {('V18 决策 p>%.2f' % th): run_dec(th) for th in (0.4, 0.5, 0.6, 0.7)}

lines = ['出场 V18：表示 C（多尺度 18 维）+ 决策标签（现在卖是否优于 12 根后卖）', '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  成本 %.2f%%' % (DIP*100, VOLR, qs[VALID_FROM_Q], COST*100))
lines.append('AUC[验证] %.3f  |  训练 %d / 验证 %d  |  标签基准率 %.1f%%' %
             (roc_auc_score(Y[te], p_te), tr.sum(), te.sum(), 100*Y.mean()))
lines.append('')
om = E5.net.mean()
lines.append('%-24s %10s %10s %10s %8s' % ('方式','均值/笔','中位/笔','捕获率','季正'))
for name, d_ in [('E3 固定持 12 根', E3), ('E1 固定持 96 根', E1), ('E5 oracle 上界', E5)] + list(runs.items()):
    g = d_.groupby(d_.q).net.mean()
    lines.append('%-24s %9.3f%% %9.3f%% %9.1f%% %7d/%d' %
                 (name, d_.net.mean()*100, d_.net.median()*100,
                  d_.net.mean()/om*100 if om else 0, int((g>0).sum()), len(g)))
lines.append('')
for th in (0.5, 0.6, 0.7):
    d_ = runs['V18 决策 p>%.2f' % th]
    lines.append('=== V18 p>%.2f 逐季' % th)
    for qq, row in d_.groupby('q').net.agg(['mean','median','size']).iterrows():
        lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))
    lines.append('')

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
