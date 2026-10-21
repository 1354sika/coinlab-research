"""
V16-SHUF：V16 的对照实验——把标签打乱后重跑四种表示。
这是验证 V16 那 0.660 是真是假的唯一办法（同时也是我上一轮漏掉的方法论步骤）。

做法：同一批 X，把 Y 用固定种子打乱（打断特征-标签对应关系），重跑四种表示。
判据：SHUF 的 AUC 应回落到 0.50 附近；若仍 0.6+，说明 V16 的 AUC 来自样本重叠/过拟合，不是真信号。
"""
import numpy as np, pandas as pd, time, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v16_SHUF.txt'

H, GAP, COST, REF, W, STEP, THRESH = 96, 24, 0.0004, 12, 48, 3, 0.01
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))
df['ret1'] = np.log(df.close.to_numpy() / df.groupby('symbol').close.shift(1).to_numpy())
mkt = df.groupby('open_time').ret1.median()

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

A_l, B_l, C_l, D_l, y_l, q_l = [], [], [], [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    o, h, l, c, v = P['o'], P['h'], P['l'], P['c'], P['v']
    lr1, sd24, volr = P['lr1'], P['sd24'], P['volr']
    q = qm_all[ei]
    for j in range(i, i + H - REF, STEP):
        lo = l[j+1:j+1+REF].min()
        y_l.append(1.0 if lo / c[j] - 1.0 < -THRESH else 0.0)
        q_l.append(str(q))
        seq = np.stack([o[j-47:j+1], h[j-47:j+1], l[j-47:j+1],
                        c[j-47:j+1], v[j-47:j+1]], axis=1) / c[j] - 1.0
        A_l.append(seq.ravel().astype(np.float32))
        denom = sd24[j] if (sd24[j] and sd24[j] > 1e-8) else 1e-4
        rn = lr1[j-47:j+1] / denom
        stat = np.array([rn.sum(), rn.std(), np.abs(rn).mean(),
                         (c[j]/c[j-12]-1.0), (c[j]/c[j-24]-1.0), (c[j]/c[j-48]-1.0),
                         volr[j], sd24[j]], dtype=np.float32)
        B_l.append(np.concatenate([rn.astype(np.float32), stat]))
        def agg(w_):
            r = lr1[j-w_+1:j+1]
            return float(r.sum()), float(r.std()), float(np.mean(np.abs(r)))
        r5, s5, a5 = agg(12); r4h, s4h, a4h = agg(48)
        C_l.append(np.array([
            r5, s5, a5, r5, s5, a5, r4h, s4h, a4h, r5/(s5+1e-6), r4h/(s4h+1e-6),
            float(c[j]/c[j-48]-1.0), float(volr[j]), float(sd24[j]),
            float((c[j]-h[j-48:j+1].max())/h[j-48:j+1].max()),
            float((c[j]-l[j-48:j+1].min())/l[j-48:j+1].min()),
            float(l[j]/c[j-1]-1.0), float(h[j]/c[j-1]-1.0)], dtype=np.float32))
        tt = P['t'][j-47:j+1]
        mm = mkt.reindex(tt).to_numpy(dtype=np.float32)
        D_l.append((lr1[j-47:j+1] - np.nan_to_num(mm, nan=0.0)).astype(np.float32))

Y = np.array(y_l, dtype=np.float32)
MQ = np.array(q_l)
tr = MQ < str(qs[VALID_FROM_Q]); te = ~tr
print('samples %d  train %d  test %d  %.1fs' % (len(Y), tr.sum(), te.sum(), time.time()-t0))

# 打乱标签
rng = np.random.default_rng(42)
Y_SHUF = rng.permutation(Y)

REPS = [('A 原始归一化', A_l), ('B 波动率归一化', B_l), ('C 多尺度', C_l), ('D 截面超额', D_l)]

lines = ['V16-SHUF：打乱标签后的对照（验证 V16 的 AUC 是真信号还是重叠/过拟合）', '']
lines.append('标签已打乱（seed=42）｜训练 %d / 验证 %d ｜ 打乱后正例率 %.1f%%' % (tr.sum(), te.sum(), 100*Y_SHUF.mean()))
lines.append('')
lines.append('%-20s %6s %12s %11s %11s' % ('表示', '维数', 'AUC[REAL]', 'AUC[SHUF]', '差值'))
lines.append('-' * 66)

for name, L in REPS:
    X = np.stack(L)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    lr = LogisticRegression(max_iter=400, C=1.0)
    lr.fit(Xtr, Y[tr])
    a_real = roc_auc_score(Y[te], lr.predict_proba(Xte)[:, 1])
    lr2 = LogisticRegression(max_iter=400, C=1.0)
    lr2.fit(Xtr, Y_SHUF[tr])
    a_shuf = roc_auc_score(Y_SHUF[te], lr2.predict_proba(Xte)[:, 1])
    lines.append('%-20s %6d %12.3f %11.3f %11.3f' % (name, X.shape[1], a_real, a_shuf, a_real - a_shuf))
    print('%-20s %6d real %.3f  shuf %.3f  diff %.3f' % (name, X.shape[1], a_real, a_shuf, a_real - a_shuf))
    del X, Xtr, Xte

lines.append('')
lines.append('判读: SHUF 应 ~0.50。若 SHUF 也 0.6+ ⇒ V16 的 0.660 来自重叠/过拟合，不是真信号。')
txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
