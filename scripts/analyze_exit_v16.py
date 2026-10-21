"""
出场 V16：问题是不是"给数据的方式不对"？
同一标签（未来 12 根内最大回撤 < -1%）、同一批样本、同一个简单模型（逻辑回归），
只换 4 种数据表示，看 AUC 能不能从 0.532 推上去。

表示 A（V15 基线）: 48 根 OHLCV / close - 1        → 240 维
表示 B（波动率归一化）: 48 根 log 收益 / 滚动 24 根 std + 状态量  → ~52 维
表示 C（多尺度）: 5m/1h/4h 三个尺度的收益、波动、偏度  → ~18 维
表示 D（截面超额）: 每根 5m 的收益 - 全市场同时刻中位数  → 48 维

判据：AUC（0.5 = 瞎猜）。若四种都卡在 0.55 以下 ⇒ 表示不是主因，数据本身没信息。
"""
import numpy as np, pandas as pd, time, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v16_REAL.txt'

H, GAP, COST, REF, W, STEP, THRESH = 96, 24, 0.0004, 12, 48, 3, 0.01
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))

df['ret1'] = np.log(df.close.to_numpy() / df.groupby('symbol').close.shift(1).to_numpy())
mkt = df.groupby('open_time').ret1.median()
print('market ret %.1fs  n=%d' % (time.time()-t0, len(mkt)))

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

# ---------- 构造四种表示 ----------
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

        # A: 原始归一化
        seq = np.stack([o[j-47:j+1], h[j-47:j+1], l[j-47:j+1],
                        c[j-47:j+1], v[j-47:j+1]], axis=1) / c[j] - 1.0
        A_l.append(seq.ravel().astype(np.float32))

        # B: log 收益 / 滚动波动率 + 状态量
        denom = sd24[j] if (sd24[j] and sd24[j] > 1e-8) else 1e-4
        rn = lr1[j-47:j+1] / denom
        stat = np.array([
            rn.sum(), rn.std(), np.abs(rn).mean(),
            (c[j]/c[j-12]-1.0), (c[j]/c[j-24]-1.0), (c[j]/c[j-48]-1.0),
            volr[j], sd24[j],
        ], dtype=np.float32)
        B_l.append(np.concatenate([rn.astype(np.float32), stat]))

        # C: 多尺度
        def agg(w_):
            r = lr1[j-w_+1:j+1]
            return float(r.sum()), float(r.std()), float(np.mean(np.abs(r)))
        r5, s5, a5 = agg(12)
        r1h, s1h, a1h = agg(12)          # 1h = 12 根 5m
        r4h, s4h, a4h = agg(48)          # 4h = 48 根 5m
        C_l.append(np.array([
            r5, s5, a5, r1h, s1h, a1h, r4h, s4h, a4h,
            r5/ (s5+1e-6), r4h/(s4h+1e-6),
            float(c[j]/c[j-48]-1.0), volr[j], float(sd24[j]),
            float((c[j]-h[j-48:j+1].max())/h[j-48:j+1].max()),
            float((c[j]-l[j-48:j+1].min())/l[j-48:j+1].min()),
            float(l[j]/c[j-1]-1.0), float(h[j]/c[j-1]-1.0),
        ], dtype=np.float32))

        # D: 截面超额
        tt = P['t'][j-47:j+1]
        mm = mkt.reindex(tt).to_numpy(dtype=np.float32)
        ex = lr1[j-47:j+1] - np.nan_to_num(mm, nan=0.0)
        D_l.append(ex.astype(np.float32))

Y = np.array(y_l, dtype=np.float32)
MQ = np.array(q_l)
tr = MQ < str(qs[VALID_FROM_Q]); te = ~tr
print('samples %d  train %d  test %d  posrate %.1f%%  %.1fs' %
      (len(Y), tr.sum(), te.sum(), 100*Y.mean(), time.time()-t0))

REPS = [('A 原始归一化', A_l), ('B 波动率归一化', B_l), ('C 多尺度', C_l), ('D 截面超额', D_l)]

lines = ['出场 V16：换数据表示（同一标签、同一样本、同一逻辑回归）', '']
lines.append('标签: 未来 %d 根内 min(low)/close-1 < -%.0f%%  |  入场: ret24<=%.0f%% 且 volr>=%.1f' %
             (REF, THRESH*100, DIP*100, VOLR))
lines.append('训练 %d / 验证 %d  |  验证自 %s  |  成本 %.2f%%' % (tr.sum(), te.sum(), qs[VALID_FROM_Q], COST*100))
lines.append('')
lines.append('%-20s %6s %9s %9s' % ('表示', '维数', 'AUC[train]', 'AUC[test]'))
lines.append('-' * 50)

for name, L in REPS:
    X = np.stack(L)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    lr = LogisticRegression(max_iter=400, C=1.0, n_jobs=-1)
    lr.fit(Xtr, Y[tr])
    p_tr = lr.predict_proba(Xtr)[:, 1]
    p_te = lr.predict_proba(Xte)[:, 1]
    a1, a2 = roc_auc_score(Y[tr], p_tr), roc_auc_score(Y[te], p_te)
    lines.append('%-20s %6d %9.3f %9.3f' % (name, X.shape[1], a1, a2))
    print('%-20s %6d %9.3f %9.3f' % (name, X.shape[1], a1, a2))
    del X, Xtr, Xte

lines.append('')
lines.append('判读: AUC 0.5=瞎猜；0.55 是"有但很弱"；>0.58 才算表示对了')
lines.append('若四种都 <=0.55 ⇒ 不是表示问题，是这份数据里没有可学信息')

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
