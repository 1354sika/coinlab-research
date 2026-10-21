"""
出场 V15：换标签——不猜「何时卖收益最高」，改猜「接下来会不会有回撤」。
V14 失败原因：标签基准率 46%（接近随机），学不到。
假设：回撤有波动率聚集结构，比方向更可学。

标签: future_maxdd = min(low[j+1..j+REF]) / close[j] - 1  （负数）
      二分类: maxdd < -THRESH  → 1（即将回撤），否则 0
特征: 过去 W 根 OHLCV（同 V14）
评估: AUC（能不能预测）+ 用它决策的收益（能不能赚）+ 对比固定持有
"""
import numpy as np, pandas as pd, time, pickle, os
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v15_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H, GAP, COST = 96, 24, 0.0004
VALID_FROM_Q = 4
DIP, VOLR = -0.02, 3.0
W, STEP, REF = 48, 3, 12
THRESH = 0.01          # 未来 12 根内回撤超过 1% 算「要跌」

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
    return dict(o=o, h=h, l=l, c=c, v=v, t=t, n=n, volr=volr, ret24=ret24)

paths = {}
entries = []
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

# ---------- 构造样本：标签 = 未来 REF 根内最大回撤 ----------
Xl, yl, ml, ddl = [], [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    o, h, l, c, v = P['o'], P['h'], P['l'], P['c'], P['v']
    q = qm_all[ei]
    for j in range(i, i + H - REF, STEP):
        lo = l[j+1:j+1+REF].min()
        maxdd = lo / c[j] - 1.0                      # <= 0
        lab = 1.0 if maxdd < -THRESH else 0.0
        seq = np.stack([o[j-W+1:j+1], h[j-W+1:j+1], l[j-W+1:j+1],
                        c[j-W+1:j+1], v[j-W+1:j+1]], axis=1)
        seq = (seq / c[j] - 1.0)
        Xl.append(seq.astype(np.float32))
        yl.append(lab); ddl.append(maxdd)
        ml.append((q, ei, j - i, c[j]/c[i] - 1.0))
X = np.stack(Xl); Y = np.array(yl, dtype=np.float32); DD = np.array(ddl, dtype=np.float32)
MQ = np.array([str(m[0]) for m in ml]); G = np.array([m[3] for m in ml], dtype=np.float32)
print('samples %d  X %s  %.1fs' % (len(X), X.shape, time.time()-t0))

tr = MQ < str(qs[VALID_FROM_Q]); te = ~tr
print('train %d  test %d  正例率 train %.1f%% test %.1f%%' %
      (tr.sum(), te.sum(), 100*Y[tr].mean(), 100*Y[te].mean()))

# ---------- 训练 ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 48, batch_first=True)
        self.fc = nn.Linear(48, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

torch.manual_seed(0)
net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
lossf = nn.BCEWithLogitsLoss()
ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
dl = DataLoader(ds, batch_size=512, shuffle=True)
t1 = time.time()
for ep in range(3):
    net.train(); tl = 0.0; nb = 0
    for xb, yb in dl:
        opt.zero_grad(); out = net(xb); loss = lossf(out, yb)
        loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print('epoch %d loss %.4f  %.0fs' % (ep+1, tl/nb, time.time()-t1))
net.eval()

with torch.no_grad():
    p_tr = torch.sigmoid(net(torch.from_numpy(X[tr]))).numpy()
    p_te = torch.sigmoid(net(torch.from_numpy(X[te]))).numpy()

auc_tr = roc_auc_score(Y[tr], p_tr)
auc_te = roc_auc_score(Y[te], p_te)
print('AUC train %.3f  test %.3f' % (auc_tr, auc_te))

# ---------- 评估 1：分位单调性（预测回撤 vs 实际回撤） ----------
dq = pd.qcut(pd.Series(p_te), 5, labels=False, duplicates='drop')
mono = pd.DataFrame({'bin': dq, 'dd': DD[te], 'y': Y[te]}).groupby('bin').agg(
    n=('dd','size'), mean_dd=('dd','mean'), posrate=('y','mean'))

# ---------- 评估 2：用它决策（p>阈值就卖） ----------
from collections import defaultdict
idx_te = np.where(te)[0]
te_rows = [(ml[k], float(p_te[n])) for n, k in enumerate(idx_te)]
dec = defaultdict(list)
for (q, ei, off, g), p in te_rows:
    dec[ei].append((off, g, p))

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

runs = {('V15 决策 p>%.2f' % th): run_dec(th) for th in (0.3, 0.5, 0.7)}

lines = ['出场 V15：预测「未来 %d 根内最大回撤」是否可学（换标签，不猜方向）' % REF, '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  序列长 %d 根  |  成本 %.2f%%' %
             (DIP*100, VOLR, qs[VALID_FROM_Q], W, COST*100))
lines.append('标签: 未来 %d 根内 min(low)/close-1 < -%.0f%%  |  训练 %d / 验证 %d' %
             (REF, THRESH*100, tr.sum(), te.sum()))
lines.append('')
lines.append('=== 能不能学到（AUC，0.5=瞎猜）')
lines.append('  AUC 训练 %.3f   验证 %.3f' % (auc_tr, auc_te))
lines.append('')
lines.append('=== 预测档 vs 实际回撤（验证段，按预测概率五等分）')
lines.append('%-8s %8s %12s %10s' % ('档','n','实际回撤均值','"要跌"占比'))
for b, row in mono.iterrows():
    lines.append('%-8d %8d %11.3f%% %9.1f%%' % (b, int(row.n), row.mean_dd*100, row.posrate*100))
lines.append('')
lines.append('=== 能不能赚（扣 %.2f%% 成本，逐笔）' % (COST*100))
om = E5.net.mean()
lines.append('%-24s %10s %10s %10s %8s' % ('方式','均值/笔','中位/笔','捕获率','季正'))
for name, d_ in [('E3 固定持 12 根', E3), ('E1 固定持 96 根', E1), ('E5 oracle 上界', E5)] + list(runs.items()):
    g = d_.groupby(d_.q).net.mean()
    lines.append('%-24s %9.3f%% %9.3f%% %9.1f%% %7d/%d' %
                 (name, d_.net.mean()*100, d_.net.median()*100,
                  d_.net.mean()/om*100 if om else 0, int((g>0).sum()), len(g)))
lines.append('')
lines.append('=== V15 决策 p>0.50 逐季')
for qq, row in runs['V15 决策 p>0.50'].groupby('q').net.agg(['mean','median','size']).iterrows():
    lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))
lines.append('')
lines.append('=== 预测分布（验证段）  p 均值 %.3f  >0.5 占比 %.1f%%' % (p_te.mean(), 100*(p_te>0.5).mean()))

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(net.state_dict(), os.path.join(MODEL_DIR, 'exit_v15_gru.pt'))
with open(os.path.join(MODEL_DIR, 'exit_v15_meta.pkl'), 'wb') as f:
    pickle.dump(dict(W=W, STEP=STEP, REF=REF, H=H, COST=COST, THRESH=THRESH,
                     auc_te=float(auc_te), valid_from=str(qs[VALID_FROM_Q]),
                     note='V15 预测未来12根最大回撤'), f)

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
