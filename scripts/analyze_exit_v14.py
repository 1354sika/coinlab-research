"""
出场 V14：序列模型（GRU 直接吃原始 K 线）能不能学到人没发现的规律？
对照 V13：同一批好入场，只换出场。
  E3 固定持 12 根 是当前最好的简单规则（-0.049%/笔）
  V14 = GRU 看过去 48 根 OHLCV，学「现在卖是否比 12 根后卖更好」
标签不是 oracle：标签 = g_now > g_{j+12}（两个都可实现的动作）
评估：时间外（验证自 2024Q4），与固定持有、oracle 对比
"""
import numpy as np, pandas as pd, time, pickle, os
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v14_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H, GAP, COST = 96, 24, 0.0004
VALID_FROM_Q = 4
DIP, VOLR = -0.02, 3.0
W, STEP, REF = 48, 3, 12

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

# ---------- 构造样本 ----------
Xl, yl, ml = [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    o, h, l, c, v = P['o'], P['h'], P['l'], P['c'], P['v']
    q = qm_all[ei]
    for j in range(i, i + H - REF, STEP):
        g_now = c[j] / c[i] - 1.0
        g_ref = c[j + REF] / c[i] - 1.0
        lab = 1.0 if g_now > g_ref else 0.0
        seq = np.stack([o[j-W+1:j+1], h[j-W+1:j+1], l[j-W+1:j+1],
                        c[j-W+1:j+1], v[j-W+1:j+1]], axis=1)
        seq = (seq / c[j] - 1.0)
        Xl.append(seq.astype(np.float32))
        yl.append(lab)
        ml.append((q, ei, j - i, g_now))
X = np.stack(Xl); Y = np.array(yl, dtype=np.float32)
MQ = np.array([str(m[0]) for m in ml])
print('samples %d  X %s  %.1fs' % (len(X), X.shape, time.time()-t0))

tr = MQ < str(qs[VALID_FROM_Q])
te = ~tr
print('train %d  test %d' % (tr.sum(), te.sum()))

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
net = Net()
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
lossf = nn.BCEWithLogitsLoss()
ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
dl = DataLoader(ds, batch_size=512, shuffle=True)
t1 = time.time()
for ep in range(2):
    net.train(); tl = 0.0; nb = 0
    for xb, yb in dl:
        opt.zero_grad(); out = net(xb); loss = lossf(out, yb)
        loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print('epoch %d loss %.4f  %.0fs' % (ep+1, tl/nb, time.time()-t1))
net.eval()

# ---------- 评估 ----------
with torch.no_grad():
    p_te = torch.sigmoid(net(torch.from_numpy(X[te]))).numpy()

# 决策模拟：每个入场点，从 i 开始每 STEP 根看一次，p>0.5 就卖
from collections import defaultdict
idx_te = np.where(te)[0]
te_rows = [ml[k] for k in idx_te]
dec = defaultdict(list)
for k, (q, ei, off, g) in enumerate(te_rows):
    dec[ei].append((off, g, float(p_te[k])))

res = []
for ei, lst in dec.items():
    lst.sort()
    done = False
    for off, g, p in lst:
        if p > 0.5:
            res.append((qm_all[ei], g - COST)); done = True; break
    if not done:
        off, g, p = lst[-1]
        res.append((qm_all[ei], g - COST))
rq = pd.DataFrame(res, columns=['q','net'])
rq['q'] = rq.q.astype(str)

# 对照：固定持有 12 根 / 96 根 / oracle
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

lines = ['出场 V14：GRU 序列模型（原始 K 线 → 卖/持）', '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  序列长 %d 根  |  成本 %.2f%%' %
             (DIP*100, VOLR, qs[VALID_FROM_Q], W, COST*100))
lines.append('标签: 现在卖(g_now) 是否优于 12 根后卖(g_{j+12})  |  训练 %d / 验证 %d' % (tr.sum(), te.sum()))
lines.append('')
om = E5.net.mean()
lines.append('%-24s %10s %10s %10s %8s' % ('方式','均值/笔','中位/笔','捕获率','季正'))
for name, d_ in [('V14 GRU 模型', rq), ('E3 固定持 12 根', E3), ('E1 固定持 96 根', E1), ('E5 oracle 上界', E5)]:
    g = d_.groupby(d_.q).net.mean()
    lines.append('%-24s %9.3f%% %9.3f%% %9.1f%% %7d/%d' %
                 (name, d_.net.mean()*100, d_.net.median()*100,
                  d_.net.mean()/om*100 if om else 0, int((g>0).sum()), len(g)))

lines.append('')
lines.append('=== V14 逐季')
for qq, row in rq.groupby('q').net.agg(['mean','median','size']).iterrows():
    lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))

lines.append('')
lines.append('=== 模型预测分布（验证段）')
lines.append('  p>0.5 占比 %.1f%%  p 均值 %.3f' % (100*(p_te>0.5).mean(), p_te.mean()))

lines.append('')
lines.append('=== 标签基准率')
lines.append('  训练段「现在卖更好」占比 %.1f%%  验证段 %.1f%%' %
             (100*Y[tr].mean(), 100*Y[te].mean()))

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(net.state_dict(), os.path.join(MODEL_DIR, 'exit_v14_gru.pt'))
with open(os.path.join(MODEL_DIR, 'exit_v14_meta.pkl'), 'wb') as f:
    pickle.dump(dict(W=W, STEP=STEP, REF=REF, H=H, COST=COST,
                     valid_from=str(qs[VALID_FROM_Q]),
                     entry_rule='ret24<=%.2f & volr>=%.1f' % (DIP, VOLR),
                     note='V14 GRU 序列模型：过去48根OHLCV → P(现在卖优于12根后卖)'), f)

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
