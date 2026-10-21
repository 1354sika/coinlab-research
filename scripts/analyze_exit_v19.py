"""
出场 V19：直接优化决策——不预测任何概率，只学"卖了之后后悔不后悔"。

标签: 后悔值 A_j = max_{k>j}(g_k - COST) - (g_j - COST)
      A_j > 0 ⇒ 继续持有还有更好的卖点；A_j <= 0 ⇒ 现在就是最好，该走
模型: 梯度提升回归预测 A_j（连续值，不是 50/50 二分类）
决策: 预测 A > 0 → 继续持；否则卖
必带: SHUF 对照 + 触发率 + 去重版（每条路径只取首个决策点）

与 V17/V18 的本质区别：V17 预测"会不会跌"（中间变量），V19 直接预测"该不该卖"（决策本身）。
"""
import numpy as np, pandas as pd, time, os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v19_REAL.txt'

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

# ---------- 特征（表示 C）+ 后悔值标签 ----------
Xl, Al, gl, ml = [], [], [], []
gpath_all = {}
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c = P['h'], P['l'], P['c']
    lr1, sd24, volr = P['lr1'], P['sd24'], P['volr']
    gp = c[i:i+H] / c[i] - 1.0
    gpath_all[ei] = (gp, i, c, h, l, lr1, sd24, volr)
    for j in range(i, i + H - REF, STEP):
        off = j - i
        g_now = float(gp[off])
        fut = gp[off+1:]
        A = float(fut.max()) - g_now if len(fut) else 0.0
        def agg(w_):
            r = lr1[j-w_+1:j+1]
            return float(r.sum()), float(r.std()), float(np.mean(np.abs(r)))
        r5, s5, a5 = agg(12); r4h, s4h, a4h = agg(48)
        Xl.append(np.array([
            r5, s5, a5, r5, s5, a5, r4h, s4h, a4h, r5/(s5+1e-6), r4h/(s4h+1e-6),
            float(c[j]/c[j-48]-1.0), float(volr[j]), float(sd24[j]),
            float((c[j]-h[j-48:j+1].max())/h[j-48:j+1].max()),
            float((c[j]-l[j-48:j+1].min())/l[j-48:j+1].min()),
            float(l[j]/c[j-1]-1.0), float(h[j]/c[j-1]-1.0),
            g_now, float(off),  # 当前浮盈 + 已持有根数
        ], dtype=np.float32))
        Al.append(A); gl.append(g_now)
        ml.append((str(qm_all[ei]), ei, off))
X = np.stack(Xl); A = np.array(Al, dtype=np.float32); G = np.array(gl, dtype=np.float32)
MQ = np.array([m[0] for m in ml]); OFF = np.array([m[2] for m in ml])
tr = MQ < str(qs[VALID_FROM_Q]); te = ~tr
print('samples %d  train %d  test %d  A>0 占比 %.1f%%  %.1fs' %
      (len(X), tr.sum(), te.sum(), 100*(A>0).mean(), time.time()-t0))

# ---------- 训练 ----------
reg = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08,
                                subsample=0.8, random_state=42)
reg.fit(X[tr], A[tr])
pred_te = reg.predict(X[te])
print('RMSE train %.5f  test %.5f' % (
    mean_squared_error(A[tr], reg.predict(X[tr]))**0.5,
    mean_squared_error(A[te], pred_te)**0.5))

rng = np.random.default_rng(42)
A_shuf = rng.permutation(A)
reg2 = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08,
                                 subsample=0.8, random_state=42)
reg2.fit(X[tr], A_shuf[tr])
pred_shuf = reg2.predict(X[te])

# ---------- 决策 ----------
dec = defaultdict(list)
for n, k in enumerate(np.where(te)[0]):
    q, ei, off = ml[k]
    dec[ei].append((off, float(pred_te[n]), float(pred_shuf[n])))

def run_dec(preds, thr, key):
    res = []; trig = 0
    for ei, lst in dec.items():
        lst.sort()
        sold = False
        for off, p, ps in lst:
            pv = p if key == 'real' else ps
            if pv <= thr:   # 预测后悔值 <= 阈值 ⇒ 现在就是最好，卖
                gp, i0, c_, h_, l_, *_ = gpath_all[ei]
                res.append((str(qm_all[ei]), float(gp[off]) - COST)); sold = True; trig += 1; break
        if not sold:
            gp, i0, c_, h_, l_, *_ = gpath_all[ei]
            res.append((str(qm_all[ei]), float(gp[-1]) - COST))
    return pd.DataFrame(res, columns=['q','net']), trig

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

res_tab = []
for th in (0.0, 0.002, 0.005):
    d_real, tr_real = run_dec(pred_te, th, 'real')
    d_shuf, tr_shuf = run_dec(pred_te, th, 'shuf')
    res_tab.append(('V19 A>%.3f REAL' % th, d_real, tr_real))
    res_tab.append(('V19 A>%.3f SHUF' % th, d_shuf, tr_shuf))

lines = ['出场 V19：直接优化决策（学"卖了后悔不后悔"，不是预测概率）', '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  成本 %.2f%%' % (DIP*100, VOLR, qs[VALID_FROM_Q], COST*100))
lines.append('标签: 后悔值 A_j = max(未来收益) - 当前收益  |  A>0 占比 %.1f%%' % (100*(A>0).mean()))
lines.append('')
om = E5.net.mean()
lines.append('%-26s %10s %10s %8s %8s %8s' % ('方式','均值/笔','中位/笔','触发率','季正','SHUF差'))
for name, d_, tg in [('E3 固定持 12 根', E3, None), ('E1 固定持 96 根', E1, None), ('E5 oracle 上界', E5, None)] + res_tab:
    g = d_.groupby(d_.q).net.mean()
    trig = '%d' % tg if tg is not None else '-'
    lines.append('%-26s %9.3f%% %9.3f%% %8s %5d/%d %8s' %
                 (name, d_.net.mean()*100, d_.net.median()*100, trig, int((g>0).sum()), len(g), ''))
lines.append('')
lines.append('=== 逐季（A>0 REAL）')
d0, _ = run_dec(pred_te, 0.0, 'real')
for qq, row in d0.groupby('q').net.agg(['mean','median','size']).iterrows():
    lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))
lines.append('')
lines.append('=== 去重版：每条路径只用首个决策点')
first = defaultdict(lambda: (999, 0.0))
for n, k in enumerate(np.where(te)[0]):
    q, ei, off = ml[k]
    if off < first[ei][0]:
        first[ei] = (off, float(pred_te[n]))
res_first = []
for ei, (off, p) in first.items():
    gp, i0, c_, h_, l_, *_ = gpath_all[ei]
    if p <= 0.0:
        res_first.append((str(qm_all[ei]), float(gp[off]) - COST))
    else:
        res_first.append((str(qm_all[ei]), float(gp[-1]) - COST))
dF = pd.DataFrame(res_first, columns=['q','net'])
gF = dF.groupby('q').net.mean()
lines.append('  均值 %.3f%%  中位 %.3f%%  季正 %d/%d  n=%d' %
             (dF.net.mean()*100, dF.net.median()*100, int((gF>0).sum()), len(gF), len(dF)))

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
