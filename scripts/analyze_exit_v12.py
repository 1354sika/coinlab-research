"""
出场 V12：趋势过滤（解决「阴跌市里抄底」）
D1 ret288>0       —— 24 小时趋势向上才买
D2 pos96>0.3      —— 不在 96 根(8h)低位区才买
D3 D1 且 D2
D4 D3 + 熔断 A（单根跌5% ≥5 币停手）
基线 = V9
"""
import numpy as np, pandas as pd, time, pickle, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v12_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H, GAP, COST = 96, 24, 0.0004
ALPHA_START, ALPHA_END = 0.15, 0.02
GAMMA = 0.97
EPS_START, EPS_END = 0.25, 0.02
VALID_FROM_Q = 4
DIP, VOLR = -0.02, 3.0
CRASH_RET1, CRASH_N = -0.05, 5

def d_gain(g):  return int(np.digitize(g, [-0.02, -0.005, 0.005, 0.02]))
def d_hold(k):  return min(2, k // 32)
def d_vol(x):   return int(np.digitize(x, [0.008]))
def d_moms(x):  return int(np.digitize(x, [-0.01, 0.01]))
def d_moml(x):  return int(np.digitize(x, [-0.03, 0.03]))
def d_pos(x):   return int(np.digitize(x, [0.33, 0.67]))
def d_vr(x):    return int(np.digitize(x, [1.0]))
N_S = 5 * 3 * 2 * 3 * 3 * 3 * 2
def sid(a, b, c, d, e, f, g):
    return ((((((a*3 + b)*2 + c)*3 + d)*3 + e)*3 + f)*2 + g)

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))

def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(float); l = g.low.to_numpy(float)
    c = g.close.to_numpy(float); v = g.volume.to_numpy(float)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H: return None
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr  = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vma  = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    def lagret(k):
        r = np.zeros(n); r[k:] = c[k:] / c[:-k] - 1.0; return r
    ret1 = lagret(1); ret6 = lagret(6); ret24 = lagret(24); ret96 = lagret(96); ret288 = lagret(288)
    mn96 = pd.Series(l).rolling(96, min_periods=96).min().to_numpy()
    mx96 = pd.Series(h).rolling(96, min_periods=96).max().to_numpy()
    rng96 = np.where(mx96 > mn96, mx96 - mn96, np.nan)
    pos96 = np.clip((c - mn96) / rng96, 0, 1)
    ratio = np.where(c > 0, atr / c, np.nan)
    return dict(c=c, t=t, n=n, ret1=ret1, ret6=ret6, ret24=ret24, ret96=ret96, ret288=ret288,
                pos96=pos96, ratio=ratio, volr=volr)

paths = {}
entries = []
t1, r1 = [], []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    t1.append(P['t'][1:]); r1.append(P['ret1'][1:])
    for i in range(300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()

a = np.concatenate(t1); b = np.concatenate(r1)
ut, inv = np.unique(a, return_inverse=True)
cnt = np.bincount(inv[b < CRASH_RET1], minlength=len(ut))
crash1 = dict(zip(ut.tolist(), cnt.tolist()))
print('好入场 %d  %.1fs' % (len(entries), time.time()-t0))

qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('验证从 %s 起' % qs[VALID_FROM_Q])

def run(name):
    Q = np.zeros((N_S, 2))
    rng = np.random.default_rng(0)
    res = []; n_upd = 0; kept = 0
    total = len(entries)
    for ei, (tms, sym, i) in enumerate(entries):
        P = paths[sym]
        c = P['c']; r6 = P['ret6']; r96 = P['ret96']
        pos = P['pos96']; ratio = P['ratio']; vr = P['volr']
        q = qm_all[ei]
        d1 = P['ret288'][i] > 0
        d2 = pos[i] > 0.3
        prev_t = tms - 300000
        a_crash = crash1.get(prev_t, 0) >= CRASH_N
        if name == 'D1' and not d1: continue
        if name == 'D2' and not d2: continue
        if name == 'D3' and not (d1 and d2): continue
        if name == 'D4' and not (d1 and d2 and not a_crash): continue
        kept += 1
        prog = ei / total
        eps = EPS_START * (1-prog) + EPS_END * prog
        alpha = ALPHA_START * (1-prog) + ALPHA_END * prog
        entry = c[i]; j = i
        while j < i + H:
            g_ = c[j] / entry - 1.0
            if not np.isfinite(ratio[j]) or not np.isfinite(pos[j]):
                j += 1; continue
            s = sid(d_gain(g_), d_hold(j-i), d_vol(ratio[j]), d_moms(r6[j]),
                    d_moml(r96[j]), d_pos(pos[j]), d_vr(vr[j]))
            act = int(rng.integers(2)) if rng.random() < eps else (1 if Q[s,1] > Q[s,0] else 0)
            if act == 1:
                r = g_ - COST
                Q[s, 1] += alpha * (r - Q[s, 1]); n_upd += 1
                if q >= qs[VALID_FROM_Q]: res.append((q, r))
                break
            j2 = j + 1
            if j2 >= i + H:
                rz = c[j2-1]/entry - 1.0 - COST
                Q[s, 0] += alpha * (rz - Q[s, 0]); n_upd += 1
                if q >= qs[VALID_FROM_Q]: res.append((q, rz))
                break
            g2 = c[j2]/entry - 1.0
            s2 = sid(d_gain(g2), d_hold(j2-i), d_vol(ratio[j2]), d_moms(r6[j2]),
                     d_moml(r96[j2]), d_pos(pos[j2]), d_vr(vr[j2]))
            Q[s, 0] += alpha * (GAMMA*max(Q[s2]) - Q[s, 0]); n_upd += 1
            j = j2
    return Q, n_upd, kept, res

lines = ['出场 V12：趋势过滤', '']
lines.append('%-28s %8s %8s %10s %10s %8s %6s' % ('变体','保留','验证笔','均值/笔','中位/笔','胜率','季正'))
summary = []
models = {}
for name, label in [('base','基线（无过滤）'), ('D1','D1 24h趋势向上(ret288>0)'),
                    ('D2','D2 不在8h低位(pos96>0.3)'), ('D3','D3 = D1且D2'),
                    ('D4','D4 = D3 + 熔断A')]:
    Q, n_upd, kept, res = run(name)
    models[label] = Q
    if not res:
        lines.append('%-28s 无样本' % label); continue
    rq = pd.DataFrame(res, columns=['q','net'])
    g = rq.groupby('q').net.agg(['mean','size'])
    posn = int((g['mean'] > 0).sum())
    summary.append((label, kept, len(rq), rq.net.mean(), rq.net.median(),
                    100*(rq.net>0).mean(), posn, len(g), rq))
    lines.append('%-28s %8d %8d %9.3f%% %9.3f%% %7.1f%% %3d/%d' %
                 (label, kept, len(rq), rq.net.mean()*100, rq.net.median()*100,
                  100*(rq.net>0).mean(), posn, len(g)))
    for qq, row in g.iterrows():
        lines.append('    %-8s %9.3f%% %6d' % (qq, row['mean']*100, int(row['size'])))

lines.append('')
if summary:
    best = max(summary, key=lambda x: x[3])
    lines.append('=== 最优: %s  均值 %.4f%%/笔' % (best[0], best[3]*100))
    for qq, row in best[8].groupby('q').net.agg(['mean','median','size']).iterrows():
        lines.append('  %-8s %9.3f%% %9.3f%% %6d' % (qq, row['mean']*100, row['median']*100, int(row['size'])))

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'exit_v12_q.pkl'), 'wb') as f:
    pickle.dump(dict(models=models, H=H, GAP=GAP, COST=COST, GAMMA=GAMMA, N_S=N_S,
                     valid_from=str(qs[VALID_FROM_Q]),
                     rules=dict(dip=DIP, volr=VOLR, crash_ret1=CRASH_RET1, crash_n=CRASH_N),
                     note='V12 趋势过滤：D1/D2/D3/D4'), f)

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
