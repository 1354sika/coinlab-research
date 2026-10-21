"""
出场 V13：固定入场，只换出场（回答「同样的过去，换一个选择，结果差多少」）
入场：V9 好入场（ret24<=-2% 且 volr>=3），全部 2910 点，验证自 2024Q4
出场对照组：
  E1 固定持有 96 根（死拿，不决策）
  E2 固定持有 24 根
  E3 固定持有 12 根
  E4 出场模型（V8 长历史+在线 Q 学习）
  E5 oracle 完美出场（H 根内最高价，上界）
指标：均值/笔、中位/笔、捕获率（组均值 / oracle 均值）
"""
import numpy as np, pandas as pd, time, pickle, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v13_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H, GAP, COST = 96, 24, 0.0004
ALPHA_START, ALPHA_END = 0.15, 0.02
GAMMA = 0.97
EPS_START, EPS_END = 0.25, 0.02
VALID_FROM_Q = 4
DIP, VOLR = -0.02, 3.0

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
    ret6 = lagret(6); ret24 = lagret(24); ret96 = lagret(96)
    mn96 = pd.Series(l).rolling(96, min_periods=96).min().to_numpy()
    mx96 = pd.Series(h).rolling(96, min_periods=96).max().to_numpy()
    rng96 = np.where(mx96 > mn96, mx96 - mn96, np.nan)
    pos96 = np.clip((c - mn96) / rng96, 0, 1)
    ratio = np.where(c > 0, atr / c, np.nan)
    return dict(h=h, c=c, t=t, n=n, ret6=ret6, ret24=ret24, ret96=ret96,
                pos96=pos96, ratio=ratio, volr=volr)

paths = {}
entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    for i in range(300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('好入场 %d  验证自 %s  %.1fs' % (len(entries), qs[VALID_FROM_Q], time.time()-t0))

# ---------- E1/E2/E3/E5 向量化 ----------
def fixed_exit(hold):
    out = []
    for ei, (tms, sym, i) in enumerate(entries):
        q = qm_all[ei]
        if q < qs[VALID_FROM_Q]: continue
        P = paths[sym]
        r = P['c'][i + hold - 1] / P['c'][i] - 1.0 - COST
        out.append((q, r))
    return pd.DataFrame(out, columns=['q','net'])

def oracle_exit():
    out = []
    for ei, (tms, sym, i) in enumerate(entries):
        q = qm_all[ei]
        if q < qs[VALID_FROM_Q]: continue
        P = paths[sym]
        mx = P['h'][i:i+H].max()
        out.append((q, mx / P['c'][i] - 1.0 - COST))
    return pd.DataFrame(out, columns=['q','net'])

t1 = time.time()
E1 = fixed_exit(H); E2 = fixed_exit(24); E3 = fixed_exit(12); E5 = oracle_exit()
print('固定持有组算完 %.1fs' % (time.time()-t1))

# ---------- E4 模型 ----------
def model_exit():
    Q = np.zeros((N_S, 2))
    rng = np.random.default_rng(0)
    res = []; n_upd = 0
    total = len(entries)
    for ei, (tms, sym, i) in enumerate(entries):
        P = paths[sym]
        c = P['c']; r6 = P['ret6']; r96 = P['ret96']
        pos = P['pos96']; ratio = P['ratio']; vr = P['volr']
        q = qm_all[ei]
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
            a = int(rng.integers(2)) if rng.random() < eps else (1 if Q[s,1] > Q[s,0] else 0)
            if a == 1:
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
    return pd.DataFrame(res, columns=['q','net']), Q, n_upd

t2 = time.time()
E4, Q4, n_upd = model_exit()
print('模型组算完 %.1fs 更新 %d' % (time.time()-t2, n_upd))

# ---------- 报告 ----------
oracle_mean = E5.net.mean()
lines = ['出场 V13：固定入场，只换出场', '']
lines.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  验证自 %s  |  笔数 %d  |  成本 %.2f%%' %
             (DIP*100, VOLR, qs[VALID_FROM_Q], len(E1), COST*100))
lines.append('')
lines.append('%-22s %10s %10s %10s %10s' % ('出场方式','均值/笔','中位/笔','捕获率','胜率'))
for name, df_ in [('E1 固定持 96 根', E1), ('E2 固定持 24 根', E2),
                  ('E3 固定持 12 根', E3), ('E4 出场模型(V8)', E4),
                  ('E5 oracle 上界', E5)]:
    cap = df_.net.mean() / oracle_mean * 100 if oracle_mean else 0
    lines.append('%-22s %9.3f%% %9.3f%% %9.1f%% %9.1f%%' %
                 (name, df_.net.mean()*100, df_.net.median()*100, cap, 100*(df_.net>0).mean()))

lines.append('')
lines.append('（捕获率 = 该组均值 / oracle 均值；oracle 卖在 H 根内最高价，实盘不可能达到）')
lines.append('')
lines.append('=== 逐季对比（均值/笔 %）')
qs_list = sorted(E1.q.astype(str).unique())
lines.append('%-8s %10s %10s %10s %10s %10s' % ('季度','E1持96','E2持24','E3持12','E4模型','E5oracle'))
for qq in qs_list:
    def m(df_):
        x = df_[df_.q.astype(str) == qq]
        return x.net.mean()*100 if len(x) else float('nan')
    lines.append('%-8s %9.3f%% %9.3f%% %9.3f%% %9.3f%% %9.3f%%' %
                 (qq, m(E1), m(E2), m(E3), m(E4), m(E5)))

lines.append('')
lines.append('=== 季度为正数')
for name, df_ in [('E1 固定持 96 根', E1), ('E2 固定持 24 根', E2),
                  ('E3 固定持 12 根', E3), ('E4 出场模型(V8)', E4), ('E5 oracle 上界', E5)]:
    g = df_.groupby(df_.q.astype(str)).net.mean()
    lines.append('%-22s %d/%d' % (name, int((g > 0).sum()), len(g)))

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'exit_v13_q.pkl'), 'wb') as f:
    pickle.dump(dict(Q=Q4, H=H, GAP=GAP, COST=COST, GAMMA=GAMMA, N_S=N_S, n_upd=n_upd,
                     valid_from=str(qs[VALID_FROM_Q]),
                     entry_rule='ret24<=%.2f & volr>=%.1f' % (DIP, VOLR),
                     note='V13 固定入场只换出场：E4 是 V8 出场模型在同一批入场上的表现'), f)

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
