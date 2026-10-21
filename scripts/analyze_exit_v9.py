"""
出场 V9：V8 架构（长历史 + 在线学习）+ 有优势的入场分布
V8 用随机入场，结果 1/3 季度为正、-0.0073%/笔 —— 随机入场本身无优势，出场救不回来。
V9 只保留「急跌 + 放量」入场点（项目里已验证的方向：dip + 巨量做多），
    再用 V8 同一套闭环 Q 学习训练出场。这是把「训练方式」和「样本优势」分开检验的一步。
"""
import numpy as np, pandas as pd, time, pickle, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v9_long_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H    = 96
GAP  = 24          # 好入场点更稀疏，采样密一点
COST = 0.0004
ALPHA_START, ALPHA_END = 0.15, 0.02
GAMMA = 0.97
EPS_START, EPS_END = 0.25, 0.02
VALID_FROM_Q = 4    # 验证从 2024Q4 起：换取 9 个季度、约 1500+ 笔样本

DIP   = -0.02      # 前 24 根(2h) 跌幅阈值
VOLR  = 3.0        # 量比阈值

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
    t = g.open_time.to_numpy()
    n = len(c)
    if n < 300 + H + 300: return None
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr  = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vma  = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    def lagret(k):
        r = np.zeros(n); r[k:] = c[k:] / c[:-k] - 1.0; return r
    ret6, ret24, ret96, ret288 = lagret(6), lagret(24), lagret(96), lagret(288)
    mn96 = pd.Series(l).rolling(96, min_periods=96).min().to_numpy()
    mx96 = pd.Series(h).rolling(96, min_periods=96).max().to_numpy()
    rng96 = np.where(mx96 > mn96, mx96 - mn96, np.nan)
    pos96 = np.clip((c - mn96) / rng96, 0, 1)
    ratio = np.where(c > 0, atr / c, np.nan)
    return dict(c=c, t=t, n=n, ret6=ret6, ret24=ret24, ret96=ret96, ret288=ret288,
                pos96=pos96, ratio=ratio, volr=volr)

paths = {}
entries = []
n_all = 0
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    for i in range(300, P['n'] - H - 1, GAP):
        n_all += 1
        # 只保留「急跌 + 放量」
        if not (P['ret24'][i] <= DIP and P['volr'][i] >= VOLR):
            continue
        entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
print('全采样 %d → 好入场 %d  (%.2f%%)  prep %.1fs' %
      (n_all, len(entries), 100.0*len(entries)/max(n_all,1), time.time()-t0))

if not entries:
    print('没有符合条件的入场点，退出')
    raise SystemExit

qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('quarters %d: %s .. %s' % (len(qs), qs[0], qs[-1]))

Q = np.zeros((N_S, 2), dtype=np.float64)
rng = np.random.default_rng(0)
n_upd = 0
res = []
total = len(entries)

for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    c = P['c']; r6 = P['ret6']; r96 = P['ret96']
    pos = P['pos96']; ratio = P['ratio']; vr = P['volr']
    q = qm_all[ei]
    prog = ei / total
    eps = EPS_START * (1-prog) + EPS_END * prog
    alpha = ALPHA_START * (1-prog) + ALPHA_END * prog
    entry = c[i]
    j = i
    mg = -9.0
    while j < i + H:
        g_ = c[j] / entry - 1.0
        if g_ > mg: mg = g_
        if not np.isfinite(ratio[j]) or not np.isfinite(pos[j]):
            j += 1; continue
        s = sid(d_gain(g_), d_hold(j - i), d_vol(ratio[j]), d_moms(r6[j]),
                d_moml(r96[j]), d_pos(pos[j]), d_vr(vr[j]))
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            a = 1 if Q[s, 1] > Q[s, 0] else 0
        if a == 1:
            r = g_ - COST
            Q[s, 1] += alpha * (r - Q[s, 1])
            n_upd += 1
            if len(qs) > VALID_FROM_Q and q >= qs[VALID_FROM_Q]:
                res.append((q, r, j - i, mg, int(tms), sym))
            break
        j2 = j + 1
        if j2 >= i + H:
            rz = c[j2 - 1] / entry - 1.0 - COST
            Q[s, 0] += alpha * (rz - Q[s, 0])
            n_upd += 1
            if len(qs) > VALID_FROM_Q and q >= qs[VALID_FROM_Q]:
                res.append((q, rz, j2 - i, mg, int(tms), sym))
            break
        g2 = c[j2] / entry - 1.0
        s2 = sid(d_gain(g2), d_hold(j2 - i), d_vol(ratio[j2]), d_moms(r6[j2]),
                 d_moml(r96[j2]), d_pos(pos[j2]), d_vr(vr[j2]))
        tgt = GAMMA * max(Q[s2])
        Q[s, 0] += alpha * (tgt - Q[s, 0])
        n_upd += 1
        j = j2

print('在线更新 %d 次  %.1fs' % (n_upd, time.time()-t0))

os.makedirs(MODEL_DIR, exist_ok=True)
MPATH = os.path.join(MODEL_DIR, 'exit_v9_long_q.pkl')
with open(MPATH, 'wb') as f:
    pickle.dump(dict(Q=Q, H=H, GAP=GAP, COST=COST, GAMMA=GAMMA, N_S=N_S,
                     n_upd=n_upd, quarters=[str(x) for x in qs],
                     valid_from=str(qs[VALID_FROM_Q]) if len(qs) > VALID_FROM_Q else None,
                     entry_rule='ret24<=%.2f & volr>=%.1f' % (DIP, VOLR),
                     bins=dict(gain=[-0.02,-0.005,0.005,0.02], vol=[0.008],
                               mom_s=[-0.01,0.01], mom_l=[-0.03,0.03],
                               pos=[0.33,0.67], vr=[1.0]),
                     note='V9 出场：V8 架构(长历史+在线) + 好入场(急跌2%+放量3倍)'),
          f)

lines = []
lines.append('出场 V9（长历史 + 在线学习 + 好入场）')
lines.append('入场条件: ret24 <= %.1f%% 且 volr >= %.1f' % (DIP*100, VOLR))
lines.append('H=%d根  GAP=%d根  成本%.2f%%  状态%d格  在线更新%d次' % (H, GAP, COST*100, N_S, n_upd))
lines.append('历史尺度: ret6(30m) ret24(2h) ret96(8h) ret288(24h) pos96')
lines.append('全部入场点 %d → 好入场 %d' % (n_all, len(entries)))
lines.append('验证从 %s 起' % (qs[VALID_FROM_Q] if len(qs) > VALID_FROM_Q else 'N/A'))
lines.append('')
if res:
    rq = pd.DataFrame(res, columns=['q','net','bars','maxgain','t0','sym'])
    rq.to_csv(r'D:/中信证券/coinlab/evidence/exit_v9_long_trades.csv', index=False)
    lines.append('%-10s %12s %12s %8s' % ('季度','净收益/笔','中位','n'))
    for qq, grp in rq.groupby('q'):
        lines.append('%-10s %11.3f%% %11.3f%% %8d' %
                     (qq, grp.net.mean()*100, grp.net.median()*100, len(grp)))
    pos_n = int((rq.groupby('q').net.mean() > 0).sum())
    lines.append('')
    lines.append('季度为正比例 %d/%d' % (pos_n, rq.q.nunique()))
    lines.append('总均值 %.4f%%/笔  中位 %.4f%%' % (rq.net.mean()*100, rq.net.median()*100))
    lines.append('胜率 %.1f%%' % (100.0*(rq.net > 0).mean()))
else:
    lines.append('验证段无样本')

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
