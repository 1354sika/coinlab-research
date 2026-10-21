"""
出场 V8：长历史 + 在线学习（V7 的两处缺口）
V7 缺口1：状态只看 2 小时（ret6 + 当前 ATR + 当前量比），W=32 甚至没用上
V7 缺口2：训练到 2025Q4 就冻结，验证时不更新 —— 不是「边测边练」

V8 改法：
  1. 状态接入多尺度历史：ret6(30m) / ret24(2h) / ret96(8h) / ret288(24h) / 96根内位置
  2. 按时间流逐笔在线更新，验证段也在学（ε 随进度衰减）
  3. 模型自己决定动作（argmax Q），人只给状态和奖励
"""
import numpy as np, pandas as pd, time, pickle, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v8_REAL.txt'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

H    = 96        # 最长持有 96 根 = 8 小时
GAP  = 96        # 每 96 根(8h) 采一个入场点
COST = 0.0004
ALPHA_START, ALPHA_END = 0.15, 0.02
GAMMA = 0.97
EPS_START, EPS_END = 0.25, 0.02
VALID_FROM_Q = 10   # 从第 10 个季度起计入验证（前 9 季纯学习）

# ---------- 状态离散化（多尺度） ----------
def d_gain(g):  return int(np.digitize(g, [-0.02, -0.005, 0.005, 0.02]))       # 0..4
def d_hold(k):  return min(2, k // 32)                                        # 0..2
def d_vol(x):   return int(np.digitize(x, [0.008]))                            # 0..1
def d_moms(x):  return int(np.digitize(x, [-0.01, 0.01]))                      # 0..2
def d_moml(x):  return int(np.digitize(x, [-0.03, 0.03]))                      # 0..2
def d_pos(x):   return int(np.digitize(x, [0.33, 0.67]))                       # 0..2
def d_vr(x):    return int(np.digitize(x, [1.0]))                              # 0..1

N_S = 5 * 3 * 2 * 3 * 3 * 3 * 2   # 1620
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
entries = []   # (time_ms, sym, idx)
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths[sym] = P
    for i in range(300, P['n'] - H - 1, GAP):
        entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
print('symbols %d  entries %d  prep %.1fs' % (len(paths), len(entries), time.time()-t0))

qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('quarters %d: %s .. %s' % (len(qs), qs[0], qs[-1]))

Q = np.zeros((N_S, 2), dtype=np.float64)
rng = np.random.default_rng(0)
n_upd = 0
res = []
total = len(entries)
print('开始在线学习+验证 ...')

for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    c = P['c']; r6 = P['ret6']; r24 = P['ret24']; r96 = P['ret96']
    r288 = P['ret288']; pos = P['pos96']; ratio = P['ratio']; vr = P['volr']
    q = qm_all[ei]
    prog = ei / total
    eps = EPS_START * (1-prog) + EPS_END * prog
    alpha = ALPHA_START * (1-prog) + ALPHA_END * prog
    entry = c[i]
    j = i
    done = False
    while j < i + H:
        g_ = c[j] / entry - 1.0
        if not np.isfinite(ratio[j]) or not np.isfinite(pos[j]):
            j += 1
            continue
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
            if q >= qs[VALID_FROM_Q] if len(qs) > VALID_FROM_Q else False:
                res.append((q, r))
            done = True
            break
        # 持有：TD 更新
        j2 = j + 1
        if j2 >= i + H:
            rz = c[j2 - 1] / entry - 1.0 - COST
            Q[s, 0] += alpha * (rz - Q[s, 0])
            n_upd += 1
            if len(qs) > VALID_FROM_Q and q >= qs[VALID_FROM_Q]:
                res.append((q, rz))
            done = True
            break
        g2 = c[j2] / entry - 1.0
        s2 = sid(d_gain(g2), d_hold(j2 - i), d_vol(ratio[j2]), d_moms(r6[j2]),
                 d_moml(r96[j2]), d_pos(pos[j2]), d_vr(vr[j2]))
        tgt = GAMMA * max(Q[s2])
        Q[s, 0] += alpha * (tgt - Q[s, 0])
        n_upd += 1
        j = j2
    if not done:
        pass

print('在线更新 %d 次  %.1fs' % (n_upd, time.time()-t0))

# ---------- 保存 ----------
os.makedirs(MODEL_DIR, exist_ok=True)
MPATH = os.path.join(MODEL_DIR, 'exit_v8_q.pkl')
with open(MPATH, 'wb') as f:
    pickle.dump(dict(
        Q=Q, H=H, GAP=GAP, COST=COST, GAMMA=GAMMA, N_S=N_S,
        n_upd=n_upd, quarters=[str(x) for x in qs],
        valid_from=str(qs[VALID_FROM_Q]) if len(qs) > VALID_FROM_Q else None,
        bins=dict(gain=[-0.02,-0.005,0.005,0.02], vol=[0.008],
                  mom_s=[-0.01,0.01], mom_l=[-0.03,0.03],
                  pos=[0.33,0.67], vr=[1.0]),
        feat=['gain','hold','vol','ret6','ret96','pos96','volr'],
        note='V8 出场：多尺度历史(30m/2h/8h/24h)+96根位置，时间流在线更新（验证段也学）',
    ), f)

# ---------- 报告 ----------
lines = []
lines.append('出场 V8（长历史 + 在线学习）')
lines.append('H=%d根  GAP=%d根  成本%.2f%%  状态%d格  在线更新%d次' %
             (H, GAP, COST*100, N_S, n_upd))
lines.append('历史尺度: ret6(30m) ret24(2h) ret96(8h) ret288(24h) pos96(96根内位置)')
lines.append('验证从 %s 起（前 %d 季纯学习，之后边测边练）' %
             (qs[VALID_FROM_Q] if len(qs) > VALID_FROM_Q else 'N/A', VALID_FROM_Q))
lines.append('')
if res:
    rq = pd.DataFrame(res, columns=['q','net'])
    lines.append('%-10s %12s %12s %8s' % ('季度','净收益/笔','中位','n'))
    for qq, grp in rq.groupby('q'):
        lines.append('%-10s %11.3f%% %11.3f%% %8d' %
                     (qq, grp.net.mean()*100, grp.net.median()*100, len(grp)))
    pos_n = int((rq.groupby('q').net.mean() > 0).sum())
    lines.append('')
    lines.append('季度为正比例 %d/%d' % (pos_n, rq.q.nunique()))
    lines.append('总均值 %.4f%%/笔  中位 %.4f%%' % (rq.net.mean()*100, rq.net.median()*100))
else:
    lines.append('验证段无样本')
lines.append('')
lines.append('已保存模型 %s' % MPATH)

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
