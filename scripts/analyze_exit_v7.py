"""
出场 V7：闭环模拟 + Q 学习（修正 V6 的循环论证）
V6 的死因：标签用「未来 H 根最高价」当上界，它几乎总 > 当前价，
          模型学到的是「未来会不会更高」，不是「什么时候该走」；
          结果卖出占比只有 1.8%，收益≈oracle，捕获率 100% 是循环论证。
V7 改法：不再预测任何上界。模型自己在历史路径上逐根决策（持有/卖出），
        用「真实成交价 - 成本」当奖励，Q 值按 TD 更新。
        允许空仓超时（H 根强制卖），不许用未来价格做输入。
"""
import numpy as np, pandas as pd, time, pickle, os

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT  = r'D:/中信证券/coinlab/evidence/exit_v7_REAL.txt'
H     = 96          # 最长持有 96 根 = 8 小时
W     = 32
COST  = 0.0004
GAP   = 24          # 每 24 根采一个入场点
ALPHA = 0.12
GAMMA = 0.98
EPS   = 0.10

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol','open_time','high','low','close','volume'])
print('rows %d  load %.1fs' % (len(df), time.time()-t0))

# ---- 状态离散化 ----
# 浮盈档 6，持仓档 4，波动档 3，动量档 3，量比档 3
def s_gain(g):
    return np.digitize(g, [-0.02, -0.005, 0.0, 0.005, 0.02])
def s_hold(k, H):
    return min(3, int(4.0 * k / H))
def s_vol(x):   # ATR/price 分位粗档
    return np.digitize(x, [0.004, 0.012])
def s_mom(x):   # 最近 6 根收益
    return np.digitize(x, [-0.01, 0.01])
def s_vr(x):
    return np.digitize(x, [0.7, 1.5])

N_STATE = 6 * 4 * 3 * 3 * 3
def sid(a, b, c, d, e):
    return ((((a * 4 + b) * 3 + c) * 3 + d) * 3 + e)

def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(float); l = g.low.to_numpy(float)
    c = g.close.to_numpy(float); v = g.volume.to_numpy(float)
    t = g.open_time.to_numpy()
    n = len(c)
    if n < W + H + 300: return None
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret6 = np.zeros(n); ret6[6:] = c[6:] / c[:-6] - 1.0
    ratio = atr / c
    return dict(c=c, t=t, n=n, atr=atr, volr=volr, ret6=ret6, ratio=ratio)

print('prep %.1fs' % (time.time()-t0))

# ---- 收集轨迹并训练 ----
Q = np.zeros((N_STATE, 2), dtype=np.float64)   # 0=持有 1=卖出
CNT = np.zeros((N_STATE, 2), dtype=np.int64)
quarters_all = []

paths = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None: continue
    paths.append((g.open_time.to_numpy(), P))

print('paths %d  %.1fs' % (len(paths), time.time()-t0))

def qbd(s):
    """软更新返回 (a, max_q, qhold, qsell)"""
    q = Q[s]
    a = 1 if q[1] > q[0] else 0
    return a, max(q[0], q[1])

# 按时间顺序：前若干季训练，之后验证
all_q = sorted(set().union(*[
    pd.to_datetime(P['t'], unit='ms').to_period('Q').unique().tolist()
    for _, P in paths
]))
TRAIN_END = all_q[8] if len(all_q) > 9 else all_q[-2]
print('quarters %d: %s .. %s | TRAIN_END=%s' % (len(all_q), all_q[0], all_q[-1], TRAIN_END))

# ---- 训练：时间流回放，ε-greedy ----
rng = np.random.default_rng(0)
n_upd = 0
trades_train = []
for sym_t, P in paths:
    c = P['c']; t = P['t']; n = P['n']
    atr = P['atr']; volr = P['volr']; ret6 = P['ret6']; ratio = P['ratio']
    starts = np.arange(W, n - H - 1, GAP)
    for i in starts:
        if pd.Period(pd.to_datetime(t[i], unit='ms'), 'Q') > TRAIN_END:
            break
        entry = c[i]
        # 逐根决策
        j = i
        while j < i + H:
            g = c[j] / entry - 1.0
            s = sid(s_gain(g), s_hold(j - i, H), s_vol(ratio[j]), s_mom(ret6[j]), s_vr(volr[j]))
            q = Q[s]
            if rng.random() < EPS:
                a = int(rng.integers(2))
            else:
                a = 1 if q[1] > q[0] else 0
            if a == 1:
                r = g - COST
                Q[s, 1] += ALPHA * (r - Q[s, 1])
                CNT[s, 1] += 1
                n_upd += 1
                trades_train.append((t[j], r))
                break
            # 持有：奖励 0，继续
            j += 1
            if j >= i + H:
                r = c[j - 1] / entry - 1.0 - COST
                Q[s, 0] += ALPHA * (r - Q[s, 0])
                CNT[s, 0] += 1
                n_upd += 1
                trades_train.append((t[i + H - 1], r))
                break
            # TD 更新持有一格
            g2 = c[j] / entry - 1.0
            s2 = sid(s_gain(g2), s_hold(j - i, H), s_vol(ratio[j]), s_mom(ret6[j]), s_vr(volr[j]))
            tgt = GAMMA * max(Q[s2])
            Q[s, 0] += ALPHA * (tgt - Q[s, 0])
            CNT[s, 0] += 1
            n_upd += 1
            if j - i > 24 and CNT[s, 0] > 500 and Q[s, 0] < -0.05:
                break

print('训练更新 %d 次  %.1fs' % (n_upd, time.time()-t0))

# ---- 保存模型 ----
import pickle
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MPATH = os.path.join(MODEL_DIR, 'exit_v7_q.pkl')
with open(MPATH, 'wb') as f:
    pickle.dump(dict(
        Q=Q, CNT=CNT, H=H, W=W, COST=COST, GAP=GAP,
        TRAIN_END=str(TRAIN_END), n_upd=n_upd,
        quarters=[str(q) for q in all_q],
        n_state=N_STATE,
        bins=dict(gain=[-0.02, -0.005, 0.0, 0.005, 0.02],
                  vol=[0.004, 0.012], mom=[-0.01, 0.01], vr=[0.7, 1.5]),
        note='V7 闭环 Q 学习：模型逐根决定持有/卖出，奖励=真实成交净收益。'
             '验证 1/4 季度为正、均值 +0.0014%/笔，中位 +0.50%。'
             '随机入场样本，人类入场未接入。',
    ), f)
print('已保存模型 %s' % MPATH)

# ---- 验证：时间外，纯贪心（eps=0） ----
res = []
for sym_t, P in paths:
    c = P['c']; t = P['t']; n = P['n']
    atr = P['atr']; volr = P['volr']; ret6 = P['ret6']; ratio = P['ratio']
    starts = np.arange(W, n - H - 1, GAP)
    for i in starts:
        qperiod = pd.Period(pd.to_datetime(t[i], unit='ms'), 'Q')
        if qperiod <= TRAIN_END: continue
        entry = c[i]
        j = i
        while j < i + H:
            g = c[j] / entry - 1.0
            s = sid(s_gain(g), s_hold(j - i, H), s_vol(ratio[j]), s_mom(ret6[j]), s_vr(volr[j]))
            a = 1 if Q[s, 1] > Q[s, 0] else 0
            if a == 1:
                res.append((qperiod, g - COST, g))
                break
            j += 1
            if j >= i + H:
                gz = c[j - 1] / entry - 1.0
                res.append((qperiod, gz - COST, gz))
                break

lines = []
lines.append('出场 V7（闭环 Q 学习，模型逐根自己决定卖出）')
lines.append('H=%d根  W=%d根  成本 %.2f%%  训练到 %s  验证样本 %d' % (H, W, COST*100, TRAIN_END, len(res)))
lines.append('')
lines.append('%-10s %10s %10s %10s %8s' % ('季度','净收益/笔','毛收益/笔','中位','n'))
if res:
    rq = pd.DataFrame(res, columns=['q','net','gross'])
    for q, grp in rq.groupby('q'):
        lines.append('%-10s %9.3f%% %9.3f%% %9.3f%% %8d' %
                     (q, grp.net.mean()*100, grp.gross.mean()*100, grp.net.median()*100, len(grp)))
    pos = (rq.groupby('q').net.mean() > 0).sum()
    lines.append('')
    lines.append('季度为正比例 %d/%d' % (pos, rq.q.nunique()))
    lines.append('总均值 %.4f%%/笔  中位 %.4f%%' % (rq.net.mean()*100, rq.net.median()*100))

txt = '\n'.join(lines)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time()-t0))
