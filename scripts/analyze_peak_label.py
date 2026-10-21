"""
"喂法"假设检验：同一批 5m 数据，换标签直接教模型认峰，能否把出场做到峰顶附近？

用户的假设：不是信息不够，是喂的方式错了。
本脚本把"喂法"拆成三件事分别检验：
  L1 标签：这一根是不是峰（局部顶 + 之后 12 根跌>=1%）   <- 直接教"认峰"
     （V17 喂的是"未来 12 根是否回撤"，V19 喂的是后悔值）
  L2 特征：只用过去的多尺度价格量（12/24/48/96 根收益、量比、波动、位置、影线）
  L3 目标：分类 + 阈值决策；窗口末强制平仓，与 V20 同口径计成本

评估：
  (a) 验证集 AUC（峰 vs 非峰），带 SHUF 对照（打乱标签重训）
  (b) 用 p>阈值 卖出（首次触发）的每笔净收益，与 V20 θ=1%(+0.053%)、A2(+0.253%) 对比
数据：klines_5m_3y.csv，入场 ret24<=-2% & volr>=3.0，验证自 2024Q4。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/peak_label_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
CTX = 96          # 入场前回看根数
TOPK = 6          # 局部顶半窗
CONF_DROP = 0.01  # 顶后需跌 1% 才认作"峰"

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SK = True
except Exception:
    HAS_SK = False

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('rows %d load %.1fs  sklearn=%s' % (len(df), time.time() - t0, HAS_SK))


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32)
    l = g.low.to_numpy(np.float32); c = g.close.to_numpy(np.float32)
    v = g.volume.to_numpy(np.float32); t = g.open_time.to_numpy()
    n = len(c)
    if n < 600 + H + CTX:
        return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    lr1 = np.zeros(n); lr1[1:] = np.log(c[1:] / c[:-1])
    sd24 = pd.Series(lr1).rolling(24, min_periods=24).std().to_numpy()
    return dict(h=h, l=l, c=c, v=v, t=t, n=n, volr=volr, ret24=ret24, sd24=sd24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None:
        continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('入场 %d  验证自 %s  %.1fs' % (len(entries), qs[VALID_FROM_Q], time.time() - t0))

FE = ['r12', 'r24', 'r48', 'r96', 'volr12', 'sd24', 'pos96', 'wick', 'gap12']
X_list, y_list, eidx = [], [], []
meta = []   # (quarter, entry_close, path_high)

for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c, v = P['h'], P['l'], P['c'], P['v']
    prevc = np.concatenate([[c[0]], c[:-1]])
    lr1 = np.log(c / prevc)
    for j in range(i, i + H):
        if j - CTX < 0:
            continue
        feats = []
        for k in (12, 24, 48, 96):
            feats.append(float(c[j] / c[j - k] - 1.0))
        v12 = float(v[j - 11:j + 1].mean()); v96 = float(v[j - 95:j + 1].mean())
        feats.append(float(v12 / v96) if v96 > 0 else 1.0)
        feats.append(float(np.std(lr1[j - 23:j + 1])))
        hi96 = float(h[j - 95:j + 1].max()); lo96 = float(l[j - 95:j + 1].min())
        feats.append(float((c[j] - lo96) / (hi96 - lo96)) if hi96 > lo96 else 0.5)
        rng = float(h[j] - l[j])
        feats.append(float((h[j] - c[j]) / rng) if rng > 0 else 0.0)
        feats.append(float(c[j] / h[j - 12] - 1.0))
        # 标签：局部顶（前后 TOPK 根最高）且之后 12 根内跌 >= CONF_DROP
        lo = max(0, j - TOPK); hi = min(P['n'] - 1, j + TOPK)
        is_top = h[j] >= h[lo:hi + 1].max() - 1e-12
        fut_lo = l[j + 1:j + 13]
        confirmed = len(fut_lo) > 0 and fut_lo.min() <= h[j] * (1.0 - CONF_DROP)
        X_list.append(feats)
        y_list.append(1 if (is_top and confirmed) else 0)
        eidx.append(ei)

X = np.asarray(X_list, np.float32)
y = np.asarray(y_list, np.int8)
eidx = np.asarray(eidx)
print('样本 %d  正样本率 %.2f%%  %.1fs' % (len(y), y.mean() * 100, time.time() - t0))

q_of = {ei: str(qm_all[ei]) for ei in range(len(entries))}
qr = np.array([q_of[e] for e in eidx])
tr_m = qr < str(qs[VALID_FROM_Q])
va_m = ~tr_m
print('训练 %d 验证 %d' % (tr_m.sum(), va_m.sum()))

mu = X[tr_m].mean(0); sd = X[tr_m].std(0) + 1e-9
Xs = (X - mu) / sd


def auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def fit_pred(ytr, seed=None):
    if HAS_SK:
        m = LogisticRegression(max_iter=1000, C=1.0)
        m.fit(Xs[tr_m], ytr)
        return m.predict_proba(Xs[va_m])[:, 1]
    # 简易 GD
    w = np.zeros(X.shape[1]); b = 0.0
    lr = 0.5
    for _ in range(400):
        z = Xs[tr_m] @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - ytr
        w -= lr * (Xs[tr_m].T @ g / len(g) + 1e-4 * w)
        b -= lr * g.mean()
    z = Xs[va_m] @ w + b
    return 1 / (1 + np.exp(-z))


p_real = fit_pred(y[tr_m])
rng = np.random.default_rng(11)
p_shuf = fit_pred(rng.permutation(y[tr_m]))
yv = y[va_m]
a_real = auc(p_real[yv == 1], p_real[yv == 0])
a_shuf = auc(p_shuf[yv == 1], p_shuf[yv == 0])
print('AUC REAL %.3f  SHUF %.3f  %.1fs' % (a_real, a_shuf, time.time() - t0))

# 决策：验证集每笔路径内，第一次 p>th 就卖；从未触发则持到窗口末
va_e = np.unique(eidx[va_m])
pos_in = {e: np.where(eidx[va_m] == e)[0] for e in va_e}


def sell_by_prob(th):
    nets = []; offs = []
    for e in va_e:
        idxs = pos_in[e]
        ps = p_real[idxs]
        k = int(np.argmax(ps > th)) if (ps > th).any() else len(idxs) - 1
        nets.append(float(ret_path[e][k]) - COST)
        offs.append(k)
    return np.array(nets), np.array(offs)


# 预计算每笔路径净收益（入场价 = close[i]）
ret_path = {}
for ei in va_e:
    P = paths[entries[ei][1]]; i = entries[ei][2]; c = P['c']
    ret_path[ei] = c[i:i + H] / c[i] - 1.0

L = []
L.append('"喂法"假设检验：直接教模型认峰（标签=局部顶）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据 klines_5m_3y | 入场 ret24<=-2%% & volr>=3.0 | H=96根 | 验证自 %s | 成本 %.2f%%单边'
         % (qs[VALID_FROM_Q], COST * 100))
L.append('样本 %d（训练 %d / 验证 %d），正样本率 %.2f%%'
         % (len(y), tr_m.sum(), va_m.sum(), y.mean() * 100))
L.append('')
L.append('=== 一、能不能学会"这根是峰"')
L.append('  AUC REAL = %.3f   AUC SHUF = %.3f   （差 %.3f）' % (a_real, a_shuf, a_real - a_shuf))
L.append('')
L.append('=== 二、用预测卖出 vs 现有方案')
L.append('  %-22s %10s %10s %8s' % ('策略', '均值/笔', '中位/笔', '季正'))
q_va = np.array([q_of[e] for e in va_e])


def report(name, nets):
    g = pd.DataFrame({'q': q_va, 'net': nets}).groupby('q').net.mean()
    L.append('  %-22s %9.3f%% %9.3f%% %6d/%d' % (name, nets.mean() * 100, np.median(nets) * 100,
                                                 int((g > 0).sum()), len(g)))


for th in (0.5, 0.7, 0.9, 0.95):
    nets, offs = sell_by_prob(th)
    report('认峰 p>%.2f 卖' % th, nets)
    L.append('       触发率 %.1f%%  平均出场第 %.0f 根'
             % ((np.array(offs) < H - 1).mean() * 100, np.mean(offs)))
report('固定持满 96 根（基线）', np.array([ret_path[e][-1] - COST for e in va_e]))
L.append('')
L.append('对照（历史结果，同入场同成本）：V20 跟踪止盈 θ=1%% = +0.053%%/笔 7/9 季正；')
L.append('                              A2 每季重训创新高概率 = +0.253%%/笔 7/9 季正。')
L.append('')
L.append('诚实边界：标签用未来 12 根定义（正确），特征只用 j 及之前（无前视）；')
L.append('  样本按入场点重叠、未做事件去重；未计滑点与成交率；"认峰"标签本身定义就带 1%% 跌幅要求，')
L.append('  所以 AUC 高低只说明"能不能提前认出这类点"，不等于能赚钱。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
