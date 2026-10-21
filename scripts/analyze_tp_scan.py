"""
归因检验：M5「仅盈利时触发」的 +1.045%/9季正，到底是谁的功劳？

M5 = "z>0.3 且当前收益>0 就卖，否则用 θ=1% 兜底"。
但 SHUF 也是 +1.070%/9季正 —— 说明模型没贡献。嫌疑对象是"只卖盈利单"这条规则本身。

本脚本不用任何模型，只测纯规则：
  TP(x)：路径内第一次 close/entry-1 >= x 就卖；从未达到则用 θ=1% 的触发根。
  x = 0 / 0.2% / 0.5% / 1% / 2% / 3%
若 TP(0) 就已经接近 +1%，则 M5 的收益 100% 来自"止盈规则"，警觉零贡献。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/tp_scan_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    if n < 600 + H: return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


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
qm = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm.unique().tolist()))
print('入场 %d  %.1fs' % (len(entries), time.time() - t0), flush=True)

val = [(ei, e) for ei, e in enumerate(entries) if str(qm[ei]) >= str(qs[VALID_FROM_Q])]
print('验证集 %d' % len(val), flush=True)


def theta_k(P, i, th=0.01):
    h, l, c = P['h'], P['l'], P['c']
    eh = h[i:i + H] / c[i] - 1.0
    el = l[i:i + H] / c[i] - 1.0
    hi = 0.0; s = H - 1
    for j in range(H):
        if eh[j] > hi: hi = eh[j]
        if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
            s = j; break
    return s


L = []
L.append('归因检验：M5(+1.045%%) 是"警觉"的功劳还是"止盈规则"的功劳？  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('纯规则，零模型。TP(x) = 首次收益>=x 就卖；从未达到则用 θ=1%% 的触发根')
L.append('验证集 %d 笔 / %d 季 / 成本 0.04%% 单边' % (len(val), len(set(str(qm[ei]) for ei, _ in val))))
L.append('')
L.append('  %-22s %10s %10s %8s %10s' % ('规则', '均值/笔', '中位/笔', '季正', '触发率'))
rows = []
for x in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
    nets = []; trig = 0; qlv = []
    for ei, (tms, sym, i) in val:
        P = paths[sym]; c = P['c']
        r = c[i:i + H] / c[i] - 1.0
        kth = theta_k(P, i)
        k = kth
        for j in range(H):
            if r[j] >= x:
                k = j; trig += 1; break
        nets.append(float(r[k]) - COST)
        qlv.append(str(qm[ei]))
    nets = np.array(nets)
    g = pd.DataFrame({'q': qlv, 'net': nets}).groupby('q').net.mean()
    rows.append((x, nets.mean()))
    L.append('  %-22s %9.3f%% %9.3f%% %6d/%d %9.1f%%'
             % ('TP(=%.1f%%)' % (x * 100), nets.mean() * 100, np.median(nets) * 100,
                int((g > 0).sum()), len(g), trig / len(val) * 100))
# 基线：只 θ
nets = []
for ei, (tms, sym, i) in val:
    P = paths[sym]; c = P['c']
    r = c[i:i + H] / c[i] - 1.0
    nets.append(float(r[theta_k(P, i)]) - COST)
nets = np.array(nets)
qlv = [str(qm[ei]) for ei, _ in val]
g = pd.DataFrame({'q': qlv, 'net': nets}).groupby('q').net.mean()
L.append('  %-22s %9.3f%% %9.3f%% %6d/%d %9.1f%%'
         % ('θ=1% 兜底（基线）', nets.mean() * 100, np.median(nets) * 100,
            int((g > 0).sum()), len(g), 99.7))
L.append('')
L.append('判定：若 TP(=0%) 已达 +1% 量级，则 M5 的收益来自"只卖盈利单"这条规则，警觉零贡献。')
L.append('诚实边界：样本重叠、零滑点；成本按一次往返 0.04% 计；TP 的卖点用 close，实盘需挂单成交假设。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
