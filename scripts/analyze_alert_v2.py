"""
警觉 v2：不"一有就触发"——把触发机制改掉（用户要求）。

上一轮事实：z>0.2~1.0 一触发就全平 → 均值 -0.022%~-0.059%（中位变好、均值仍负）。
原因：触发率 100%，等于每笔都在波动放大时离场，砍掉了尾部大赢家。

本脚本测 7 种"不立即全平"的机制：
  M1 半仓：警觉点平 50%，剩 50% 交给 θ=1%
  M2 分三级：z>0.3 平 1/3、z>0.6 再 1/3、z>1.0 再 1/3，余下交给 θ
  M3 去抖：z 连续 2 根 > 阈值才卖
  M4 去抖：z 连续 3 根 > 阈值才卖
  M5 只在盈利时触发（当前收益 > 0），亏损交给 θ
  M6 确认：z>阈值后，等下一根不创新高才卖
  M7 半仓 + 去抖 2 根
全部在线，SHUF 对照；与 V20（θ=1%）同一批入场点、同一成本。
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/alert_v2_REAL.txt'
H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4
CTX, TOPK, CONF_DROP = 96, 6, 0.01

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
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

Xl, yl, ei_l = [], [], []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    h, l, c, v = P['h'], P['l'], P['c'], P['v']
    prevc = np.concatenate([[c[0]], c[:-1]])
    lr1 = np.log(c / prevc)
    for j in range(i, i + H):
        f = [float(c[j] / c[j - k] - 1.0) for k in (12, 24, 48, 96)]
        v12 = float(v[j - 11:j + 1].mean()); v96 = float(v[j - 95:j + 1].mean())
        f.append(float(v12 / v96) if v96 > 0 else 1.0)
        f.append(float(np.std(lr1[j - 23:j + 1])))
        hi96 = float(h[j - 95:j + 1].max()); lo96 = float(l[j - 95:j + 1].min())
        f.append(float((c[j] - lo96) / (hi96 - lo96)) if hi96 > lo96 else 0.5)
        rng = float(h[j] - l[j])
        f.append(float((h[j] - c[j]) / rng) if rng > 0 else 0.0)
        f.append(float(c[j] / h[j - 12] - 1.0))
        is_top = h[j] >= h[max(0, j - TOPK):min(P['n'] - 1, j + TOPK) + 1].max() - 1e-12
        fut = l[j + 1:j + 13]
        ok = len(fut) > 0 and fut.min() <= h[j] * (1.0 - CONF_DROP)
        Xl.append(f); yl.append(1 if (is_top and ok) else 0); ei_l.append(ei)

X = np.asarray(Xl, np.float32); y = np.asarray(yl, np.int8); ei_l = np.asarray(ei_l)
n_e = len(entries)
q_of = {e: str(qm_all[e]) for e in range(n_e)}
qr = np.array([q_of[e] for e in ei_l])
tr = qr < str(qs[VALID_FROM_Q]); va = ~tr
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
p_r = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr]).predict_proba(Xs[va])[:, 1]
rng = np.random.default_rng(11)
p_s = LogisticRegression(max_iter=1000).fit(Xs[tr], rng.permutation(y[tr])).predict_proba(Xs[va])[:, 1]
print('训练完成 %.1fs' % (time.time() - t0), flush=True)

va_e = np.unique(ei_l[va])
pos = {e: np.where(ei_l[va] == e)[0] for e in va_e}
retp = {}
for e in va_e:
    P = paths[entries[e][1]]; i = entries[e][2]
    retp[e] = P['c'][i:i + H] / P['c'][i] - 1.0
qv = np.array([q_of[e] for e in va_e])


def zsc(ps, win=24):
    z = np.zeros(len(ps))
    for j in range(win, len(ps)):
        m = ps[j - win:j].mean(); s = ps[j - win:j].std() + 1e-6
        z[j] = (ps[j] - m) / s
    return z


def k_theta(th=0.01):
    ks = []
    for e in va_e:
        P = paths[entries[e][1]]; i = entries[e][2]
        h, l, c = P['h'], P['l'], P['c']
        eh = h[i:i + H] / c[i] - 1.0
        el = l[i:i + H] / c[i] - 1.0
        hi = 0.0; s = H - 1
        for j in range(H):
            if eh[j] > hi:
                hi = eh[j]
            if j >= 1 and (1.0 + el[j]) / (1.0 + hi) - 1.0 <= -th:
                s = j; break
        ks.append(s)
    return np.array(ks)


KTH = k_theta(0.01)


def report(name, nets):
    g = pd.DataFrame({'q': qv, 'net': nets}).groupby('q').net.mean()
    return '  %-34s %9.3f%% %9.3f%% %6d/%d' % (name, nets.mean() * 100, np.median(nets) * 100,
                                               int((g > 0).sum()), len(g))


def m_half(P, th=0.3, frac=0.5):
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zsc(ps); r = retp[e]; kth = int(KTH[n])
        ka = None
        for j in range(24, len(ps)):
            if z[j] > th:
                ka = j; break
        if ka is None:
            out.append(float(r[kth]) - COST)
        else:
            net = frac * float(r[ka]) + (1 - frac) * float(r[kth]) - COST
            out.append(net)
    return np.array(out)


def m_tier(P, th1=0.3, th2=0.6, th3=1.0):
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zsc(ps); r = retp[e]; kth = int(KTH[n])
        sold = 0.0; w = 0.0
        for j in range(24, len(ps)):
            if sold < 1e-9 and z[j] > th1:
                w += float(r[j]) / 3; sold += 1 / 3
            elif sold < 1 / 3 + 1e-9 and z[j] > th2:
                w += float(r[j]) / 3; sold += 1 / 3
            elif sold < 2 / 3 + 1e-9 and z[j] > th3:
                w += float(r[j]) / 3; sold += 1 / 3
        w += (1 - sold) * float(r[kth])
        out.append(w - COST)
    return np.array(out)


def m_debounce(P, th=0.3, k=2):
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zsc(ps); r = retp[e]; kth = int(KTH[n])
        cnt = 0; kk = kth
        for j in range(24, len(ps)):
            cnt = cnt + 1 if z[j] > th else 0
            if cnt >= k:
                kk = j; break
        out.append(float(r[kk]) - COST)
    return np.array(out)


def m_inprofit(P, th=0.3):
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zsc(ps); r = retp[e]; kth = int(KTH[n])
        kk = kth
        for j in range(24, len(ps)):
            if z[j] > th and r[j] > 0:
                kk = j; break
        out.append(float(r[kk]) - COST)
    return np.array(out)


def m_confirm(P, th=0.3):
    """z>th 后，等下一根 close 低于当根才卖"""
    out = []
    for n, e in enumerate(va_e):
        ps = P[pos[e]]; z = zsc(ps); r = retp[e]; kth = int(KTH[n])
        kk = kth; armed = -1
        for j in range(24, len(ps)):
            if armed < 0 and z[j] > th:
                armed = j
            elif armed >= 0 and j > armed and r[j] < r[armed]:
                kk = j; break
        out.append(float(r[kk]) - COST)
    return np.array(out)


L = []
L.append('警觉 v2：不"一有就触发"——改触发机制  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('验证集 %d 笔 / %d 季 | 成本 0.04%% 单边 | z 滚动窗 24 根' % (len(va_e), len(set(qv))))
L.append('')
L.append('  %-34s %10s %10s %8s' % ('策略', '均值/笔', '中位/笔', '季正'))
L.append(report('固定持满 96 根（基线）', np.array([retp[e][-1] - COST for e in va_e])))
L.append(report('θ=1% 跟踪止盈（V20）', np.array([float(retp[e][k]) - COST for e, k in zip(va_e, KTH)])))
prev_nets = []
for n, e in enumerate(va_e):
    ps = p_r[pos[e]]; z = zsc(ps); kk = int(KTH[n])
    for j in range(24, len(ps)):
        if z[j] > 0.3:
            kk = j
            break
    prev_nets.append(float(retp[e][kk]) - COST)
L.append(report('REAL 一触发就全平（上一轮）', np.array(prev_nets)))
L.append(report('M1 REAL 半仓（50% + θ）', m_half(p_r, 0.3, 0.5)))
L.append(report('M1 SHUF 半仓', m_half(p_s, 0.3, 0.5)))
L.append(report('M1 REAL 半仓 75%', m_half(p_r, 0.3, 0.25)))
L.append(report('M2 REAL 分三级 1/3', m_tier(p_r)))
L.append(report('M2 SHUF 分三级 1/3', m_tier(p_s)))
L.append(report('M3 REAL 去抖 2 根', m_debounce(p_r, 0.3, 2)))
L.append(report('M4 REAL 去抖 3 根', m_debounce(p_r, 0.3, 3)))
L.append(report('M5 REAL 仅盈利时触发', m_inprofit(p_r, 0.3)))
L.append(report('M5 SHUF 仅盈利时触发', m_inprofit(p_s, 0.3)))
L.append(report('M6 REAL 等确认一根', m_confirm(p_r, 0.3)))
L.append('')
L.append('读法：目标不是"卖得更早"，而是"不砍光尾部的同时保住中位"。')
L.append('      半仓/分级若把均值拉回正、且 REAL 优于 SHUF，才算机制改对了。')
L.append('诚实边界：样本重叠、零滑点；成本按一次往返 0.04% 计（分两次卖的费率未单独计）。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
