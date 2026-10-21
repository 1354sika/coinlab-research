"""
峰前认峰 v4 —— 放宽标签，把"能不能分辨顶"变成有样本量的检验。

v3 的标签太严（顶=之后30min最大涨幅<0.5% 且 跌>=2%），只剩 6 个正样本。
v4 改口径：
  候选 = 正在涨的时刻（过去 60 秒涨幅 > 0.5%），每币去重（>=60s）
  y=1  = 之后 30 分钟内跌 >= 2%（涨完就砸）
  y=0  = 之后 30 分钟内跌幅 < 1%（涨完没砸）
  中间态（跌 1%~2%）剔除，避免标签噪声
特征 = 决策前 60 秒（严格因果）
输出 AUC + bootstrap 95% CI（1000 次）
"""
import pandas as pd, numpy as np, os, time

BASE = r'D:/中信证券/coinlab/data/live/2026-10-18'
OUT = r'D:/中信证券/coinlab/evidence/live_peak_probe4.txt'
T0 = pd.Timestamp('2026-10-18 08:00:00').value // 10 ** 6
FUT, RISE = 1800, 0.005
Y1_DROP, Y0_DROP = -0.02, -0.01


def smart_read(p):
    with open(p, encoding='utf-8') as f:
        hdr = f.readline().strip().lstrip('\ufeff').split(',')
        row = f.readline().strip().split(',')
    hdr = [str(h).strip() for h in hdr]
    if len(row) == len(hdr) + 1:
        hdr = hdr + ['extra_%d' % len(hdr)]
    return pd.read_csv(p, names=hdr, header=0, index_col=False)


def auc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    ra = r[:len(a)].sum()
    return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


t0 = time.time()
k = smart_read(os.path.join(BASE, 'klines_1s.csv'))
k = k[k.open_time >= T0]
k['sec'] = k.open_time // 1000
tr = smart_read(os.path.join(BASE, 'aggTrades.csv'))
tr = tr[tr.time >= T0]
tr['sec'] = tr.time // 1000
dp = smart_read(os.path.join(BASE, 'depth20.csv'))
dp = dp[dp.ts_ms >= T0]
print('读入 %.1fs' % (time.time() - t0))

feats = ['r60', 'accel', 'volr', 'ntr', 'buy', 'big', 'imb', 'spr']
pos = {f: [] for f in feats}
neg = {f: [] for f in feats}

for sym, g in k.groupby('symbol', sort=False):
    g = g.sort_values('sec')
    sec = g.sec.to_numpy(); c = g.close.to_numpy(np.float64); v = g.volume.to_numpy(np.float64)
    n = len(c)
    if n < 900:
        continue
    j60 = np.clip(np.searchsorted(sec, sec - 60), 0, n - 1)
    r60 = c / c[j60] - 1.0
    tg = tr[tr.symbol == sym].sort_values('time')
    tsec = tg.sec.to_numpy(); tq = tg.qty.to_numpy(np.float64)
    tbm = tg.is_buyer_maker.to_numpy()
    big_thr = np.percentile(tq, 99) if len(tq) > 100 else np.inf
    dg = dp[dp.symbol == sym]
    if len(dg):
        piv = dg.pivot_table(index='ts_ms', columns='side', values='qty', aggfunc='sum')
        piv = piv.reindex(columns=['bid', 'ask']).fillna(0.0)
        dsec = piv.index.to_numpy()
        dimb = (piv.bid / (piv.bid + piv.ask).replace(0, np.nan)).to_numpy()
        best = dg[dg.level == 0].pivot_table(index='ts_ms', columns='side', values='price',
                                             aggfunc='first').reindex(columns=['bid', 'ask'])
        dspread = ((best.ask - best.bid) / ((best.ask + best.bid) / 2)).to_numpy()
    else:
        dsec = np.array([]); dimb = np.array([]); dspread = np.array([])

    idx = np.arange(n)
    cand = idx[(r60 > RISE) & (idx >= 300) & (idx < n - FUT - 1)]
    keep = []
    for i in cand:
        if not keep or i - keep[-1] >= 60:
            keep.append(int(i))
    for i in keep:
        fut = c[i:i + FUT]
        drop = fut.min() / c[i] - 1.0
        if drop <= Y1_DROP:
            y = 1
        elif drop > Y0_DROP:
            y = 0
        else:
            continue
        lo, hi = sec[i] - 60, sec[i]
        a = np.searchsorted(sec, lo); b = np.searchsorted(sec, hi)
        if b - a < 10:
            continue
        r = c[b - 1] / c[a] - 1.0
        mid = (a + b) // 2
        accel = (c[b - 1] / c[mid] - 1.0) - (c[mid] / c[a] - 1.0)
        v_win = v[a:b].sum()
        v_base = v[max(0, a - 240):a].mean() * 60 if a > 240 else np.nan
        ta = np.searchsorted(tsec, lo); tb = np.searchsorted(tsec, hi)
        ntr = int(tb - ta)
        buy = float((~tbm[ta:tb]).mean()) if ntr > 0 else np.nan
        big = float((tq[ta:tb] > big_thr).mean()) if ntr > 0 else np.nan
        im = np.nan; sp = np.nan
        if len(dsec):
            j = np.searchsorted(dsec, hi * 1000)
            if j > 0:
                im = dimb[j - 1]; sp = dspread[j - 1]
        vals = dict(r60=r, accel=accel,
                    volr=(v_win / v_base) if np.isfinite(v_base) and v_base > 0 else np.nan,
                    ntr=ntr, buy=buy, big=big, imb=im, spr=sp)
        bucket = pos if y == 1 else neg
        for kk, vv in vals.items():
            if np.isfinite(vv):
                bucket[kk].append(vv)

npos = len(pos['r60']); nneg = len(neg['r60'])
print('正样本 %d / 负样本 %d  %.1fs' % (npos, nneg, time.time() - t0))

rng = np.random.default_rng(7)
L = []
L.append('峰前认峰 v4（放宽标签，带 bootstrap 置信区间）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据: data/live/2026-10-18  30 币  1s + aggTrades + depth20')
L.append('候选: 过去60秒涨>0.5% 的时刻（每币去重>=60s），决策前 60 秒做特征')
L.append('标签: 正=之后30min跌>=2%; 负=之后30min跌幅<1%; 中间态剔除')
L.append('样本: 正 %d / 负 %d' % (npos, nneg))
L.append('')
L.append('  %-7s %12s %12s %8s %18s' % ('特征', '正均值', '负均值', 'AUC', 'bootstrap 95%CI'))
out = []
for f in feats:
    a = np.array(pos[f]); b = np.array(neg[f])
    if len(a) < 5 or len(b) < 5:
        L.append('  %-7s  (样本不足 %d/%d)' % (f, len(a), len(b)))
        continue
    v = auc(a, b)
    bs = []
    for _ in range(1000):
        bs.append(auc(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append((abs(v - 0.5), f, v, lo, hi))
    L.append('  %-7s %12.5f %12.5f %8.3f   [%.3f, %.3f]' % (f, a.mean(), b.mean(), v, lo, hi))
L.append('')
if out:
    best = sorted(out)[-1]
    L.append('最强特征 %s AUC=%.3f CI[%.3f,%.3f]  %s'
             % (best[1], best[2], best[3], best[4],
                'CI 不含 0.5 => 有区分力' if (best[3] > 0.5 or best[4] < 0.5) else 'CI 含 0.5 => 无定论'))
L.append('')
L.append('诚实边界：')
L.append('  1. 单日、30 币、约 8 小时。正样本 %d 个，bootstrap CI 已经很宽。' % npos)
L.append('  2. 1s K 线只覆盖约 65%% 的秒；盘口 10 秒一张，做不了秒级盘口。')
L.append('  3. 只回答"有没有区分力"，不回答"能不能赚钱"（未计成本/滑点/成交率）。')
L.append('  4. 标签用未来 30 分钟定义（事后），特征严格只用过去 —— 无前视泄漏。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
