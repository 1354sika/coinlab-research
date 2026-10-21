"""
峰前认峰 v3 —— 修正 v2 的对照设计缺陷。

v2 的错：把"峰前1分钟的特征"与"当下正在涨0.5%的时刻"对比，两组时间语义不同
（前者多半还在平静期，后者正在拉升），AUC 0.003 是设计产物，不是发现。

v3 的做法（这才是真问题）：
  候选时刻 = 正在拉升的时刻（过去 60 秒涨幅 > 0.5%），每币去重（间隔 >=60s）
  标签 y=1（这就是顶）：之后 30 分钟内最大涨幅 < 0.5% 且 期间跌幅 >= 2%
  标签 y=0（不是顶）  ：之后 30 分钟内还涨 >=1%，或没跌到 2%
  特征 = 该时刻之前 60 秒的微观结构（1s 收益/加速/量比/逐笔/主动买/大单/盘口失衡）

两组的"当下状态"完全一样（都在涨），所以 AUC 检验的是：
    涨的时候都像顶，能不能分辨哪一个真会跌？
"""
import pandas as pd, numpy as np, os, time

BASE = r'D:/中信证券/coinlab/data/live/2026-10-18'
OUT = r'D:/中信证券/coinlab/evidence/live_peak_probe3.txt'
T0 = pd.Timestamp('2026-10-18 08:00:00').value // 10 ** 6
FUT = 1800
RISE = 0.005
TOP_MAXGAIN = 0.005
TOP_DROP = 0.02


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
    if len(a) < 10 or len(b) < 10:
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
print('读入 k=%d tr=%d dp=%d %.1fs' % (len(k), len(tr), len(dp), time.time() - t0))

feats = ['r60', 'accel', 'volr', 'ntr', 'buy', 'big', 'imb', 'spr']
pos = {f: [] for f in feats}
neg = {f: [] for f in feats}
n_y = 0; n_n = 0

for sym, g in k.groupby('symbol', sort=False):
    g = g.sort_values('sec')
    sec = g.sec.to_numpy(); c = g.close.to_numpy(np.float64); v = g.volume.to_numpy(np.float64)
    n = len(c)
    if n < 900:
        continue
    j60 = np.searchsorted(sec, sec - 60)
    j60 = np.clip(j60, 0, n - 1)
    r60 = c / c[j60] - 1.0

    tg = tr[tr.symbol == sym]
    tg = tg.sort_values('time')
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
        max_gain = fut.max() / c[i] - 1.0
        min_drop = fut.min() / c[i] - 1.0
        if max_gain < TOP_MAXGAIN and min_drop <= -TOP_DROP:
            y = 1
        elif max_gain >= 0.01 or min_drop > -TOP_DROP:
            y = 0
        else:
            continue
        lo = sec[i] - 60; hi = sec[i]
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
        if y == 1:
            n_y += 1
        else:
            n_n += 1
        for kk, vv in vals.items():
            if np.isfinite(vv):
                bucket[kk].append(vv)

print('候选(正在涨) 顶 %d / 非顶 %d  %.1fs' % (n_y, n_n, time.time() - t0))

L = []
L.append('峰前认峰 v3（所有样本都"正在涨"，检验能否分辨真顶）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据: data/live/2026-10-18  30 币  1s + aggTrades + depth20(10s快照)')
L.append('候选: 过去60秒涨幅>0.5%% 的时刻，每币去重(间隔>=60s)')
L.append('标签 顶=之后30min内最大涨幅<0.5%% 且 期间跌>=2%%；非顶=之后仍涨>=1%% 或 没跌到2%%')
L.append('特征窗口: 决策前 60 秒（严格只用过去）')
L.append('')
L.append('样本: 顶 %d / 非顶 %d' % (n_y, n_n))
L.append('')
L.append('  %-7s %12s %12s %8s' % ('特征', '顶均值', '非顶均值', 'AUC'))
res = []
for f in feats:
    a = np.array(pos[f]); b = np.array(neg[f])
    if len(a) < 10 or len(b) < 10:
        L.append('  %-7s  (样本不足 %d/%d)' % (f, len(a), len(b)))
        continue
    v = auc(a, b)
    res.append((abs(v - 0.5), f, v))
    L.append('  %-7s %12.5f %12.5f %8.3f' % (f, a.mean(), b.mean(), v))
L.append('')
L.append('最强特征: ' + ('%s AUC=%.3f' % (sorted(res)[-1][1], sorted(res)[-1][2]) if res else '无'))
L.append('')
L.append('诚实边界：')
L.append('  1. 单日（2026-10-18）、30 币、约 8 小时、1s 粒度；样本 %d 个"顶"。' % n_y)
L.append('  2. "顶"的标签用未来 30 分钟定义（标签本身是事后的，正确；特征全部只用过去）。')
L.append('  3. 盘口快照 10 秒一张，做不了秒级盘口；1s K 线覆盖约 65%% 的秒。')
L.append('  4. 未计成本/滑点；本实验只回答"有没有区分力"，不回答"能不能赚钱"。')
L.append('  5. AUC 在 %d 个正样本下，95%% 置信区间约 ±0.1，仍需多日数据复核。' % n_y)

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
