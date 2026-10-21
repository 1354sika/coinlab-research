"""
峰前认峰 v2 —— 修正 v1 的两个问题：

v1 的错：(1) r10/r30/r60 的 AUC 0.88~0.90 是**定义泄漏**：峰的定义就是"前后60秒最高"，
            所以峰前10秒必然在涨。那不是"认峰"，那是"峰"这个定义本身。
         (2) 盘口特征对齐用了未去重的 ts 数组，imb/spr 全部取空。

v2 的做法：
  决策时刻 = 峰 **前 1 分钟**（t-60），特征窗口 = [t-120, t-60)。
  这样特征与"峰"的定义不重叠，才是真正的"提前 1 分钟能不能认出顶"。
  对照分两组：
    A 随机时刻（什么都可能）
    B **同样在涨的非峰时刻**（该 60 秒涨幅 >0.5%，但之后 30 分钟没跌 2%）
      —— 只有 B 才回答真问题："涨的时候都像顶，能不能分辨哪一个真会跌？"

数据：data/live/2026-10-18，30 币，1s + 逐笔 + 10s 盘口。
样本量警告：峰约 37 个，AUC 的 95% 置信区间约 ±0.2，只能算机制观察。
"""
import pandas as pd, numpy as np, os, time

BASE = r'D:/中信证券/coinlab/data/live/2026-10-18'
OUT = r'D:/中信证券/coinlab/evidence/live_peak_probe2.txt'
T0 = pd.Timestamp('2026-10-18 08:00:00').value // 10 ** 6
PEAK_DROP = 0.02
FUT_SEC = 1800
LOCAL_WIN = 60
LEAD = 60          # 决策提前量：峰前 60 秒
RNG = np.random.default_rng(20261018)


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
print('读入 k=%d tr=%d dp=%d %.1fs' % (len(k), len(tr), len(dp), time.time() - t0))

feats = ['r60', 'accel', 'volr', 'ntr', 'buy', 'big', 'imb', 'spr']
pos = {f: [] for f in feats}
negA = {f: [] for f in feats}
negB = {f: [] for f in feats}
peak_n = 0; up_n = 0

for sym, g in k.groupby('symbol', sort=False):
    g = g.sort_values('sec')
    sec = g.sec.to_numpy(); c = g.close.to_numpy(np.float64); v = g.volume.to_numpy(np.float64)
    n = len(c)
    if n < 900:
        continue
    s = pd.Series(c)
    loc = (s == s.rolling(2 * LOCAL_WIN + 1, center=True, min_periods=1).max()).to_numpy()
    cand = []
    for i in np.where(loc)[0]:
        if i + FUT_SEC >= n:
            continue
        if c[i + 1:i + 1 + FUT_SEC].min() / c[i] - 1.0 <= -PEAK_DROP:
            cand.append(int(i))
    peaks = []
    for i in cand:
        if peaks and sec[i] - sec[peaks[-1]] < FUT_SEC:
            if c[i] > c[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)

    tg = tr[tr.symbol == sym]
    tsec = tg.sec.to_numpy(); tq = tg.qty.to_numpy(np.float64)
    tbm = tg.is_buyer_maker.to_numpy()
    big_thr = np.percentile(tq, 99) if len(tq) > 100 else np.inf
    dg = dp[dp.symbol == sym]
    dsec = np.sort(dg.ts_ms.unique()) if len(dg) else np.array([])
    if len(dg):
        piv = dg.pivot_table(index='ts_ms', columns='side', values='qty', aggfunc='sum')
        piv = piv.reindex(columns=['bid', 'ask']).fillna(0.0)
        dimb = (piv.bid / (piv.bid + piv.ask).replace(0, np.nan)).to_numpy()
        best = dg[dg.level == 0].pivot_table(index='ts_ms', columns='side', values='price',
                                             aggfunc='first').reindex(columns=['bid', 'ask'])
        dspread = ((best.ask - best.bid) / ((best.ask + best.bid) / 2)).to_numpy()
        # 对齐：piv.index 与 dsec 必须同序同长
        dsec = piv.index.to_numpy()

    def grab(dec_i, bucket):
        """特征窗口 [dec-60s, dec)，dec_i 为决策时刻索引；只用过去。"""
        lo = sec[dec_i] - 60; hi = sec[dec_i]
        j_lo = np.searchsorted(sec, lo); j_hi = np.searchsorted(sec, hi)
        if j_hi - j_lo < 10:
            return
        r = c[j_hi - 1] / c[j_lo] - 1.0
        # 前 60 秒的加速度：后半段涨幅 - 前半段涨幅
        mid = (j_lo + j_hi) // 2
        r_half2 = c[j_hi - 1] / c[mid] - 1.0
        r_half1 = c[mid] / c[j_lo] - 1.0
        v_win = v[j_lo:j_hi].sum()
        v_base = v[max(0, j_lo - 240):j_lo].mean() * 60 if j_lo > 240 else np.nan
        tm = (tsec >= lo) & (tsec < hi)
        ntr = int(tm.sum())
        buy = float((~tbm[tm]).mean()) if ntr > 0 else np.nan
        big = float((tq[tm] > big_thr).mean()) if ntr > 0 else np.nan
        im = np.nan; sp = np.nan
        if len(dsec):
            j = np.searchsorted(dsec, hi * 1000)
            if j > 0:
                im = dimb[j - 1]; sp = dspread[j - 1]
        vals = dict(r60=r, accel=r_half2 - r_half1,
                    volr=(v_win / v_base) if np.isfinite(v_base) and v_base > 0 else np.nan,
                    ntr=ntr, buy=buy, big=big, imb=im, spr=sp)
        for kk, vv in vals.items():
            if np.isfinite(vv):
                bucket[kk].append(vv)

    for i in peaks:
        dec = i - LEAD
        if dec < 300:
            continue
        grab(dec, pos); peak_n += 1

    # 对照 A：随机时刻
    taken = 0; tries = 0
    while taken < 150 and tries < 600:
        tries += 1
        i = int(RNG.integers(300, n - 10))
        if any(abs(sec[i] - sec[p]) < FUT_SEC for p in peaks):
            continue
        grab(i, negA); taken += 1

    # 对照 B：同样在涨（该 60s 涨幅 >0.5%）但不是峰，且之后 30 分钟未跌 2%
    taken = 0; tries = 0
    while taken < 150 and tries < 3000:
        tries += 1
        i = int(RNG.integers(300, n - FUT_SEC - 10))
        j_lo = np.searchsorted(sec, sec[i] - 60)
        if j_lo >= i or c[i] / c[j_lo] - 1.0 < 0.005:
            continue
        if c[i + 1:i + 1 + FUT_SEC].min() / c[i] - 1.0 <= -PEAK_DROP:
            continue
        if any(abs(sec[i] - sec[p]) < FUT_SEC for p in peaks):
            continue
        grab(i, negB); taken += 1
        up_n += 1

print('峰 %d, 涨非峰 %d  %.1fs' % (peak_n, up_n, time.time() - t0))

L = []
L.append('峰前认峰 v2（修正定义泄漏 + 增加"同样在涨但不是峰"对照）  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据: data/live/2026-10-18  30 币  1s K线 + aggTrades + depth20')
L.append('决策时刻 = 峰前 %d 秒；特征窗口 = 决策前 60 秒（不含未来）' % LEAD)
L.append('峰: 1s close 为前后60s最高 且 之后30min跌>=2%%；检出 %d 个' % peak_n)
L.append('对照A: 随机时刻（每币 150 个）')
L.append('对照B: 同样在涨(60s涨幅>0.5%%)但之后30min未跌2%%（每币 150 个）')
L.append('')
L.append('  %-7s | %-28s | %-28s' % ('特征', 'A: 峰 vs 随机 (AUC)', 'B: 峰 vs 上涨非峰 (AUC)'))
L.append('  %-7s | %10s %10s %6s | %10s %10s %6s' % ('', '峰均值', '对照均值', 'AUC', '峰均值', '对照均值', 'AUC'))
for f in feats:
    a = np.array(pos[f]); bA = np.array(negA[f]); bB = np.array(negB[f])
    L.append('  %-7s | %10.4f %10.4f %6.3f | %10.4f %10.4f %6.3f'
             % (f, a.mean() if len(a) else np.nan, bA.mean() if len(bA) else np.nan, auc(a, bA),
                a.mean() if len(a) else np.nan, bB.mean() if len(bB) else np.nan, auc(a, bB)))
L.append('')
L.append('样本: 峰 %d / 随机 %d / 上涨非峰 %d' % (peak_n, len(negA['r60']), len(negB['r60'])))
L.append('')
L.append('诚实边界：')
L.append('  1. 单日、30 币、约 8 小时；峰只有 %d 个，AUC 95%% 置信区间约 ±0.2 —— 本表只能算"机制观察"，')
L.append('     任何 >0.6 或 <0.4 都不足以称为发现。')
L.append('  2. 盘口快照间隔中位 10 秒，做不了秒级盘口微结构。')
L.append('  3. 1s K 线每币仅覆盖约 65% 的秒（10-18），缺失秒会让特征有噪声。')
L.append('  4. 未计成本、未计滑点，本实验不涉及任何收益估计。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
