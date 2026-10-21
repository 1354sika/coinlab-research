"""
峰前认峰 · 探索性实测（data/live 2026-10-18，1s + 逐笔 + 10s 盘口）

问题：能不能在"峰"发生之前 1~5 分钟，从 1s 微观结构里看出"这根就是顶"？
做法：不训练模型，直接问单特征有没有区分力（AUC），再用 SHUF/RAND 对照。

峰的定义（事后标注，仅用于当标签）：
  1s 收盘价是前后 60 秒内最高，且之后 30 分钟内至少跌 2%。
  同一段 30 分钟内只保留最高点。

特征（全部只用峰前信息，因果）：
  r10/r30/r60      峰前 10/30/60 秒累计收益
  accel            r10 - r30/3（加速）
  volr30           前 30 秒成交量 / 前 300 秒均量
  ntr60            前 60 秒逐笔笔数
  buy60            前 60 秒主动买占比（is_buyer_maker=False）
  big60            前 60 秒大单占比（qty > 该币 P99）
  imb              峰前最近一张盘口的买卖深度失衡 bid/(bid+ask)
  spr              该盘口相对价差

对照：同币随机时刻（距任何峰 >30 分钟），同样抽特征。
诚实边界：单日、30 币、约 8 小时；样本小；结论只能算"假设"，不能算"发现"。
"""
import pandas as pd, numpy as np, os, time

BASE = r'D:/中信证券/coinlab/data/live/2026-10-18'
OUT = r'D:/中信证券/coinlab/evidence/live_peak_probe.txt'
T0 = pd.Timestamp('2026-10-18 08:00:00').value // 10 ** 6   # ms
PEAK_DROP = 0.02
FUT_SEC = 1800
LOCAL_WIN = 60
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
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    ra = r[:len(a)].sum()
    return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


t0 = time.time()
L = []
L.append('峰前认峰 · 探索性实测  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('数据: data/live/2026-10-18  1s K线 + aggTrades + depth20(20档)')
L.append('峰定义: 1s close 为前后60s最高 且 之后30min跌幅>=2%; 同段只留最高点')
L.append('')

k = smart_read(os.path.join(BASE, 'klines_1s.csv'))
k = k[k.open_time >= T0]
k['sec'] = k.open_time // 1000
tr = smart_read(os.path.join(BASE, 'aggTrades.csv'))
tr = tr[tr.time >= T0]
tr['sec'] = tr.time // 1000
dp = smart_read(os.path.join(BASE, 'depth20.csv'))
dp = dp[dp.ts_ms >= T0]
dp['sec'] = dp.ts_ms // 1000
print('读入 k=%d tr=%d dp=%d  %.1fs' % (len(k), len(tr), len(dp), time.time() - t0))
L.append('读入: 1s K线 %d 行 / 逐笔 %d 行 / 盘口 %d 行' % (len(k), len(tr), len(dp)))

feats = ['r10', 'r30', 'r60', 'accel', 'volr30', 'ntr60', 'buy60', 'big60', 'imb', 'spr']
pos = {f: [] for f in feats}
neg = {f: [] for f in feats}
peak_rows = []

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
        fut = c[i + 1:i + 1 + FUT_SEC].min()
        if fut / c[i] - 1.0 <= -PEAK_DROP:
            cand.append(i)
    # 去重：30 分钟内只留最高
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
    dg = dp[dp.symbol == sym].sort_values('ts_ms')
    dsec = np.sort(dg.ts_ms.to_numpy())
    # 每个快照的 bid/ask 总量
    if len(dg):
        piv = dg.pivot_table(index='ts_ms', columns='side', values='qty', aggfunc='sum')
        piv = piv.reindex(columns=['bid', 'ask']).fillna(0.0)
        ds_ms = piv.index.to_numpy()
        dimb = (piv.bid / (piv.bid + piv.ask).replace(0, np.nan)).to_numpy()
        # spread 用 best bid/ask
        best = dg[dg.level == 0].pivot_table(index='ts_ms', columns='side', values='price', aggfunc='first')
        best = best.reindex(columns=['bid', 'ask'])
        dspread = ((best.ask - best.bid) / ((best.ask + best.bid) / 2)).to_numpy()
    else:
        dimb = np.array([]); dspread = np.array([])

    def grab(i, bucket):
        def ret(sec_back):
            j = np.searchsorted(sec, sec[i] - sec_back)
            return c[i] / c[j] - 1.0 if j < i else np.nan

        r10, r30, r60 = ret(10), ret(30), ret(60)
        m = (sec >= sec[i] - 300) & (sec < sec[i])
        v300 = v[m]
        vma = v300.mean() if len(v300) else np.nan
        m30 = (sec >= sec[i] - 30) & (sec < sec[i])
        v30 = v[m30].sum() if m30.any() else np.nan
        tm = (tsec >= sec[i] - 60) & (tsec < sec[i])
        ntr = int(tm.sum())
        buy = float((~tbm[tm]).mean()) if ntr > 0 else np.nan
        big = float((tq[tm] > big_thr).mean()) if ntr > 0 else np.nan
        im = np.nan; sp = np.nan
        if len(dsec):
            j = np.searchsorted(dsec, (sec[i] - 1) * 1000)
            if j > 0:
                im = dimb[j - 1] if j - 1 < len(dimb) else np.nan
                sp = dspread[j - 1] if j - 1 < len(dspread) else np.nan
        vals = dict(r10=r10, r30=r30, r60=r60,
                    accel=(r10 - r30 / 3.0) if np.isfinite(r10) and np.isfinite(r30) else np.nan,
                    volr30=(v30 / vma) if vma and np.isfinite(vma) and np.isfinite(v30) else np.nan,
                    ntr60=ntr, buy60=buy, big60=big, imb=im, spr=sp)
        for key, val in vals.items():
            if np.isfinite(val):
                bucket[key].append(val)

    for i in peaks:
        grab(int(i), pos)
        peak_rows.append((sym, sec[i], c[i]))
    # 对照组：同币随机时刻，离峰 >30min，且在数据范围内
    taken = 0
    tries = 0
    while taken < 120 and tries < 400:
        tries += 1
        i = int(RNG.integers(600, n - 10))
        if any(abs(sec[i] - sec[p]) < FUT_SEC for p in peaks):
            continue
        grab(i, neg)
        taken += 1

print('峰总数 %d  对照采样完成  %.1fs' % (len(peak_rows), time.time() - t0))
L.append('检出峰 %d 个（30 币，约 8 小时）；对照为同币随机时刻' % len(peak_rows))
L.append('')
L.append('=== 单特征区分力（峰前 10~300 秒的微观特征 vs 随机时刻）')
L.append('  %-8s %10s %10s %10s %10s %8s' % ('特征', '峰均值', '峰中位', '对照中位', '对照均值', 'AUC'))
for f in feats:
    a = np.array(pos[f]); b = np.array(neg[f])
    if len(a) < 5 or len(b) < 5:
        L.append('  %-8s %10s  (样本不足 n=%d/%d)' % (f, '-', len(a), len(b)))
        continue
    L.append('  %-8s %10.4f %10.4f %10.4f %10.4f %8.3f'
             % (f, a.mean(), np.median(a), np.median(b), b.mean(), auc(a, b)))
L.append('')
L.append('读法: AUC 0.5=无区分力；>0.6 才算有点意思；<0.4 且方向稳定才是"反着有用"。')
L.append('')
L.append('=== 峰样本的时间/价格（前 10 个）')
for sym, s, p in peak_rows[:10]:
    L.append('  %-10s %s  %.6f' % (sym, pd.to_datetime(s * 1000, unit='ms'), p))

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
