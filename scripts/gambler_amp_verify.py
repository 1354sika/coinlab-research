# -*- coding: utf-8 -*-
"""gambler_amp_verify.py — 验证「振幅≥2 买入、不止损、持有3天」是不是真的

已知（job-24708-36 gambler_amp_hold，严格口径，含 95% 未到 ±30% 的样本）：
    振幅≥2 买入：1天 +0.38% ｜ **3天 +1.37%** ｜ 7天 +1.16%
    随机买入  ：1天 -0.16% ｜ 3天 -0.24%   ｜ 7天 -0.45%
    增量      ：+0.54% / **+1.61%** / +1.61%
    ⇒ 首个在"全部样本 + 到期结算 + 扣费"口径下为正的信号。

必须排除的四种假象（本脚本）：
    ① 币偏差：是不是只有几个币 → 币留出（两半币各自计算）
    ② 年份偏差：是不是只有某一年 → 逐年
    ③ 集中度：是不是靠少数几笔 → 分位数、去掉 top1% 后的期望
    ④ 执行可行性：每天出多少次信号？同时要占多少仓位？
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 96; WDAY = 288; FEE = 0.0009
HOLD = 864           # 3 天
AMP_K = 2.0

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 112)
log('gambler_amp_verify · 振幅≥2 买入 / 不止损 / 持有3天 —— 四项排除')
log('=' * 112)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
del C; gc.collect()
log(f'币 {N}   全局 {T}   耗时 {time.time()-t0:.0f}s')

recs = []
for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < WDAY * 4 + HOLD + 2:
        continue
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    base = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        ar = amp / np.maximum(base, 1e-12)
    sig = np.isfinite(ar) & (ar >= AMP_K) & (np.arange(n) < n - HOLD - 2)
    idx = np.flatnonzero(sig)
    if len(idx) == 0:
        continue
    if len(idx) > 2000:
        idx = idx[::max(1, len(idx)//2000)]
    v = c[idx + HOLD] / c[idx] - 1.0 - FEE
    tpos = np.searchsorted(ts, ot)[idx]
    for q in range(len(idx)):
        recs.append((sym, int(tpos[q]), float(v[q])))
    del amp, base, ar
del RAW; gc.collect()

DF = pd.DataFrame(recs, columns=['sym', 'tpos', 'net'])
DF['year'] = pd.to_datetime(ts[DF.tpos.values], unit='ms').year
DF['day'] = (ts[DF.tpos.values] // 86_400_000)
te = DF.tpos.values >= t_cut
log(f'信号 {len(DF):,}   币 {DF.sym.nunique()}   独立天 {DF.day.nunique()}   耗时 {time.time()-t0:.0f}s')

def stat(d, tag):
    if len(d) < 30:
        log(f'  {tag:<26}样本不足'); return
    v = d.net.values
    log(f'  {tag:<26}{len(v):>8,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
        f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanpercentile(v, 95):>9.1f}')

log('\n' + '=' * 112)
log('【A】基准与切分')
log('=' * 112)
log(f'  {"口径":<26}{"笔数":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"p95%":>9}')
stat(DF, '全样本')
stat(DF[te], '时间留出（后30%）')

coins = np.array(sorted(DF.sym.unique()))
rng = np.random.default_rng(11)
half = set(rng.choice(coins, size=len(coins)//2, replace=False).tolist())
stat(DF[DF.sym.isin(list(half)) & te], '币留出 A组')
stat(DF[~DF.sym.isin(list(half)) & te], '币留出 B组')

log('\n' + '=' * 112)
log('【B】逐年（时间留出后）')
log('=' * 112)
for y in sorted(DF.year.unique()):
    d = DF[te & (DF.year == y)]
    if len(d) < 30:
        continue
    log(f'  {y}  笔 {len(d):>7,}  期望净 {100*d.net.mean():>+7.2f}%  '
        f'中位 {100*d.net.median():>+7.2f}%  胜率 {100*(d.net > 0).mean():>5.1f}%')

log('\n' + '=' * 112)
log('【C】集中度：去掉最好的 1% / 5% 后还剩多少')
log('=' * 112)
d = DF[te]
v = np.sort(d.net.values)
for q in (0.99, 0.95):
    cut = np.quantile(v, q)
    w = v[v <= cut]
    log(f'  去掉 top {100*(1-q):.0f}%（>{100*cut:.1f}%）后：{len(w):,} 笔  期望净 {100*w.mean():+.2f}%')
log(f'  原始：{len(v):,} 笔  期望净 {100*v.mean():+.2f}%')

log('\n' + '=' * 112)
log('【D】执行可行性')
log('=' * 112)
per_day = d.groupby('day').size()
log(f'  测试段 {d.day.nunique()} 天，信号 {len(d):,} 个')
log(f'  每天信号数：中位 {per_day.median():.1f}  均值 {per_day.mean():.1f}  '
    f'p90 {per_day.quantile(0.9):.1f}  最大 {per_day.max()}')
log(f'  持有 3 天 ⇒ 同时持仓数约 {per_day.mean()*3:.0f} 个（等权 1/N 资金）')
log(f'  若每笔固定 1% 资金、同时 {per_day.mean()*3:.0f} 笔 ⇒ 资金占用约 '
    f'{min(100, per_day.mean()*3):.0f}%，剩余现金留作缓冲')
log(f'  单笔期望 {100*d.net.mean():+.2f}% × 每日 {per_day.mean():.0f} 笔 ÷ 3 天轮换 '
    f'⇒ 组合日均贡献约 {100*d.net.mean()*per_day.mean()/3:.3f}%')

log('\n' + '=' * 112)
log('【判读】')
log('=' * 112)
log('  · 币留出两半都为正 + 逐年多数为正 + 去 top1% 仍为正 ⇒ 信号稳健')
log('  · 任一不满足 ⇒ 是少数币/少数笔贡献的假象')

out = os.path.join(OUT, 'amp_verify_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
