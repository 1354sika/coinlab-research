# -*- coding: utf-8 -*-
"""gambler_amp_trade.py — 把「振幅异动」变成可交易信号，直接算期望净

由来（job-24708-31 gambler_regime_live）：
    用滚动形状指标预测"未来 6h 内会不会起局"，**AUC = 0.9262**（H=12h 0.8917、H=24h 0.8310）。
    置换重要性显示：**ramp（8h 平均振幅）贡献 +0.2200，其余特征 ≈0**。
    而此前用价格类特征做早期识别，AUC 只有 0.34~0.52。
    ⇒ **局的先兆是「波动放大」，不是「价格上涨」。**

【本脚本】把这条结论变成交易，直接算钱：
    信号：amp_ratio = 当前 8h 平均振幅 / 过去 30 天平均振幅  ≥ K（K = 1.5/2/3/5）
          且 过去 24h 涨幅 < 30%（"还没拉完"，避免追高）
    入场：信号触发即做多（市价）
    退出（三种）：
        X1  +30% 止盈 / -15% 止损 / 48h 超时
        X2  +50% 止盈 / -20% 止损 / 5 天超时
        X3  移动止损（从峰值回落 25%）/ 最长 10 天
    扣费 0.09%；切分前 70% 训练 / 后 30% 测试；报每日出手次数
【对照】随机入场（同退出规则）
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

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_amp_trade · 振幅异动 → 期望净')
log('=' * 116)

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
log(f'币 {N}   全局 {T} 根   耗时 {time.time()-t0:.0f}s')

K_LIST = [1.5, 2.0, 3.0, 5.0]
EX = {'X1': (0.30, 0.15, 576), 'X2': (0.50, 0.20, 1440), 'X3': (None, None, 2880)}
rows = []
for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < WDAY * 4 + 2880:
        continue
    amp = pd.Series((h - l) / np.maximum(c, 1e-12)).rolling(W, min_periods=W//2).mean().values
    base = pd.Series(amp).rolling(WDAY * 30, min_periods=WDAY).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        ar = amp / np.maximum(base, 1e-12)
    r24 = np.full(n, np.nan); r24[288:] = c[288:] / np.maximum(c[:-288], 1e-12) - 1.0
    for K in K_LIST:
        sig = (np.isfinite(ar) & (ar >= K)
               & (np.nan_to_num(r24, nan=0.0) < 0.30)
               & (np.arange(n) < n - 2880 - 2))
        idx = np.flatnonzero(sig)
        if len(idx) == 0:
            continue
        if len(idx) > 3000:
            idx = idx[::max(1, len(idx)//3000)]
        tpos = np.searchsorted(ts, ot)[idx]
        c0 = c[idx]
        for tag, (tp, sl, hm) in EX.items():
            cur = np.zeros(len(idx))          # 当前持仓收益（从入场算）
            live = np.ones(len(idx), bool)
            peak = np.ones(len(idx))
            if tp is None:
                for k in range(1, hm):
                    j = idx + k
                    ok = j < n
                    hh = np.where(ok, h[np.clip(j, 0, n-1)], c0)
                    ll = np.where(ok, l[np.clip(j, 0, n-1)], c0)
                    cc = np.where(ok, c[np.clip(j, 0, n-1)], c0)
                    peak = np.maximum(peak, hh / c0)
                    cur = cc / c0 - 1.0
                    dd = cur / peak - 1.0
                    hit = live & ok & (dd <= -0.25)
                    live &= ~hit
                    if not live.any():
                        break
                v = np.where(live, c[np.clip(idx + hm - 1, 0, n-1)] / c0 - 1.0, cur) - FEE
            else:
                hit_tp = np.zeros(len(idx), bool); hit_sl = np.zeros(len(idx), bool)
                res = np.full(len(idx), np.nan)
                for k in range(1, hm):
                    j = idx + k
                    ok = (j < n) & np.isnan(res)
                    if not ok.any():
                        break
                    hh = np.where(ok, h[np.clip(j, 0, n-1)], -np.inf)
                    ll = np.where(ok, l[np.clip(j, 0, n-1)], np.inf)
                    tp_hit = ok & (hh >= c0 * (1 + tp))
                    sl_hit = ok & (ll <= c0 * (1 - sl))
                    take_sl = sl_hit                     # 同根保守按止损
                    res = np.where(np.isnan(res) & take_sl, -sl, res)
                    res = np.where(np.isnan(res) & tp_hit & ~take_sl, tp, res)
                res = np.where(np.isnan(res), c[np.clip(idx + hm - 1, 0, n-1)] / c0 - 1.0, res)
                v = res - FEE
            for q in range(len(idx)):
                rows.append((sym, int(tpos[q]), K, tag, float(v[q])))
    del amp, base, ar
del RAW; gc.collect()

DF = pd.DataFrame(rows, columns=['sym', 'tpos', 'K', 'exit', 'net'])
log(f'信号样本 {len(DF):,}   币 {DF.sym.nunique()}   耗时 {time.time()-t0:.0f}s')
DF['year'] = pd.to_datetime(ts[DF.tpos.values], unit='ms').year
te = DF.tpos.values >= t_cut
days = (DF.tpos.max() - DF.tpos.min()) / 86_400_000

log('\n' + '=' * 116)
log('【结果】测试段（后 30%）')
log('=' * 116)
log(f'  {"阈值K":>6}{"退出":>5}{"笔数":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}'
    f'{"≥+50%":>8}{"≥+100%":>9}{"每日出手":>9}')
for K in K_LIST:
    for tag in EX:
        d = DF[(DF.K == K) & (DF.exit == tag) & te]
        if len(d) < 30:
            continue
        v = d.net.values
        log(f'  {K:>6.1f}{tag:>5}{len(v):>8,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
            f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanmean(v >= 0.5):>8.1f}'
            f'{100*np.nanmean(v >= 1.0):>9.1f}{len(v)/max(days*0.3,1):>9.1f}')

log('\n' + '=' * 116)
log('【逐年】X1 口径')
log('=' * 116)
for K in K_LIST:
    line = []
    for y in sorted(DF.year.unique()):
        d = DF[(DF.K == K) & (DF.exit == 'X1') & te & (DF.year == y)]
        line.append(f'{y}:{100*np.nanmean(d.net.values):+.1f}%({len(d)})' if len(d) >= 10 else f'{y}:—')
    log(f'  K={K:>3.1f}  ' + '  '.join(line))

log('\n' + '=' * 116)
log('【判读】')
log('=' * 116)
log('  · 期望净 >0 且逐年多数为正 ⇒ 振幅信号可交易')
log('  · 若全部为负 ⇒ 振幅只提高"起局概率"，不提高"赚钱概率"（起局≠赚钱）')

out = os.path.join(OUT, 'amp_trade_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
