# -*- coding: utf-8 -*-
"""gambler_cross_transmit · 条件传导实验
BTC 剧烈波动（15 分钟跌/涨超阈值）时，其他币未来 60 分钟跌 2% / 涨 2% 的概率
对比基线，看提升倍数。这是"什么时候会一起跌"而非"谁先谁后"。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60; STEP = 25

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_cross_transmit · 条件传导（BTC 剧烈波动 → 跟跌/跟涨概率）')
log('=' * 100)

# 第一遍：提取 BTC / ETH 的分钟序列
leader = {}
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32'},
                      chunksize=3_000_000):
    for sym in ('BTCUSDT','ETHUSDT'):
        g = ch[ch['symbol'] == sym]
        if len(g):
            g = g.sort_values('open_time')
            arr = leader.get(sym)
            ot = g['open_time'].values.astype('int64')
            cl = g['close'].values.astype('float64')
            if arr is None:
                leader[sym] = (ot, cl)
            else:
                leader[sym] = (np.concatenate([arr[0], ot]), np.concatenate([arr[1], cl]))
    del ch; gc.collect()
for sym, (ot, cl) in leader.items():
    log(f'{sym}: {len(ot):,} 分钟')

# 计算 BTC/ETH 每个时间点的 r15
lead_r15 = {}
for sym, (ot, cl) in leader.items():
    r = np.full(len(cl), np.nan)
    r[15:] = cl[15:] / cl[:-15] - 1.0
    # 映射 open_time -> r15
    lead_r15[sym] = dict(zip(ot.tolist(), r.tolist()))
btc_map = lead_r15['BTCUSDT']
eth_map = lead_r15.get('ETHUSDT', {})

# 第二遍：其他币的样本
rows = []   # (btc_r15, eth_r15, down2, up2, sym)
sym_id = {}
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym in ('BTCUSDT','ETHUSDT'):
            continue
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= 300:
            continue
        cc = c[:m]
        fmax = pd.Series(h).rolling(HOLD).max().shift(-HOLD).values[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        ok = np.isfinite(fmax) & np.isfinite(fmin)
        idx = np.arange(15, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        b_r = np.array([btc_map.get(int(k), np.nan) for k in ot[idx]], dtype='float64')
        e_r = np.array([eth_map.get(int(k), np.nan) for k in ot[idx]], dtype='float64')
        d2 = (fmin[idx] <= cc[idx] * 0.98).astype('int8')
        u2 = (fmax[idx] >= cc[idx] * 1.02).astype('int8')
        good = np.isfinite(b_r)
        if good.sum() == 0:
            continue
        rows.append(pd.DataFrame({'b_r15': b_r[good], 'e_r15': e_r[good],
                                  'd2': d2[good], 'u2': u2[good]}))
    del ch; gc.collect()

df = pd.concat(rows, ignore_index=True)
log(f'样本 {len(df):,}')

base_d2 = df['d2'].mean(); base_u2 = df['u2'].mean()
log(f'基线：跌2% {base_d2:.2%}  涨2% {base_u2:.2%}')
log(f'BTC r15 分位：p1={np.nanpercentile(df["b_r15"],1):.3%} p5={np.nanpercentile(df["b_r15"],5):.3%} p50={np.nanpercentile(df["b_r15"],50):.3%} p95={np.nanpercentile(df["b_r15"],95):.3%} p99={np.nanpercentile(df["b_r15"],99):.3%}')
log(f'BTC r15 绝对值>1% 的样本占比：{(df["b_r15"].abs()>0.01).mean():.2%}')

log('\n' + '=' * 100)
log('【BTC 15 分钟剧烈波动 → 其他币未来 60 分钟跌2%/涨2%】')
log('=' * 100)
log(f'  {"条件":<22s}{"n":>9s}{"跌2%":>9s}{"提升":>8s}{"涨2%":>9s}{"提升":>8s}')

for name, mask in [
    ('BTC r15 <= -0.5%', df['b_r15'] <= -0.005),
    ('BTC r15 <= -1%', df['b_r15'] <= -0.01),
    ('BTC r15 <= -1.5%', df['b_r15'] <= -0.015),
    ('BTC r15 <= -2%', df['b_r15'] <= -0.02),
    ('BTC r15 <= -3%', df['b_r15'] <= -0.03),
    ('BTC r15 >= +0.5%', df['b_r15'] >= 0.005),
    ('BTC r15 >= +1%', df['b_r15'] >= 0.01),
    ('BTC r15 >= +1.5%', df['b_r15'] >= 0.015),
    ('BTC r15 >= +2%', df['b_r15'] >= 0.02),
    ('BTC r15 >= +3%', df['b_r15'] >= 0.03),
]:
    n = mask.sum()
    if n < 50: continue
    d = df.loc[mask, 'd2'].mean(); u = df.loc[mask, 'u2'].mean()
    log(f'  {name:<22s}{n:>9,}{d:>9.2%}{d/base_d2:>7.2f}x{u:>9.2%}{u/base_u2:>7.2f}x')

log('\n【ETH 15 分钟剧烈波动 → 其他币】')
for name, mask in [
    ('ETH r15 <= -0.5%', df['e_r15'] <= -0.005),
    ('ETH r15 <= -1%', df['e_r15'] <= -0.01),
    ('ETH r15 <= -2%', df['e_r15'] <= -0.02),
    ('ETH r15 >= +0.5%', df['e_r15'] >= 0.005),
    ('ETH r15 >= +1%', df['e_r15'] >= 0.01),
    ('ETH r15 >= +2%', df['e_r15'] >= 0.02),
]:
    n = mask.sum()
    if n < 50: continue
    d = df.loc[mask, 'd2'].mean(); u = df.loc[mask, 'u2'].mean()
    log(f'  {name:<22s}{n:>9,}{d:>9.2%}{d/base_d2:>7.2f}x{u:>9.2%}{u/base_u2:>7.2f}x')

log('\n' + '=' * 100)
log('【联合：BTC 跌 + 该币自身涨（追涨后遇大盘跌）】')
log('=' * 100)
# 这里只做粗略：BTC 跌 3% 时，跌2%提升倍数若 >1.3x 则有过滤价值

log('\n【判读】')
log('  · 提升倍数 > 1.3x 且样本 > 500 ⇒ 条件传导有效，可做 A1/A2 的过滤器')
log('  · 提升 ≈ 1.0x ⇒ 传导确实是陷阱（与 leadlag/downcross 一致）')
log('  · 若"跌"提升大而"涨"提升小 ⇒ 跌的传导不对称，这正是用户要的"看跌能力"')

out = os.path.join(OUT, 'cross_transmit_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
