# -*- coding: utf-8 -*-
"""订单流体检：把 aggTrades 聚成分钟级，看主动买卖比/大单占比/笔数
对**下一分钟收益**有没有预测力（只做多，四段检验）。

这是"换原料"的第一步——之前所有牌都来自 OHLCV，这次用逐笔成交。
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

F = r'D:\中信证券\coinlab\data\live\2026-10-18\aggTrades.csv'
OUT = r'D:\中信证券\coinlab\evidence\flow_gate.txt'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

say('=' * 100)
say('订单流体检 · aggTrades → 分钟特征 → 下一分钟收益')
say('=' * 100)
t0 = time.time()

# ---------- 1. 读并聚合到分钟 ----------
rows = []
nrow = 0
for ch in pd.read_csv(F, chunksize=2_000_000,
                      dtype={'agg_id': 'int64', 'symbol': 'str', 'price': 'float64',
                             'qty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'}):
    nrow += len(ch)
    ch['minute'] = ch['time'] // 60000
    ch['bq'] = np.where(~ch['is_buyer_maker'], ch['qty'], 0.0)   # 主动买（taker 买）
    ch['sq'] = np.where(ch['is_buyer_maker'], ch['qty'], 0.0)    # 主动卖
    g = ch.groupby(['symbol', 'minute'], sort=False).agg(
        px=('price', 'last'), vol=('qty', 'sum'), bq=('bq', 'sum'),
        sq=('sq', 'sum'), n=('qty', 'size'), mx=('qty', 'max'))
    rows.append(g.reset_index())
    if nrow % 4_000_000 < 2_000_000:
        say(f'  读 {nrow:,} 行  {time.time()-t0:.0f}s')

df = pd.concat(rows, ignore_index=True)
del rows
df = df.groupby(['symbol', 'minute'], sort=False).agg(
    px=('px', 'last'), vol=('vol', 'sum'), bq=('bq', 'sum'),
    sq=('sq', 'sum'), n=('n', 'sum'), mx=('mx', 'max')).reset_index()
df = df.sort_values(['symbol', 'minute']).reset_index(drop=True)
say(f'  聚合完成: {len(df):,} 个 (币,分钟) 格 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

# ---------- 2. 特征 ----------
df['tbr'] = df['bq'] / np.maximum(df['bq'] + df['sq'], 1e-12)      # 主动买占比
df['nlog'] = np.log1p(df['n'])
df['vlog'] = np.log1p(df['vol'])

# 大单占比：用该币当分钟 qty 的相对大小（mx 相对该币近 60 分钟均值）
g = df.groupby('symbol', sort=False)
df['mxr'] = df['mx'] / np.maximum(g['mx'].transform(lambda s: s.rolling(60, min_periods=10).mean()), 1e-12)
df['volr'] = df['vol'] / np.maximum(g['vol'].transform(lambda s: s.rolling(60, min_periods=10).mean()), 1e-12)
df['tbr_m'] = g['tbr'].transform(lambda s: s.rolling(5, min_periods=3).mean())

# 标签：下一分钟收益（同一币的下一个分钟格）
df['ret1'] = g['px'].transform(lambda s: s.shift(-1) / s - 1.0)
# 只用有下一分钟的连续点：minute 差 == 1
df['mn'] = g['minute'].shift(-1)
df['ok'] = (df['mn'] - df['minute']) == 1
df = df[df['ok'] & np.isfinite(df['ret1'])].copy()

say(f'  有标签样本: {len(df):,}  时间 {pd.to_datetime(df.minute.min()*60000, unit="ms")} ~ {pd.to_datetime(df.minute.max()*60000, unit="ms")}')

# ---------- 3. 四段检验 ----------
FEATS = ['tbr', 'tbr_m', 'nlog', 'vlog', 'mxr', 'volr']
df = df.sort_values('minute').reset_index(drop=True)
q = np.linspace(0, len(df), 5).astype(int)
segs = [df.iloc[q[i]:q[i+1]] for i in range(4)]

say('')
say('=' * 100)
say('【一】每个特征分 5 档，看下一分钟收益（只做多 = 买高主动买占比）')
say('=' * 100)
for f in FEATS:
    say(f'\n  [{f}]')
    line = '    档: '
    for i in range(4):
        s = segs[i]
        x = s[f].values
        r = s['ret1'].values
        ok = np.isfinite(x) & np.isfinite(r)
        if ok.sum() < 500:
            line += f'  段{i+1}:样本少'
            continue
        x, r = x[ok], r[ok]
        # 最高档（前 20%）
        thr = np.nanquantile(x, 0.8)
        top = r[x >= thr]
        allm = r.mean()
        line += f'  段{i+1}:{top.mean()*100:+.4f}%(n={len(top)})'
    say(line)
    # 全样本最高档
    x = df[f].values; r = df['ret1'].values
    ok = np.isfinite(x) & np.isfinite(r)
    x, r = x[ok], r[ok]
    thr = np.nanquantile(x, 0.8)
    top = r[x >= thr]
    say(f'    全样本最高档: {top.mean()*100:+.4f}%  n={len(top):,}  对比全样本 {r.mean()*100:+.4f}%')
    # t 值
    if len(top) > 30:
        t = top.mean() / (top.std(ddof=1) / np.sqrt(len(top)))
        say(f'    t = {t:+.2f}   {"✅" if t > 2.5 else "⚠" if t > 1.5 else "❌"}')

say('')
say('=' * 100)
say('【二】判据')
say('=' * 100)
say('  · 四段同号且 t>2.5 ⇒ 这张订单流牌有资格进赌徒基元池')
say('  · 只有全样本显著但分段不同号 ⇒ 时段效应，不是可交易结构')
say(f'\nTOTAL {time.time()-t0:.0f}s')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'\n写入 {OUT}')
