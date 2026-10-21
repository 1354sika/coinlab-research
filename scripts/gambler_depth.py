# -*- coding: utf-8 -*-
"""gambler_depth.py — 真实盘口滑点实测：把"0.5% 生死线"从假设换成实测
数据源：data-api.binance.vision /api/v3/depth（真实撮合盘口，非测试网）
方法：拉 100 档盘口，模拟市价买入 X 美元（吃 ask 档），算实际均价 vs 中间价
对照：异动币（24h 涨幅 >=+20% 或 <=-20%，即"局中/局后"的币） vs 主流币
目的：回答"扣费+滑点后到底有没有正期望"——此前一直用假设滑点，这是第一次实测。
"""
import json, time, os, urllib.request
import numpy as np
import pandas as pd

BASE = 'https://data-api.binance.vision/api/v3'
EV = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EV, 'gambler_depth.txt')
SIZES = (100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0)

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

def get(path, retry=3):
    last = None
    for i in range(retry):
        try:
            with urllib.request.urlopen(BASE + path, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise last

t0 = time.time()
log('=' * 100)
log('gambler_depth · 真实盘口滑点实测（data-api.binance.vision /depth）')
log('=' * 100)

tk = get('/ticker/24hr')
rows = []
for x in tk:
    s = x['symbol']
    if not s.endswith('USDT'):
        continue
    try:
        rows.append((s, float(x['lastPrice']), float(x['quoteVolume']), float(x['priceChangePercent'])))
    except Exception:
        pass
T = pd.DataFrame(rows, columns=['symbol', 'last', 'qvol', 'chg24'])
log(f'USDT 交易对 {len(T)}  耗时 {time.time()-t0:.0f}s')

T = T[T['qvol'] > 1e6].sort_values('qvol', ascending=False).reset_index(drop=True)
main = T.head(10)['symbol'].tolist()                                   # 主流币
pump = T[T['chg24'] >= 20].sort_values('chg24', ascending=False).head(15)['symbol'].tolist()
dump = T[T['chg24'] <= -20].sort_values('chg24').head(15)['symbol'].tolist()
log(f'主流对照 {len(main)} 个：{", ".join(main[:6])} ...')
log(f'暴涨币 {len(pump)} 个：{", ".join(pump[:8])}')
log(f'暴跌币 {len(dump)} 个：{", ".join(dump[:8])}')

def slip(sym):
    d = get(f'/depth?symbol={sym}&limit=100')
    bids = np.array([[float(p), float(q)] for p, q in d['bids']])
    asks = np.array([[float(p), float(q)] for p, q in d['asks']])
    if len(bids) == 0 or len(asks) == 0:
        return None
    mid = (bids[0, 0] + asks[0, 0]) / 2.0
    spread = (asks[0, 0] - bids[0, 0]) / mid
    out = {'symbol': sym, 'mid': mid, 'spread': spread,
           'ask1_usd': asks[0, 0] * asks[0, 1]}
    for usd in SIZES:
        rem = usd; cost = 0.0; qty = 0.0
        for p, q in asks:
            v = p * q
            if v >= rem:
                qty += rem / p; cost += rem; rem = 0.0; break
            cost += v; qty += q; rem -= v
        if rem > 0 or qty <= 0:
            out['s%d' % int(usd)] = np.nan          # 深度不够，吃穿 100 档
        else:
            avg = cost / qty
            out['s%d' % int(usd)] = (avg - mid) / mid
    return out

def run(group, name):
    log('')
    log('=' * 100)
    log(f'【{name}】{len(group)} 个币 — 市价买入的滑点（相对中间价，含价差）')
    log('=' * 100)
    hdr = f'{"symbol":16s} {"24h涨跌":>8s} {"价差%":>7s} {"1档$":>10s}'
    for usd in SIZES:
        hdr += f' {"买"+str(int(usd))+"$":>9s}'
    log(hdr)
    recs = []
    for s in group:
        try:
            chg = float(T.loc[T['symbol'] == s, 'chg24'].iloc[0])
            r = slip(s)
            if r is None:
                continue
            line = f'{s:16s} {chg:>7.1f}% {100*r["spread"]:>6.3f}% {r["ask1_usd"]:>10,.0f}'
            for usd in SIZES:
                v = r.get('s%d' % int(usd), np.nan)
                line += f' {100*v:>8.3f}%' if np.isfinite(v) else f' {"吃穿":>9s}'
            log(line)
            recs.append(r)
        except Exception as e:
            log(f'{s:16s} 失败 {type(e).__name__}')
    if recs:
        R = pd.DataFrame(recs)
        log('  --- 组内中位 ---')
        line = f'{"中位":16s} {"":>8s} {100*R["spread"].median():>6.3f}% {"":>10s}'
        for usd in SIZES:
            v = R.get('s%d' % int(usd))
            line += f' {100*np.nanmedian(v.values):>8.3f}%' if v is not None else f' {"-":>9s}'
        log(line)
    return recs

A = run(main, 'A 主流币（对照）')
B = run(pump, 'B 暴涨币（24h >= +20%）')
C = run(dump, 'C 暴跌币（24h <= -20%）')

log('')
log('=' * 100)
log('【结论】')
log('=' * 100)
for nm, recs in (('主流币', A), ('暴涨币', B), ('暴跌币', C)):
    if not recs:
        continue
    R = pd.DataFrame(recs)
    s1 = 100 * np.nanmedian(R['s1000'].values) if 's1000' in R else float('nan')
    s5 = 100 * np.nanmedian(R['s5000'].values) if 's5000' in R else float('nan')
    sp = 100 * R['spread'].median()
    log(f'  {nm:8s} 价差中位 {sp:.3f}%  买 1000$ 滑点中位 {s1:.3f}%  买 5000$ 滑点中位 {s5:.3f}%')
log('  对照此前假设：滑点 0.5% 是生死线、1% 全负')
log(f'TOTAL {time.time()-t0:.0f}s')
open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
