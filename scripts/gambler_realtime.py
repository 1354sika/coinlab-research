# -*- coding: utf-8 -*-
"""gambler_realtime.py — 接真数据：实时扫描全市场，输出"当前最可能正在做局"的币
数据源：data-api.binance.vision（本机唯一可达的币安公开行情域名）
流程：拉 24h ticker → 选成交额 top N + 涨幅异常币 → 拉最近 500 根 5m → 算 24 特征 → g3 模型打分
"""
import json, time, os, urllib.request
import numpy as np
import pandas as pd
import joblib

BASE = 'https://data-api.binance.vision/api/v3'
EV = r'D:\中信证券\coinlab\evidence'
MD = r'D:\中信证券\coinlab\models'
OUT = os.path.join(EV, 'gambler_realtime.txt')
N_SCAN = 200          # 最多扫多少币

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
            time.sleep(0.6)
    raise last

t0 = time.time()
log('=' * 100)
log('gambler_realtime · 真数据实时扫描（data-api.binance.vision）')
log('=' * 100)

ti = get('/ticker/24hr')
rows = []
for x in ti:
    s = x['symbol']
    if not s.endswith('USDT'):
        continue
    try:
        rows.append((s, float(x['lastPrice']), float(x['quoteVolume']),
                     float(x['priceChangePercent']), int(x['count'])))
    except Exception:
        pass
tk = pd.DataFrame(rows, columns=['symbol', 'last', 'qvol', 'chg24', 'trades24'])
log(f'USDT 交易对 {len(tk)}  耗时 {time.time()-t0:.0f}s')

# 选币：成交额 top N 并集（成交额太小的币 K 线噪声大）
tk = tk.sort_values('qvol', ascending=False).reset_index(drop=True)
watch = tk.head(N_SCAN)['symbol'].tolist()
log(f'扫描池 {len(watch)} 币（按 24h 成交额排序）')

mdl = None
mfile = os.path.join(MD, 'g3_涨_候选=吸筹&.joblib')
for cand in [mfile, os.path.join(MD, 'g3_涨_候选=吸筹&_未.joblib')]:
    if os.path.exists(cand):
        mdl = joblib.load(cand)
        log('加载模型', cand)
        break
if mdl is None:
    for f in sorted(os.listdir(MD)):
        if f.startswith('g3_涨'):
            mdl = joblib.load(os.path.join(MD, f)); log('加载模型', f); break
if mdl is None:
    log('✗ 找不到 g3 涨模型'); raise SystemExit(1)
m = mdl['model']; FNAMES = mdl['features']

def rmean(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).mean().values

def rmax(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).max().values

def rmin(a, k):
    return pd.Series(a).rolling(k, min_periods=max(2, k // 4)).min().values

def feats(g):
    n = len(g)
    c = g['close'].values.astype('float64'); h = g['high'].values.astype('float64')
    l = g['low'].values.astype('float64'); v = g['volume'].values.astype('float64')
    q = g['quote_volume'].values.astype('float64'); tr = g['trades'].values.astype('float64')
    tq = g['taker_quote'].values.astype('float64')
    F = {}
    for k in (3, 6, 12, 24, 48, 96, 192, 288):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0
        F['ret%d' % k] = r
    mtr = rmean(tr, 48); mq = rmean(q, 48)
    trr = np.where(mtr > 0, tr / np.maximum(mtr, 1e-12), np.nan)
    volr = np.where(mq > 0, q / np.maximum(mq, 1e-12), np.nan)
    for k in (12, 48, 192):
        mm = rmean(v, k); F['volr%d' % k] = np.where(mm > 0, v / np.maximum(mm, 1e-12), np.nan)
        mk = rmean(q, k); F['vol%d' % k] = np.where(mk > 0, q / np.maximum(mk, 1e-12), np.nan)
    for k in (24, 96, 288):
        F['hh%d' % k] = rmax(h, k) / c - 1.0
        F['ll%d' % k] = rmin(l, k) / c - 1.0
    ra = tq / np.maximum(q, 1e-12)
    F['trr'] = ra; F['trr_dev'] = ra - rmean(ra, 48)
    r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
    F['up12'] = rmean((r1 > 0).astype('float64'), 12)
    F['dn12'] = rmean((r1 < 0).astype('float64'), 12)
    F['sig_tr'] = np.nan_to_num(trr, nan=0.0)
    F['sig_vq'] = np.nan_to_num(volr, nan=0.0)
    l48 = rmin(l, 48); l288 = rmin(l, 288)
    F['sig_newlow'] = (l48 <= l288 * 0.995).astype('float64')
    X = np.stack([F[k][-1] for k in FNAMES], axis=0).astype('float32')[None, :]
    cand = (trr[-1] >= 5) and (volr[-1] >= 3) and (F['ret288'][-1] <= 0.30)
    return X, bool(cand), float(F['ret288'][-1]), bool(F['sig_newlow'][-1])

res = []
ok_n = 0
for i, s in enumerate(watch):
    try:
        kl = get(f'/klines?symbol={s}&interval=5m&limit=500')
        g = pd.DataFrame(kl, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                      'close_time', 'quote_volume', 'trades', 'taker_base',
                                      'taker_quote', 'ignore'])
        for ccol in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_quote']:
            g[ccol] = g[ccol].astype('float64')
        # 上溯补足 288+ 根（limit=500 已够）
        X, is_cand, ret288, newlow = feats(g)
        if not np.isfinite(X).all():
            continue
        p = float(m.predict_proba(X)[0, 1])
        ok_n += 1
        chg = float(g['close'].iloc[-1] / g['close'].iloc[-49] - 1.0)
        res.append((s, p, is_cand, ret288, chg, float(g['quote_volume'].iloc[-1])))
    except Exception as e:
        continue
    if (i + 1) % 50 == 0:
        log(f'  已扫 {i+1}/{len(watch)}  成功 {ok_n}  耗时 {time.time()-t0:.0f}s')

R = pd.DataFrame(res, columns=['symbol', 'p', '候选', 'ret288', 'chg1h', 'qvol'])
log(f'扫描完成：成功 {ok_n} / {len(watch)}  耗时 {time.time()-t0:.0f}s')
log('')
log('=' * 100)
log('【A】模型分最高的 25 个币（不限候选）')
log('=' * 100)
log(f'{"symbol":16s} {"P(涨5%/1h)":>11s} {"是否候选":>8s} {"24h涨幅":>9s} {"近1h":>8s}')
for _, r in R.sort_values('p', ascending=False).head(25).iterrows():
    log(f'{r["symbol"]:16s} {r["p"]:>11.4f} {"是" if r["候选"] else "否":>8s} {100*r["ret288"]:>8.1f}% {100*r["chg1h"]:>7.2f}%')

log('')
log('=' * 100)
log('【B】当前满足"吸筹候选"的币（笔数×5 & 量×3 & 未拉完）')
log('=' * 100)
C = R[R['候选']].sort_values('p', ascending=False)
if len(C) == 0:
    log('  当前无币满足候选条件')
else:
    log(f'{"symbol":16s} {"P(涨5%/1h)":>11s} {"24h涨幅":>9s} {"近1h":>8s} {"成交额":>14s}')
    for _, r in C.head(30).iterrows():
        log(f'{r["symbol"]:16s} {r["p"]:>11.4f} {100*r["ret288"]:>8.1f}% {100*r["chg1h"]:>7.2f}% {r["qvol"]:>14,.0f}')

log('')
log('【C】判读')
log(f'  扫描池 {len(watch)} 个 USDT 对，成功拉到 K 线 {ok_n} 个')
log(f'  其中"吸筹候选" {len(C)} 个（占比 {100*len(C)/max(ok_n,1):.1f}%）')
if len(C):
    log(f'  候选里 P>=0.5 的 {int((C["p"]>=0.5).sum())} 个，P>=0.7 的 {int((C["p"]>=0.7).sum())} 个')
log(f'TOTAL {time.time()-t0:.0f}s')

open(OUT, 'w', encoding='utf-8').write('\n'.join(buf))
log('报告已写', OUT)
