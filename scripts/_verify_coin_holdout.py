# -*- coding: utf-8 -*-
"""_verify_coin_holdout.py — 对 A 阶段 top 组合做「币留出」验证

A 阶段只做了**时间留出**（前 70% 训练 / 后 30% 测试）。
但一个"蝴蝶效应"要真能用，必须**跨币也成立**——否则它只是记住了某几个币的历史。

本脚本对 A 阶段测试段 lift 最高的若干组合，做三种留出对照：
  ① 时间留出（前 70% → 后 30%）—— 复现 A 阶段结论
  ② **币留出**（随机 60% 币训练 → 其余 40% 币测试）
  ③ **双重留出**（时间 + 币都不重叠）
若 ②③ 的真局率仍显著高于基准 ⇒ 该组合跨币稳定，值得继续。

产物：D:\中信证券\coinlab\evidence\butterfly\coin_holdout.txt
"""
import os, re, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
OUTDIR = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY, H48 = 288, 576
MIN_N = 100

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.append(s)

log('=' * 100)
log('_verify_coin_holdout · A 阶段 top 组合的「跨币稳定性」验证')
log('=' * 100)

# 从 static_50_all.csv 取测试段 lift 最高的组合（n_test>=100）
CSV = os.path.join(OUTDIR, 'static_50_all.csv')
d0 = pd.read_csv(CSV)
d0 = d0[np.isfinite(d0['lift_test']) & (d0['n_test'] >= 100)]
TOPS = d0.sort_values('lift_test', ascending=False).head(8)['combo'].tolist()
extra = ['hour12~18 & over>=15% & pos>=0.5 & syncDn>=3',
         'r12>=10% & r288>=30%', 'over>=25% & r288>=30%',
         'hour18~24 & r288>=30% & volr24>=5']
for e in extra:
    if e not in TOPS: TOPS.append(e)
log(f'待验证组合 {len(TOPS)} 个')

# ---------- 数据 ----------
RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

syms = [r[0] for r in RAW]
Cw = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = Cw.index.values; T, N = Cw.shape
R12 = Cw.pct_change(12).values
up3 = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
dn3 = (np.nan_to_num(R12, nan=0.0) <= -0.05).sum(axis=1)
sync_up = pd.Series(up3).rolling(24, min_periods=1).max().values
sync_dn = pd.Series(dn3).rolling(24, min_periods=1).max().values
log(f'币 {len(syms)}  矩阵 {Cw.shape}  耗时 {time.time()-t0t:.0f}s')
del Cw; gc.collect()

frames = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    hi288 = hs.rolling(WDAY, min_periods=72).max().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r = {}
    for k in (6, 12, 24, 96, 288):
        r[k] = np.full(n, np.nan); r[k][k:] = c[k:] / c[:-k] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6  = pd.Series(volr).rolling(6,  min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    for t in np.flatnonzero(np.isfinite(f_up)):
        e = min(n - 1, t + H48)
        seg = h[t+1:e+1]
        if len(seg) < 50: continue
        ipk = int(np.argmax(seg)); pk = seg[ipk]
        e2 = min(n - 1, t + 1 + ipk + WDAY)
        if e2 > t + 1 + ipk:
            dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    ii = pd.Index(ot)
    sd = pd.Series(sync_dn, index=ts).reindex(ii).values
    su = pd.Series(sync_up, index=ts).reindex(ii).values
    hh = (ot // 3600000) % 24
    sel = (over >= 0.03) & (over <= 0.40) & np.isfinite(volr24) & (volr24 >= 1) \
          & np.isfinite(f_up) & np.isfinite(dd)
    idx = np.flatnonzero(sel)[::3]
    if len(idx) < 200: continue
    frames.append(pd.DataFrame({
        't': ot[idx], 'sym': sym, 'over': over[idx],
        'r6': r[6][idx], 'r12': r[12][idx], 'r24': r[24][idx], 'r96': r[96][idx], 'r288': r[288][idx],
        'volr6': volr6[idx], 'volr24': volr24[idx], 'volr96': volr96[idx],
        'pos': (c[idx] - lo288[idx]) / np.maximum(hi288[idx] - lo288[idx], 1e-12),
        'syncUp': su[idx], 'syncDn': sd[idx], 'hour': hh[idx],
        'true': ((f_up[idx] >= 0.50) & (dd[idx] >= 0.30)).astype('int8')}))
    del hs, ls, qs; gc.collect()

df = pd.concat(frames, ignore_index=True)
Y = df['true'].values.astype('int8')
cut_t = np.quantile(df['t'].values, 0.70)
TR_t = df['t'].values <= cut_t
log(f'样本 {len(df):,}  币 {df["sym"].nunique()}  基准 {100*Y.mean():.3f}%  '
    f'训练段基准 {100*Y[TR_t].mean():.4f}%  测试段基准 {100*Y[~TR_t].mean():.4f}%  耗时 {time.time()-t0t:.0f}s')

# 币留出切分
rng = np.random.default_rng(7)
usyms = np.array(sorted(df['sym'].unique()))
perm = rng.permutation(len(usyms))
n_tr_coin = int(len(usyms) * 0.6)
TRc = set(usyms[perm[:n_tr_coin]]); TEc = set(usyms[perm[n_tr_coin:]])
isTRc = df['sym'].isin(TRc).values; isTEc = ~isTRc
log(f'币留出：训练币 {len(TRc)} → 测试币 {len(TEc)}')

def eval_combo(combo):
    m = np.ones(len(df), dtype=bool)
    for t in [x.strip() for x in combo.split(' & ')]:
        _c = re.match(r'^([A-Za-z]+)', t)
        if _c is None or _c.group(1) not in df.columns:
            return None      # 本验证脚本未构造该列（如 evt/trr24/run/vola）→ 跳过，不误判
        mm = re.match(r'^(\w+)>=([\d.]+)%?$', t)
        if mm:
            col, th = mm.group(1), float(mm.group(2)) / 100.0
            m &= (df[col].values >= th); continue
        mm = re.match(r'^(\w+)>=([\d.]+)$', t)
        if mm:
            m &= (df[mm.group(1)].values >= float(mm.group(2))); continue
        mm = re.match(r'^(\w+)<([\d.]+)%?$', t)
        if mm:
            col, th = mm.group(1), float(mm.group(2))
            if '%' in t: th /= 100.0
            m &= (df[col].values < th); continue
        mm = re.match(r'^pos<([\d.]+)$', t)
        if mm:
            m &= (df['pos'].values < float(mm.group(1))); continue
        mm = re.match(r'^hour(\d+)~(\d+)$', t)
        if mm:
            a, b = int(mm.group(1)), int(mm.group(2))
            m &= (df['hour'].values >= a) & (df['hour'].values < b); continue
        mm = re.match(r'^syncUp>=(\d+)$', t)
        if mm:
            m &= (df['syncUp'].values >= int(mm.group(1))); continue
        mm = re.match(r'^syncDn>=(\d+)$', t)
        if mm:
            m &= (df['syncDn'].values >= int(mm.group(1))); continue
        return None
    return m

log('\n' + '=' * 110)
log(f'{"组合":<58} {"时间留出(测)":>16} {"币留出(测)":>16} {"双重留出(测)":>16}')
log('=' * 110)
for combo in TOPS:
    m = eval_combo(combo)
    if m is None or m.sum() < MIN_N:
        log(f'{combo[:58]:<58} {"解析失败或样本不足":>50}'); continue
    # ① 时间留出
    a1 = df[m & ~TR_t]
    # ② 币留出
    a2 = df[m & isTEc]
    # ③ 双重
    a3 = df[m & isTEc & ~TR_t]
    def show(a, base):
        if len(a) < 20: return f'n={len(a):>5,} 太少'
        return f'{100*a["true"].mean():>6.2f}%(n={len(a):>5,})'
    b_te = Y[~TR_t].mean(); b_c = Y[isTEc].mean(); b_d = Y[isTEc & ~TR_t].mean()
    log(f'{combo[:58]:<58} {show(a1, b_te):>16} {show(a2, b_c):>16} {show(a3, b_d):>16}')
log(f'\n对照基准：时间测试段 {100*b_te:.3f}% ｜ 币测试段 {100*b_c:.3f}% ｜ 双重 {100*b_d:.3f}%')

log('\n【判读】')
log('  · 时间留出 高 但 币留出 掉回基准 ⇒ 该组合只是记住了特定币的历史，不可用')
log('  · 币留出 仍显著高于基准 ⇒ 跨币稳定，值得继续')

log(f'\nTOTAL {time.time()-t0t:.0f}s')
with open(os.path.join(OUTDIR, 'coin_holdout.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log(f'报告已写 {os.path.join(OUTDIR, "coin_holdout.txt")}')
