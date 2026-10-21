# -*- coding: utf-8 -*-
"""gambler_regime_detect.py — 换五种方式重新"分辨局"（用户：可能是你分辨的方式错了）

用户判断（2026-10-17）：
    「可能确实没有局，但是翻几倍的波动是真的。同时没有局的概率是小到几乎没有的。
      高等数学要处理的或许并不是分辨有没有局，但也可能是你分辨局的方式错了。」

【上一版 VR 检验的五个缺陷（自检）】
    1. 全段平均：若局是"前 90% 噪声 + 后 10% 猛拉"，整段 VR 必然≈1 → 必须滑窗局部化
    2. 只用收盘价：信息可能在振幅/成交量，不在价 → 必须加成交维度
    3. 只看二阶矩：局可能只改右尾（偏度/峰度） → 必须看高阶矩
    4. 线性检验：局可能是状态依赖（平时回归、异动时趋势） → 必须条件化
    5. 事后选段：认定点是"涨了 30%"才入选，本身有选择效应 → 需与同长度随机段对照

【本脚本五路检测，全部对照"局内 vs 局外等长随机段"】
    D1 滑窗 VR：窗口 96 根、步长 24，统计"子段 VR>1.5 的占比"
    D2 条件 VR：只在 volr6 ≥ 5（量能异动根）上构造收益序列再算 VR
    D3 高阶矩：偏度、峰度、>3σ 右尾占比
    D4 成交额 Hurst（R/S 法）：局内成交额的长程相关性是否更强
    D5 变点：CUSUM 最大偏移 / √n（结构性突变的强度）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
MAX_BARS = 576; TRIG = 0.30; R = 0.10

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 118)
log('gambler_regime_detect · 五路重新检测「局」')
log('=' * 118)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'读入 {len(RAW)} 币   耗时 {time.time()-t0:.0f}s')

def vr(x, q):
    r = np.diff(x)
    n = len(r)
    if n < q * 4:
        return np.nan
    mu = r.mean(); var1 = ((r - mu) ** 2).sum() / (n - 1)
    if var1 <= 0:
        return np.nan
    rq = x[q:] - x[:-q]
    varq = ((rq - q * mu) ** 2).sum() / (len(rq) - 1)
    return varq / (q * var1)

def hurst_rs(x):
    """R/S 法 Hurst 指数（对数尺度回归）"""
    n = len(x)
    if n < 64:
        return np.nan
    ns = [16, 32, 64, 128]
    rs = []
    for w in ns:
        if w >= n:
            break
        k = n // w
        vals = []
        for i in range(k):
            seg = x[i*w:(i+1)*w]
            seg = seg - seg.mean()
            cum = np.cumsum(seg)
            Rr = cum.max() - cum.min()
            S = seg.std()
            if S > 0:
                vals.append(Rr / S)
        if vals:
            rs.append((w, np.mean(vals)))
    if len(rs) < 2:
        return np.nan
    xs = np.log([r[0] for r in rs]); ys = np.log([r[1] for r in rs])
    return float(np.polyfit(xs, ys, 1)[0])

def feats(c, h, l, q):
    """对一段序列算五路特征"""
    out = {}
    lg = np.log(np.maximum(c, 1e-12))
    r = np.diff(lg)
    # D1 滑窗 VR（窗口 96，步长 24）
    sub = []
    for s in range(0, max(len(lg) - 96, 0), 24):
        v = vr(lg[s:s+96], 24)
        if np.isfinite(v):
            sub.append(v)
    out['win_vr_med'] = float(np.median(sub)) if sub else np.nan
    out['win_vr_gt15'] = float(np.mean([v > 1.5 for v in sub])) if sub else np.nan
    # D2 条件 VR：只在量能异动根上
    qs = pd.Series(q)
    base = qs.rolling(288 * 30, min_periods=288).mean().values
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = q / np.maximum(base, 1e-12)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    mask = np.nan_to_num(volr6, nan=0.0) >= 5
    if mask.sum() >= 64:
        idx = np.flatnonzero(mask)
        # 只取连续段，避免拼接口径
        out['cond_vr'] = float(vr(lg[idx], 12)) if len(idx) > 60 else np.nan
    else:
        out['cond_vr'] = np.nan
    # D3 高阶矩
    if len(r) > 30:
        m = r.mean(); s = r.std()
        out['skew'] = float(((r - m) ** 3).mean() / max(s ** 3, 1e-12))
        out['kurt'] = float(((r - m) ** 4).mean() / max(s ** 4, 1e-12))
        out['tail3'] = float((r > m + 3 * s).mean())
    else:
        out['skew'] = out['kurt'] = out['tail3'] = np.nan
    out['amp_ratio'] = float(np.mean((h - l) / np.maximum(c, 1e-12)))
    # D4 成交额 Hurst
    out['hurst_vol'] = hurst_rs(np.log1p(q))
    # D5 CUSUM 最大偏移 / sqrt(n)
    if len(r) > 30:
        cs = np.cumsum(r - r.mean())
        out['cusum'] = float(np.max(np.abs(cs)) / np.sqrt(len(r)))
    else:
        out['cusum'] = np.nan
    return out

rows = []
rng = np.random.default_rng(7)
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < 600:
        continue
    i = 0; spans = []
    while i < n - 1:
        lo_idx = i; lo = l[i]; conf = -1; peak = -np.inf
        j = i; done = False
        while j < n - 1:
            j += 1
            if conf < 0:
                if l[j] < lo:
                    lo = l[j]; lo_idx = j
                if j - lo_idx > MAX_BARS:
                    i = lo_idx + 1; done = True; break
                if h[j] >= lo * (1 + TRIG):
                    conf = j; peak = h[j]
            else:
                if h[j] > peak:
                    peak = h[j]
                if j - conf > MAX_BARS or l[j] <= peak * (1 - R):
                    spans.append((lo_idx, j))
                    L = conf - lo_idx + 1
                    if L >= 160:
                        f = feats(c[lo_idx:conf+1], h[lo_idx:conf+1], l[lo_idx:conf+1], q[lo_idx:conf+1])
                        f.update(sym=sym, inside=1, L=L, gain=peak/lo-1)
                        rows.append(f)
                    i = j + 1; done = True; break
        if done:
            continue
        break
    # 局外同长度随机段
    if len(spans) >= 2:
        mask = np.zeros(n, bool)
        for a, b in spans:
            mask[a:b+1] = True
        free = np.flatnonzero(~mask)
        for a, b in spans[:6]:
            L = b - a + 1
            if L < 160 or len(free) < L + 10:
                continue
            st = int(rng.choice(free[:max(len(free)-L, 1)]))
            f = feats(c[st:st+L], h[st:st+L], l[st:st+L], q[st:st+L])
            f.update(sym=sym, inside=0, L=L, gain=np.nan)
            rows.append(f)
del RAW; gc.collect()

DF = pd.DataFrame(rows)
DF.to_csv(os.path.join(OUT, 'regime_detect.csv'), index=False, encoding='utf-8-sig')
log(f'样本：局内 {int(DF.inside.sum())}   局外 {int((1-DF.inside).sum())}   耗时 {time.time()-t0:.0f}s')

FS = ['win_vr_med', 'win_vr_gt15', 'cond_vr', 'skew', 'kurt', 'tail3', 'amp_ratio', 'hurst_vol', 'cusum']
log('\n' + '=' * 118)
log('【结果】局内 vs 局外（中位数），以及 Mann-Whitney 显著性')
log('=' * 118)
from scipy import stats
log(f'  {"特征":<14}{"局内":>12}{"局外":>12}{"比值":>9}{"p值":>12}{"判定":>8}')
verdict = {}
for f in FS:
    a = DF[DF.inside == 1][f].dropna(); b = DF[DF.inside == 0][f].dropna()
    if len(a) < 20 or len(b) < 20:
        continue
    ma, mb = a.median(), b.median()
    try:
        p = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
    except Exception:
        p = np.nan
    ratio = ma / mb if mb not in (0, np.nan) and np.isfinite(mb) else np.nan
    v = '★显著' if (np.isfinite(p) and p < 0.001) else ('显著' if np.isfinite(p) and p < 0.01 else '—')
    verdict[f] = p
    log(f'  {f:<14}{ma:>12.4f}{mb:>12.4f}{ratio:>9.2f}{p:>12.2e}{v:>8}')

log('\n' + '=' * 118)
log('【判读】')
log('=' * 118)
sig = [f for f, p in verdict.items() if np.isfinite(p) and p < 0.001]
log(f'  · 显著差异的特征：{sig if sig else "无"}')
log('  · 若有特征显著 ⇒ 「局」在某个维度上确实可检测，上一版 VR 是方法错，不是否定')
log('  · 若全部不显著 ⇒ 在这五种方式下依然看不到局（仍未穷尽）')

out = os.path.join(OUT, 'regime_detect_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
