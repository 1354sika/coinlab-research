# -*- coding: utf-8 -*-
"""分化时刻内找「会大涨的币」：特征 -> P(未来5分钟涨幅>阈值) 与期望收益
样本 = 分钟 x 币（约 4300 万），限制在横截面离散度 disp > P90 的分化时刻。
回答：分化时，哪些特征的币更容易大涨？扣费后是否为正？
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV   = os.path.join(ROOT, 'evidence')
os.makedirs(EV, exist_ok=True)

FEE_TAKER = 0.0020
FEE_MAKER = 0.0004
TXT = []

def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=== 读 502 币分钟数据 ===')
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
say('rows=%d elapsed=%.1fs' % (len(df), time.time() - t0))

px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
say('matrix %d x %d  elapsed=%.1fs' % (px.shape[0], px.shape[1], time.time() - t0))
del df

r = px.pct_change()
mkt = r.median(axis=1)
disp = r.std(axis=1)
thr = float(disp.quantile(0.90))
say('disp: median=%.4f%%  P90=%.4f%%  P99=%.4f%%' % (
    disp.median() * 100, thr * 100, disp.quantile(0.99) * 100))

sel = disp > thr
say('分化时刻 %d / %d = %.1f%%' % (sel.sum(), len(sel), 100 * sel.mean()))
idx = sel[sel].index

# ---- 特征（先全时段算，再按分化行切片）----
feats = {
    'R_15m': px / px.shift(15) - 1.0,
    'R_60m': px / px.shift(60) - 1.0,
    'MA60dev': px / px.rolling(60).mean() - 1.0,
    'VolRatio': qv / qv.rolling(60).mean(),
    'R_5m': px / px.shift(5) - 1.0,
    'Rng15': (px.rolling(15).max() - px.rolling(15).min()) / px,
}
fwd5 = px.shift(-5) / px - 1.0

target = fwd5.loc[idx]
say('\n=== 目标：未来 5 分钟收益 ===')
ty = target.values.ravel(); ty = ty[np.isfinite(ty)]
say('  n=%d  P(>0.5%%)=%.3f%%  P(>1%%)=%.3f%%  P(>2%%)=%.3f%%  P(>5%%)=%.4f%%' % (
    len(ty), 100*(ty > .005).mean(), 100*(ty > .01).mean(),
    100*(ty > .02).mean(), 100*(ty > .05).mean()))
say('  均值=%.4f%%  中位=%.4f%%   往返成本: taker 0.20%% / maker 0.04%%' % (
    100*ty.mean(), 100*np.median(ty)))


def table(name, F, thr_hi=0.02, nbins=5):
    f = F.loc[idx].values.ravel(); y = target.values.ravel()
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    b2 = (y > thr_hi).mean(); b1 = (y > 0.01).mean(); bm = y.mean()
    say('\n--- %s ---' % name)
    say('  分化基准: P(>2%%)=%.4f%%  P(>1%%)=%.4f%%  均值=%.4f%%  n=%d' % (
        100*b2, 100*b1, 100*bm, len(y)))
    qs = np.quantile(f, np.linspace(0, 1, nbins + 1))
    for i in range(nbins):
        lo, hi = qs[i], qs[i + 1]
        m = (f >= lo) & (f <= hi) if i == nbins - 1 else (f >= lo) & (f < hi)
        if m.sum() < 500:
            continue
        yy = y[m]
        say('  [%7.2f%%..%7.2f%%] n=%7d  P>2%%=%.4f%%(%4.2fx)  P>1%%=%.4f%%(%4.2fx)  均值=%+.4f%%  净(taker)=%+.4f%%' % (
            lo*100, hi*100, m.sum(),
            100*(yy > thr_hi).mean(), (yy > thr_hi).mean()/max(b2, 1e-12),
            100*(yy > 0.01).mean(), (yy > 0.01).mean()/max(b1, 1e-12),
            100*yy.mean(), 100*(yy.mean() - FEE_TAKER)))
    return f, y, b2, b1


res = {}
for k, F in feats.items():
    res[k] = table(k, F)

# ---- 组合条件 ----
say('\n\n=== 组合条件（过去跌 + 放量）===')
F15 = feats['R_15m'].loc[idx].values.ravel()
FV = feats['VolRatio'].loc[idx].values.ravel()
y = target.values.ravel()
ok = np.isfinite(F15) & np.isfinite(FV) & np.isfinite(y)
F15, FV, y = F15[ok], FV[ok], y[ok]
vm = np.nanmedian(FV[np.isfinite(FV)])
combos = [
    ('R15<0 & 量比>中位', (F15 < 0) & (FV > vm)),
    ('R15<0 & 量比<中位', (F15 < 0) & (FV <= vm)),
    ('R15>0 & 量比>中位', (F15 > 0) & (FV > vm)),
    ('R15<-5% & 量比>2', (F15 < -0.05) & (FV > 2)),
    ('R15>+5% & 量比>2', (F15 > 0.05) & (FV > 2)),
    ('R15<-5% & 量比<2', (F15 < -0.05) & (FV <= 2)),
    ('R15>+5% & 量比<2', (F15 > 0.05) & (FV <= 2)),
]
for nm, m in combos:
    if m.sum() < 500:
        continue
    yy = y[m]
    say('  %-22s n=%8d  P>2%%=%.4f%%  P>1%%=%.4f%%  均值=%+.4f%%  净taker=%+.4f%%  净maker=%+.4f%%' % (
        nm, m.sum(), 100*(yy > .02).mean(), 100*(yy > .01).mean(),
        100*yy.mean(), 100*(yy.mean() - FEE_TAKER), 100*(yy.mean() - FEE_MAKER)))

# ---- 反向：大跌 ----
say('\n=== 同样条件，未来 5 分钟跌幅 > 2% 的概率（做空视角）===')
for nm, m in combos:
    if m.sum() < 500:
        continue
    yy = y[m]
    say('  %-22s P(<-2%%)=%.4f%%  P(<-1%%)=%.4f%%  均值=%+.4f%%' % (
        nm, 100*(yy < -.02).mean(), 100*(yy < -.01).mean(), 100*yy.mean()))

# ---- 出图 ----
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
axs = axs.ravel()
for i, (k, F) in enumerate(list(feats.items())[:6]):
    f, yv, b2, b1 = res[k]
    qs = np.quantile(f, np.linspace(0, 1, 6))
    xs, hs, ns = [], [], []
    for j in range(5):
        lo, hi = qs[j], qs[j + 1]
        m = (f >= lo) & (f <= hi) if j == 4 else (f >= lo) & (f < hi)
        if m.sum() < 500:
            continue
        xs.append((lo + hi) / 2 * 100)
        hs.append(100 * (yv[m] > 0.02).mean())
        ns.append(m.sum())
    ax = axs[i]
    ax.plot(xs, hs, 'o-')
    ax.axhline(100 * b2, ls='--', lw=1)
    ax.set_title('%s -> P(5min ret > 2%%)' % k)
    ax.set_xlabel(k)
    ax.set_ylabel('P(%)')
    for x, h, n in zip(xs, hs, ns):
        ax.annotate('n=%d' % n, (x, h), fontsize=7)
axs[5].axis('off')
plt.tight_layout()
p = os.path.join(EV, 'bigwin.png')
plt.savefig(p, dpi=125)
say('\nsaved: %s' % p)

with open(os.path.join(EV, 'bigwin.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('TOTAL %.1fs' % (time.time() - t0))
