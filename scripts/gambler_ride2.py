# -*- coding: utf-8 -*-
"""gambler · 跟盘实验 v2 —— 修正入场信号（对齐报告里唯一通过交叉验证的稳健结论）

为什么重做：
  ride v1 用「过去 1h 涨幅 ≥A」当入场信号 → 36 组全负（-0.55% ~ -4.8%/笔）。
  原因：1h 涨 10%~30% 里混了大量**假突破**，不等于"操盘手的局"。
  报告里唯一通过两版交叉验证的稳健结论是：
      **「4h 内涨 ≥50%」的事件发生后 0~6h，做多期望 +6%~+9%**
  而这个入场条件是**完全可见的**（只看过去 4h 涨幅，不用任何未来信息）。

本版（严格无前视）：
  入场：过去 4h 涨幅 ret48 ≥ A（A = 30% / 50% / 100%）且（可选）量比 ≥ B
  去重：同币相邻入场间隔 ≥ HOLD 根
  持有：最多 HOLD = 72 根（6h）
  撤出：移动止损（从入场后最高点回落 D%）或持满 6h
  成本：maker 双边 0.04%
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
FEE = 0.0004
HOLD = 72           # 6h
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 跟盘 v2：入场=过去 4h 涨幅≥A（可见）→ 持有≤6h + 移动止损；判据=期望净收益率')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
def _piv(c):
    m = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
    return m.where(m > 0).ffill(limit=3)
close, high, low = _piv('close'), _piv('high'), _piv('low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
qv = qv.where(qv >= 0)
del df; gc.collect()
T, N = close.shape
tm = close.index.values
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

for w in (48, 96):
    a = np.full((T, N), np.nan, np.float32); a[w:] = CV[w:] / CV[:-w] - 1.0
    globals()['ret%d' % w] = a
qv60 = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)
volr = (QV / qv60).astype('float32')
say('指标算完  %.0fs' % (time.time() - t0))


def collect(A, B, gate):
    g = (ret48 >= A) & np.isfinite(ret48)
    if B is not None:
        g = g & np.isfinite(volr) & (volr >= B)
    if gate is not None:
        g = g & gate
    ent = []
    for j in range(N):
        idx = np.where(g[:, j])[0]
        if len(idx) == 0:
            continue
        last = -10 ** 9
        for t in idx:
            if t - last < HOLD:
                continue
            ent.append((int(t), j)); last = int(t)
    return ent


def sim(entries, D):
    hs, rs = [], []
    for (t, j) in entries:
        end = min(T, t + HOLD + 1)
        segH = HV[t + 1:end, j]; segL = LV[t + 1:end, j]
        if len(segH) == 0:
            continue
        if D is None:
            u = len(segH) - 1; px = CV[t + u + 1, j]
        else:
            peak = np.maximum.accumulate(np.concatenate([[HV[t, j]], segH]))[1:]
            stop = peak * (1.0 - D)
            hit = np.where(np.isfinite(segL) & (segL <= stop))[0]
            if len(hit):
                u = int(hit[0]); px = stop[u]
            else:
                u = len(segH) - 1; px = CV[t + u + 1, j]
        if not np.isfinite(px) or px <= 0:
            continue
        hs.append(u + 1); rs.append(float(px / CV[t, j] - 1.0))
    return np.array(hs), np.array(rs)


say('\n' + '=' * 100)
say('【一】入场=过去 4h 涨幅≥A（可见）→ 持有≤6h；判据=期望净收益率（扣 maker 0.04%%）')
say('=' * 100)
say('  %-9s %-7s %-8s %7s %11s %9s %8s %8s %8s' %
    ('A(4h涨)', '量比B', '止损D', '笔数', '期望净%', '中位%', '胜率', '≥+50%', '≥+100%'))
res = {}
for A in (0.30, 0.50, 1.00):
    for B in (None, 3.0):
        ent = collect(A, B, None)
        for D in (None, 0.15, 0.25):
            hs, rs = sim(ent, D)
            if len(rs) < 20:
                continue
            net = rs - FEE
            res[(A, B, D)] = (len(rs), net.mean(), np.median(net), (net > 0).mean(),
                              (net >= 0.5).mean(), (net >= 1.0).mean(), hs.mean())
            say('  %-9.0f%% %-7s %-8s %7d %+10.3f%% %+8.2f%% %7.1f%% %7.1f%% %7.1f%%'
                % (100 * A, 'any' if B is None else '%.0f' % B, 'none' if D is None else '%.0f%%' % (100 * D),
                   len(rs), 100 * net.mean(), 100 * np.median(net),
                   100 * (net > 0).mean(), 100 * (net >= 0.5).mean(), 100 * (net >= 1.0).mean()))

say('\n' + '=' * 100)
say('【二】对照：随机时刻入场（同币同持有 6h + 同止损）')
say('=' * 100)
rng = np.random.default_rng(0)
rnd = [(int(rng.integers(300, T - HOLD - 1)), int(rng.integers(0, N))) for _ in range(30000)]
for D in (None, 0.25):
    hs, rs = sim(rnd, D)
    net = rs - FEE
    net = net[np.isfinite(net)]
    say('  止损 %-5s：笔数 %d  期望净 %+.3f%%  中位 %+.2f%%  胜率 %.1f%%  ≥+100%% %.2f%%'
        % ('none' if D is None else '%.0f%%' % (100 * D), len(net), 100 * net.mean(),
           100 * np.median(net), 100 * (net > 0).mean(), 100 * (net >= 1.0).mean()))

if res:
    pos = [k for k, v in res.items() if v[1] > 0]
    say('\n正期望组合数 = %d / %d' % (len(pos), len(res)))
    best = max(res.items(), key=lambda kv: kv[1][1])
    say('★ 期望净最高：入场 4h涨≥%.0f%% / 量比 %s / 止损 %s → 笔数 %d，期望净 %+.3f%%/笔，中位 %+.2f%%，胜率 %.1f%%，≥+100%% 占 %.1f%%'
        % (100 * best[0][0], 'any' if best[0][1] is None else '%.0f' % best[0][1],
           'none' if best[0][2] is None else '%.0f%%' % (100 * best[0][2]),
           best[1][0], 100 * best[1][1], 100 * best[1][2], 100 * best[1][3], 100 * best[1][5]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    labels, vals = [], []
    for (A, B, D), v in sorted(res.items()):
        labels.append('4h≥%.0f%% volr%s stop%s' % (100 * A, 'any' if B is None else '%.0f' % B,
                                                   'none' if D is None else '%.0f' % (100 * D)))
        vals.append(100 * v[1])
    ax[0].barh(range(len(vals)), vals, color='C0')
    ax[0].axvline(0, color='k', lw=1)
    ax[0].set_yticks(range(len(vals))); ax[0].set_yticklabels(labels, fontsize=7)
    ax[0].set_xlabel('expected net return per trade (%)')
    ax[0].set_title('ride v2: entry = past-4h gain >= A (visible)')
    ax[0].grid(alpha=.3, axis='x')
    A, B, D = best[0]
    ent = collect(A, B, None)
    hs, rs = sim(ent, D)
    net = (rs - FEE) * 100
    net = np.clip(net, -100, 400)
    ax[1].hist(net, bins=50)
    ax[1].axvline(0, ls='--', color='k', lw=1)
    ax[1].set_xlabel('net return per trade (%)'); ax[1].set_ylabel('trades')
    ax[1].set_title('return distribution (best rule, n=%d)' % len(net))
    ax[1].grid(alpha=.3, axis='y')
    plt.tight_layout()
    p = os.path.join(EV, 'gambler_ride2.png')
    plt.savefig(p, dpi=125, bbox_inches='tight')
    say('saved: %s' % p)

open(os.path.join(EV, 'gambler_ride2.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
