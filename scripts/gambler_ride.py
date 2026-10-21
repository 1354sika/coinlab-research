# -*- coding: utf-8 -*-
"""gambler · 跟盘实验 v1（事件级投注，判据 = 期望净收益率）

用户口径（2026-10-16 修正）：
  · 要的是**期望净收益率**，不是涨跌概率
  · **不是**连续多笔小交易，而是：**发现操盘者 → 前期跟盘 → 提前撤出 → 单笔几倍收益**
  · 扣费本来就要算进去

做法（全部只用 t 时刻可见信息，无事件起点、无未来信息）：
  入场条件：过去 1h 涨幅 ret60 ≥ A  且  量比 volr ≥ B      → 按当根收盘价买入
  去重：同币相邻入场间隔 ≥ 288 根（24h），避免同一次行情重复计数
  撤出：**移动止损** —— 价格从入场后最高点回落 D% → 卖出；或持满 HOLD 根 → 卖出
  成本：maker 双边 0.04%
  统计：笔数、期望净收益率、中位、胜率、≥50% / ≥100% / ≥200% 的比例
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
FEE = 0.0004
HOLD = 288          # 最长持有 24h
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 跟盘实验：入场(过去1h涨幅≥A, 量比≥B) → 移动止损 D%% 撤出；判据=期望净收益率')
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
syms = list(close.columns)
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret60 = np.full((T, N), np.nan, np.float32); ret60[60:] = CV[60:] / CV[:-60] - 1.0
ret240 = np.full((T, N), np.nan, np.float32); ret240[240:] = CV[240:] / CV[:-240] - 1.0
qv60 = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)
volr = (QV / qv60).astype('float32')
say('指标算完  %.0fs' % (time.time() - t0))


def sim(entries, D):
    """entries: list of (t, j)；返回 (hold_bars, gross_ret)"""
    hs, rs = [], []
    for (t, j) in entries:
        end = min(T, t + HOLD + 1)
        segH = HV[t + 1:end, j]; segL = LV[t + 1:end, j]
        if len(segH) == 0:
            continue
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
say('【一】入场条件 × 止损宽度 → 期望净收益率（扣 maker 双边 0.04%%）')
say('=' * 100)
say('  %-6s %-5s %-6s %8s %11s %9s %8s %8s %8s' %
    ('A(1h涨)', 'B(量比)', '止损D', '笔数', '期望净%', '中位%', '胜率', '≥+50%', '≥+100%'))
results = {}
for A in (0.10, 0.20, 0.30):
    for B in (1.0, 3.0, 5.0):
        trig = np.isfinite(ret60) & np.isfinite(volr) & (ret60 >= A) & (volr >= B)
        entries = []
        for j in range(N):
            col = trig[:, j]
            if not col.any():
                continue
            idx = np.where(col)[0]
            last = -10 ** 9
            for t in idx:
                if t - last < HOLD:
                    continue
                entries.append((int(t), j)); last = int(t)
        if len(entries) < 30:
            continue
        for D in (0.15, 0.25, 0.35, 0.50):
            hs, rs = sim(entries, D)
            if len(rs) < 30:
                continue
            net = rs - FEE
            results[(A, B, D)] = (len(rs), net.mean(), np.median(net), (net > 0).mean(),
                                  (net >= 0.5).mean(), (net >= 1.0).mean(), hs.mean())
            say('  %-6.2f %-5.1f %-6.2f %8d %+10.3f%% %+8.2f%% %7.1f%% %7.1f%% %7.1f%%'
                % (A, B, D, len(rs), 100 * net.mean(), 100 * np.median(net),
                   100 * (net > 0).mean(), 100 * (net >= 0.5).mean(), 100 * (net >= 1.0).mean()))

say('\n' + '=' * 100)
say('【二】对照：无条件随机入场（同样持有 24h / 同样移动止损）')
say('=' * 100)
rng = np.random.default_rng(0)
rnd_e = [(int(rng.integers(300, T - HOLD - 1)), int(rng.integers(0, N))) for _ in range(30000)]
for D in (0.25, 0.50):
    hs, rs = sim(rnd_e, D)
    net = rs - FEE
    say('  止损 %.2f：笔数 %d  期望净 %+.3f%%  中位 %+.2f%%  胜率 %.1f%%  ≥+100%% 占 %.2f%%'
        % (D, len(rs), 100 * net.mean(), 100 * np.median(net), 100 * (net > 0).mean(),
           100 * (net >= 1.0).mean()))

if results:
    best = max(results.items(), key=lambda kv: kv[1][1])
    say('\n★ 期望净收益率最高的一组：入场(过去1h涨幅≥%.0f%%, 量比≥%.1f) + 止损 %.0f%%'
        % (100 * best[0][0], best[0][1], 100 * best[0][2]))
    say('   笔数 %d，期望净 %+.3f%%/笔，中位 %+.2f%%，胜率 %.1f%%，≥+100%% 占 %.1f%%'
        % (best[1][0], 100 * best[1][1], 100 * best[1][2], 100 * best[1][3], 100 * best[1][5]))

    # 图：各入场阈值下，期望净收益 vs 止损宽度
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    Ds = (0.15, 0.25, 0.35, 0.50)
    for (A, B) in [(0.10, 1.0), (0.20, 1.0), (0.30, 1.0), (0.20, 3.0), (0.20, 5.0)]:
        ys = [results.get((A, B, D), (0, np.nan))[1] * 100 if (A, B, D) in results else np.nan for D in Ds]
        ax[0].plot([100 * d for d in Ds], ys, 'o-', label='entry ret1h≥%.0f%% volr≥%.0f' % (100 * A, B))
    ax[0].axhline(0, ls='--', color='k', lw=1)
    ax[0].set_xlabel('trailing stop width (%)'); ax[0].set_ylabel('expected net return per trade (%)')
    ax[0].set_title('ride-the-pump: expected net return vs stop width')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # 右：收益分布（最优组）
    A, B, D = best[0]
    trig = np.isfinite(ret60) & np.isfinite(volr) & (ret60 >= A) & (volr >= B)
    ent = []
    for j in range(N):
        idx = np.where(trig[:, j])[0]; last = -10 ** 9
        for t in idx:
            if t - last < HOLD:
                continue
            ent.append((int(t), j)); last = int(t)
    hs, rs = sim(ent, D)
    net = (rs - FEE) * 100
    ax[1].hist(np.clip(net, -100, 300), bins=60)
    ax[1].axvline(0, ls='--', color='k', lw=1)
    ax[1].set_xlabel('net return per trade (%)'); ax[1].set_ylabel('trades')
    ax[1].set_title('return distribution (best rule, n=%d)' % len(net))
    ax[1].grid(alpha=.3, axis='y')
    plt.tight_layout()
    p = os.path.join(EV, 'gambler_ride.png')
    plt.savefig(p, dpi=125, bbox_inches='tight')
    say('saved: %s' % p)

open(os.path.join(EV, 'gambler_ride.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
