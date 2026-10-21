# -*- coding: utf-8 -*-
"""gambler · 操盘者行为心理侧写（个人型 / 群体型）

用户指出：缺的是「对操盘者个人和群体的**行为概率心理侧写**」。
侧写不是猜心思 —— 每种动机在盘口都留下**可测量的行为**：

  动机           行为              可测量的量
  ──────────────────────────────────────────────────────────────────
  贪婪·目标位    涨到某个数就走    → 峰值涨幅是否**聚集在共同目标位**
                                       (+50% / +100% / 整数倍)
  急切·怕被抢    拉得越快越好      → 拉升速度 = 涨幅 ÷ 到峰值耗时
  耐心·建仓      低位悄悄吸        → 启动前 24h 的横盘度 + 量能（放量滞涨）
  恐惧·要跑      见顶就想出        → 峰值后 1h/2h/4h/8h 跌幅与量比
  博弈·囚徒困境  谁先跑            → 撤退是否分**波**（量能台阶）→ 第一波是谁
  从众·FOMO      别人涨我也买      → 启动后 1~2 根的量能追涨强度

并对照**个人型 vs 群体型**（群体 = 启动时全市场同步异动的币数多）。

输出 evidence/gambler_psych.txt（+ gambler_psych.png）
"""
import os, gc, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler · 操盘者行为心理侧写（把动机翻译成可测量的盘口行为）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
m = {}
for c in ('close', 'high', 'low', 'quote_volume'):
    m[c] = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
del df; gc.collect()
close, high, low, qv = m['close'], m['high'], m['low'], m['quote_volume']
close = close.where(close > 0).ffill(limit=3)
qv = qv.where(qv >= 0)
idx = close.index.values; T, N = close.shape
syms = np.array(list(close.columns)); pos = {s: i for i, s in enumerate(syms)}
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
med = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
med = np.where(med > 0, med, np.nan)
volr = QV / med
OK = np.isfinite(volr) & np.isfinite(ret1)
ret1[OK & (volr > 200)] = np.nan
OK = np.isfinite(volr) & np.isfinite(ret1)
M_UP = OK & (volr >= 2) & (ret1 >= 0.03)
n_up = M_UP.sum(axis=1).astype(np.int32)

ev = pd.read_csv(os.path.join(EV, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)
def _to_ms(s):
    v = pd.to_datetime(s).astype('int64').values.astype('int64')
    d = int(v[0])
    if d > 10**17:
        v = v // 10**6
    elif d > 10**14:
        v = v // 10**3
    elif d > 10**11:
        pass
    else:
        v = v * 1000
    return v
t0m = _to_ms(ev['start_time'])
t1m = _to_ms(ev['peak_time'])
p0 = np.searchsorted(idx, t0m); p1 = np.searchsorted(idx, t1m)
keep = (p0 > 300) & (p1 < T - 300) & (p0 < p1 + 1)
ev = ev[keep].reset_index(drop=True); p0 = p0[keep]; p1 = p1[keep]
say('事件 %d 个  %.0fs' % (len(ev), time.time() - t0))

rec = []
for i in range(len(ev)):
    s = ev['symbol'][i]; j = pos.get(s)
    if j is None: continue
    a, b = int(p0[i]), int(p1[i])
    gain = float(ev['gain'][i])
    if not np.isfinite(gain) or gain <= 0: continue
    mins = max(float(ev['min_to_peak'][i]), 1.0)
    # ① 目标位
    # ② 拉升速度（每小时涨幅）
    speed = gain / (mins / 12.0)
    # ③ 建仓期（启动前 24h = 288 根）
    vb = a - 288
    if vb > 20:
        volr_b = np.nanmean(volr[vb:a, j])
        rng_b = (np.nanmax(HV[vb:a, j]) - np.nanmin(LV[vb:a, j])) / max(np.nanmean(CV[vb:a, j]), 1e-12)
        ret_b = CV[a - 1, j] / CV[vb, j] - 1.0
    else:
        volr_b, rng_b, ret_b = np.nan, np.nan, np.nan
    # ④ 撤退节奏（峰值后 288 根 = 24h）
    e = min(b + 288, T - 1)
    seg_r, seg_v = ret1[b:e, j], volr[b:e, j]
    seg_c = CV[b:e, j]
    dd = seg_c.min() / HV[b, j] - 1.0 if len(seg_c) and np.isfinite(HV[b, j]) and HV[b, j] > 0 else np.nan
    waves = int(np.nansum((seg_r < 0) & (seg_v > 3)))          # 撤退波数（放量下跌根数）
    t_half = np.nan
    below = np.where(seg_c / HV[b, j] - 1.0 <= -0.5)[0] if np.isfinite(HV[b, j]) and HV[b, j] > 0 else np.array([])
    if len(below): t_half = below[0] * 5 / 60.0                # 跌破峰值-50% 用了多少小时
    r1h = CV[min(b + 12, T - 1), j] / HV[b, j] - 1.0 if np.isfinite(HV[b, j]) and HV[b, j] > 0 else np.nan
    # ⑤ 追涨强度（启动后 2 根量比）
    fol = np.nanmean(volr[max(a - 1, 0):a + 2, j])
    # ⑥ 群体度
    sync = np.nanmean(n_up[max(a - 2, 0):a + 3])
    rec.append((s, gain, mins, speed, volr_b, rng_b, ret_b, dd, waves, t_half, r1h, fol, sync,
                float(np.nansum(QV[:, j])), float(ev['thr'][i])))
R = pd.DataFrame(rec, columns=['symbol', 'gain', 'mins', 'speed', 'volr_b', 'rng_b', 'ret_b',
                               'dd24', 'waves', 't_half', 'r1h', 'follow', 'sync', 'qv_sum', 'thr'])
R.to_csv(os.path.join(EV, 'gambler_psych.csv'), index=False, encoding='utf-8-sig')
say('有效事件 %d  %.0fs' % (len(R), time.time() - t0))

# ---------- A 目标位聚集 ----------
say('\n' + '=' * 100)
say('【A 贪婪·目标位：峰值涨幅是否聚集在共同目标位】')
say('=' * 100)
g = R['gain'].values
bins = np.logspace(np.log10(0.5), np.log10(20), 26)
h, _ = np.histogram(np.clip(g, 0.5, 20), bins=bins)
mid = np.sqrt(bins[:-1] * bins[1:])
for k in np.argsort(-h)[:6]:
    say('  峰位 %.2f~%.2f 倍涨幅：%4d 事件（占 %.1f%%）' % (bins[k], bins[k + 1], h[k], 100 * h[k] / len(g)))
say('  涨幅中位 = %.2f 倍；均值 = %.2f 倍' % (np.median(g), g.mean()))
for lv in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    hit = ((g >= lv * 0.9) & (g <= lv * 1.1)).sum()
    say('   在 +%.0f%% 附近(±10%%)落点：%4d 事件（占 %4.1f%%）' % (lv * 100, hit, 100 * hit / len(g)))

# ---------- B 拉升速度 ----------
say('\n' + '=' * 100)
say('【B 急切·拉升速度：快庄/中/慢庄 及各自后续】')
say('=' * 100)
q = R['speed'].quantile([1 / 3, 2 / 3]).values
R['tempo'] = np.where(R['speed'] <= q[0], '慢庄', np.where(R['speed'] <= q[1], '中庄', '快庄'))
for k in ('慢庄', '中庄', '快庄'):
    s = R[R['tempo'] == k]
    say('  %-3s n=%4d  速度%6.2f倍/h  涨幅中位%5.2f倍  到峰值%5.0f分  24h回撤%6.1f%%  撤退出货波%5.1f  跌破-50%%耗时%5.1fh'
        % (k, len(s), s['speed'].median(), s['gain'].median(), s['mins'].median(),
           100 * s['dd24'].median(), s['waves'].median(), s['t_half'].median()))

# ---------- C 建仓期 ----------
say('\n' + '=' * 100)
say('【C 耐心·建仓：启动前 24h 有预谋（放量滞涨）vs 突发】')
say('=' * 100)
R['prep'] = np.where((R['volr_b'] >= 2.0) & (R['rng_b'] < 0.20), '有预谋', '突发')
for k in ('有预谋', '突发'):
    s = R[R['prep'] == k]
    if not len(s): continue
    say('  %-4s n=%4d（%.0f%%）  涨前量比%5.2f  涨前24h振幅%5.1f%%  涨前24h收益%+6.1f%%  涨幅中位%5.2f倍'
        % (k, len(s), 100 * len(s) / len(R), s['volr_b'].median(), 100 * s['rng_b'].median(),
           100 * s['ret_b'].median(), s['gain'].median()))

# ---------- D 撤退节奏 ----------
say('\n' + '=' * 100)
say('【D 恐惧·撤退节奏：撤退波数 vs 最终回撤（谁先跑）】')
say('=' * 100)
for lo_, hi_ in ((0, 2), (2, 5), (5, 999)):
    s = R[(R['waves'] >= lo_) & (R['waves'] < hi_)]
    if len(s) < 10: continue
    say('  出货波%-6s n=%4d  24h回撤%6.1f%%  峰值后1h%6.1f%%  跌破-50%%耗时%5.1fh  涨幅%5.2f倍'
        % ('%d~%s' % (lo_, hi_ if hi_ < 999 else '+'), len(s), 100 * s['dd24'].median(),
           100 * s['r1h'].median(), s['t_half'].median(), s['gain'].median()))
say('  peaks 后 1h 就已经≤-20%% 的事件占比：%.1f%%（这些人第一时间跑了）'
    % (100 * (R['r1h'] <= -0.20).mean()))

# ---------- E 个人 vs 群体 ----------
say('\n' + '=' * 100)
say('【E 个人型 vs 群体型：心理画像并列】')
say('=' * 100)
med_sync = R['sync'].median()
R['kind'] = np.where(R['sync'] >= med_sync, '群体', '个人')
say('  分组依据：启动时全市场同步异动币数（中位 %.2f）⇒ 群体 %d / 个人 %d'
    % (med_sync, int((R['kind'] == '群体').sum()), int((R['kind'] == '个人').sum())))
say('  %-4s %6s %8s %8s %8s %8s %8s %8s %8s'
    % ('类型', '事件', '涨幅中位', '速度', '到峰值min', '涨前量比', '出货波', '24h回撤', '跌破-50%h'))
for k in ('群体', '个人'):
    s = R[R['kind'] == k]
    say('  %-4s %6d %7.2f倍 %7.2f %8.0f %8.2f %8.1f %7.1f%% %8.1f'
        % (k, len(s), s['gain'].median(), s['speed'].median(), s['mins'].median(),
           s['volr_b'].median(), s['waves'].median(), 100 * s['dd24'].median(), s['t_half'].median()))

# ---------- 图 ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].hist(np.clip(g, 0.5, 20), bins=bins, color='#4472c4')
ax[0, 0].set_xscale('log'); ax[0, 0].set_title('A. Peak gain distribution (target levels)')
ax[0, 0].set_xlabel('peak multiple'); ax[0, 0].set_ylabel('events')
for k, c in (('慢庄', '#4472c4'), ('中庄', '#a5a5a5'), ('快庄', '#ed7d31')):
    s = R[R['tempo'] == k]
    ax[0, 1].scatter(s['speed'], np.clip(s['gain'], 0.5, 20), s=6, alpha=.4, label=k, color=c)
ax[0, 1].set_xscale('log'); ax[0, 1].set_yscale('log')
ax[0, 1].set_title('B. Pump speed vs peak gain'); ax[0, 1].legend()
ax[1, 0].scatter(R['waves'], np.clip(-R['dd24'], 0, 1), s=6, alpha=.4, color='#4472c4')
ax[1, 0].set_title('D. Dump waves vs 24h drawdown'); ax[1, 0].set_xlabel('dump waves'); ax[1, 0].set_ylabel('drawdown')
for k, c in (('群体', '#ed7d31'), ('个人', '#4472c4')):
    s = R[R['kind'] == k]
    ax[1, 1].scatter(s['mins'], s['speed'], s=6, alpha=.4, label=k, color=c)
ax[1, 1].set_xscale('log'); ax[1, 1].set_yscale('log')
ax[1, 1].set_title('E. Group vs solo: time-to-peak vs speed'); ax[1, 1].legend()
plt.tight_layout(); plt.savefig(os.path.join(EV, 'gambler_psych.png'), dpi=110); plt.close()

open(os.path.join(EV, 'gambler_psych.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
