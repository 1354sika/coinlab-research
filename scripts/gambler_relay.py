# -*- coding: utf-8 -*-
"""gambler · 接力检验（连续接力 / 间隔接力）

用户提出：协同除了「同时异动」，还有**接力** —— 一个币拉完，另一个接着拉。
且要分两类：
  连续接力：B 在 A **见顶后 ≤1h** 内启动（几乎无缝衔接）
  间隔接力：B 在 A **见顶后 1h~24h** 才启动（中间有间隔，可能先"蓄势"）

做法（只用事件表 + 零模型，无前视）：
  1 接力对：A 见顶后 H 分钟内，另一个币启动
  2 零模型：**保持每个事件的"拉升时长"不变，把启动时间在总跨度内均匀随机重排 200 次**
     → 实际 vs 随机。实际显著更高 ⇒ 接力是真的（不是事件密集造成的巧合）
  3 间隔分布：A 见顶 → B 启动 的间隔（连续 / 间隔 / 长间隔）
  4 链：以 H 为连接阈值串链 → 链长、链内位置与涨幅（**越靠后是否越差**）
  5 接力 vs 同时：同刻启动占比
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler · 接力检验：A 拉完 → B 接着拉（连续接力 vs 间隔接力）')
say('=' * 100)

ev = pd.read_csv(os.path.join(EV, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)
ev['st'] = pd.to_datetime(ev['start_time'])
ev['pk'] = pd.to_datetime(ev['peak_time'])
ev = ev.sort_values('st').reset_index(drop=True)
ts = ev['st'].values.astype('datetime64[m]').astype('int64')     # 启动（分钟）
tp = ev['pk'].values.astype('datetime64[m]').astype('int64')     # 见顶（分钟）
dur = np.maximum(tp - ts, 1)
n = len(ev)
span_d = (ts[-1] - ts[0]) / 1440.0
say('事件 %d 个（覆盖 %d 币）；跨度 %.0f 天；平均每天 %.2f 个事件'
    % (n, ev['symbol'].nunique(), span_d, n / max(span_d, 1e-9)))
say('拉升时长（启动→见顶）中位 %.0f 分钟' % np.median(dur))

def relay_count(ts, tp, H):
    """A 见顶后 H 分钟内、由别的币启动的次数"""
    c = 0
    for i in range(len(ts)):
        lo = np.searchsorted(ts, tp[i], side='left')
        hi = np.searchsorted(ts, tp[i] + H, side='right')
        c += max(0, hi - max(lo, i + 1))
    return c

def overlap_count(ts, tp):
    """叠盖接力：B 在 A **启动之后、见顶之前**就启动（A 还没拉完 B 就上了）"""
    c = 0
    for i in range(len(ts)):
        lo = np.searchsorted(ts, ts[i], side='right')
        hi = np.searchsorted(ts, tp[i], side='right')
        c += max(0, hi - lo)
    return c

say('\n' + '=' * 100)
say('【一】接力对数：A 见顶后 H 内另一个币启动（实际 vs 零模型）')
say('=' * 100)
HIST = (30, 60, 120, 240, 480, 1440, 2880)
rng = np.random.default_rng(0)
for H in HIST:
    real = relay_count(ts, tp, H)
    vals = []
    for _ in range(200):
        ts_n = np.sort(rng.integers(ts[0], ts[-1] + 1, n))
        vals.append(relay_count(ts_n, ts_n + dur, H))
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    z = (real - mu) / max(sd, 1e-9)
    say('  H=%5d分(%4.1fh)：实际 %4d 对 ｜ 随机 %.1f±%.1f ｜ %5.2f× ｜ z=%+.1f'
        % (H, H / 60, real, mu, sd, real / max(mu, 1e-9), z))

ov_real = overlap_count(ts, tp)
ov_null = []
for _ in range(200):
    ts_n = np.sort(rng.integers(ts[0], ts[-1] + 1, n))
    ov_null.append(overlap_count(ts_n, ts_n + dur))
mu2, sd2 = float(np.mean(ov_null)), float(np.std(ov_null))
say('  ★ 叠盖接力（A 还没拉完、B 就已经启动）：实际 %d 对 ｜ 随机 %.1f±%.1f ｜ %.2f× ｜ z=%+.1f'
    % (ov_real, mu2, sd2, ov_real / max(mu2, 1e-9), (ov_real - mu2) / max(sd2, 1e-9)))

# ---------- 二 间隔分布 ----------
say('\n' + '=' * 100)
say('【二】间隔分布：A 见顶 → B 启动 的间隔（连续 / 间隔）')
say('=' * 100)
gaps = []
for i in range(n):
    lo = np.searchsorted(ts, ts[i], side='right')
    if lo < n:
        gaps.append(int(ts[lo] - tp[i]))        # 可为负 = 叠盖
    else:
        gaps.append(0)
gaps = np.array(gaps, float)
say('  可配对的 A 共 %d 个；间隔中位 %.0f 分钟' % (len(gaps), np.median(gaps)))
for nm, lo_, hi_ in (('★叠盖（A未完成B已启动）', -10**9, 0), ('同刻/即刻 (≤15min)', 0, 15),
                     ('连续接力 (15min~1h)', 15, 60),
                     ('短间隔 (1~4h)', 60, 240), ('中间隔 (4~12h)', 240, 720),
                     ('长间隔 (12~24h)', 720, 1440), ('超长 (>24h)', 1440, 10**9)):
    m = (gaps > lo_) & (gaps <= hi_)
    say('    %-24s %4d 次（%.1f%%）' % (nm, int(m.sum()), 100 * m.mean()))

# ---------- 三 链 ----------
say('\n' + '=' * 100)
say('【三】接力链：以 H 为连接阈值串链（链长 / 链内位置与涨幅）')
say('=' * 100)
for H in (60, 240, 1440):
    chains, cur = [], [0]
    for i in range(1, n):
        if ts[i] - tp[cur[-1]] <= H:
            cur.append(i)
        else:
            chains.append(cur); cur = [i]
    chains.append(cur)
    lens = np.array([len(c) for c in chains])
    say('  H=%4d分：链数 %4d ｜ 链长≥2 占 %5.1f%% ｜ 最长链 %d'
        % (H, len(chains), 100 * (lens >= 2).mean(), lens.max()))
    big = [c for c in chains if len(c) >= 4]
    if big:
        say('     链长≥4 的链 %d 条；例：%s'
            % (len(big), ' → '.join(ev['symbol'][j].replace('USDT', '') for j in big[0][:8])))
        pos_gain = {}
        for c in big:
            for k, j in enumerate(c):
                pos_gain.setdefault(k, []).append(ev['gain'][j])
        say('     链内位置 → 涨幅中位：' + ' ｜ '.join(
            'P%d=%.1f倍' % (k + 1, float(np.median(v))) for k, v in sorted(pos_gain.items()) if k < 8))

# ---------- 四 接力 vs 同时 ----------
say('\n' + '=' * 100)
say('【四】接力 vs 同时：启动时刻完全相同的事件对占比')
say('=' * 100)
same = 0
for v in pd.Series(ts).value_counts().values:
    same += int(v * (v - 1) / 2)
tot = relay_count(ts, tp, 1440)
say('  同刻启动对 = %d ｜ 24h 内接力对 = %d ｜ 同时/接力 = %.2f'
    % (same, tot, same / max(tot, 1e-9)))
say('  ⇒ 若"同时"远多于"接力"，说明协同以**并发**为主；反之则以**轮动**为主')

# ---------- 五 跨币对 ----------
say('\n' + '=' * 100)
say('【五】接力是否集中在少数"链条币"上') 
say('=' * 100)
pairs = []
for i in range(n):
    lo = np.searchsorted(ts, tp[i], side='left')
    hi = np.searchsorted(ts, tp[i] + 1440, side='right')
    for j in range(max(lo, i + 1), hi):
        pairs.append((ev['symbol'][i], ev['symbol'][j], int(ts[j] - tp[i])))
P = pd.DataFrame(pairs, columns=['A', 'B', 'gap'])
if len(P):
    say('  接力对总数（24h 内）= %d' % len(P))
    say('  A 端最常出现的币：' + ', '.join('%s(%d)' % (k.replace('USDT', ''), v)
        for k, v in P['A'].value_counts().head(8).items()))
    say('  B 端最常出现的币：' + ', '.join('%s(%d)' % (k.replace('USDT', ''), v)
        for k, v in P['B'].value_counts().head(8).items()))
    both = set(P['A']) & set(P['B'])
    say('  既当 A 又当 B 的币 %d 个' % len(both))
    P.to_csv(os.path.join(EV, 'gambler_relay_pairs.csv'), index=False, encoding='utf-8-sig')

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(np.clip(gaps, -60, 2880), bins=60, color='#4472c4')
ax[0].set_title('Relay gap: A peak -> B start (min)'); ax[0].set_xlabel('minutes')
xs = list(HIST); ys = [relay_count(ts, tp, H) for H in HIST]
ax[1].plot(xs, ys, 'o-', color='#ed7d31'); ax[1].set_xscale('log')
ax[1].set_title('Relay pairs vs horizon H'); ax[1].set_xlabel('H (min)')
plt.tight_layout(); plt.savefig(os.path.join(EV, 'gambler_relay.png'), dpi=110); plt.close()

open(os.path.join(EV, 'gambler_relay.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
