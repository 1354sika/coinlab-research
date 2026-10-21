# -*- coding: utf-8 -*-
"""gambler · 第三者（跟风盘 / 散户）行为心理侧写

三方结构：**操盘者（庄）／第三者（跟风者）／我们**。
第三者既不是敌人也不是朋友 —— 他们是**接力棒的下一个持有者**：
  庄要出货，必须有人接盘；我们要跑，必须赶在第三者恐慌之前。

诚实边界：**没有账户/持仓数据**，第三者只能从「成交量 + 价格」反推，无法直接看到谁在买。

从 K 线可测的第三者行为（无前视）：

  A 追涨延迟：启动后多少根出现量能高峰 → 跟风者多久上钩
  B 顶部换手：峰值 ±6 根的成交量 ÷ 整段成交量 → 庄家在这出了多少货 / 谁在接盘
  C **恐慌阈值**：从峰值跌到多少 % 时成交量再次爆发 → 第三者的止损参考线（踩踏点预判）
  D 站岗比例：峰值后 24h 有多少根仍在"高位"（未割肉）
  E 接盘枯竭：跌破 −50% 后的量能萎缩程度 → 没人接了（底部区域）
  F 与规模的关系：局越大，第三者越多、接盘越久？
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
say('gambler · 第三者（跟风盘）行为侧写：庄 / 跟风者 / 我们 三方')
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
keep = (p0 > 300) & (p1 < T - 300)
ev = ev[keep].reset_index(drop=True); p0 = p0[keep]; p1 = p1[keep]
say('事件 %d 个  %.0fs' % (len(ev), time.time() - t0))

rec = []
for i in range(len(ev)):
    s = ev['symbol'][i]; j = pos.get(s)
    if j is None: continue
    a, b = int(p0[i]), int(p1[i])
    gain = float(ev['gain'][i])
    ph = float(HV[b, j])
    if not (np.isfinite(gain) and gain > 0 and np.isfinite(ph) and ph > 0): continue
    # A 追涨延迟：启动后 48 根内量能峰值位置
    w = volr[a:min(a + 48, T), j]
    lead = int(np.nanargmax(w)) if np.isfinite(w).any() and len(w) else -1
    peak_volr = float(np.nanmax(w)) if len(w) else np.nan
    # B 顶部换手
    tot = float(np.nansum(QV[a:min(b + 288, T), j]))
    top_share = float(np.nansum(QV[max(b - 6, 0):min(b + 7, T), j])) / tot if tot > 0 else np.nan
    # C 恐慌阈值（peak 后第一根放量下跌时的跌幅）
    e = min(b + 288, T)
    seg_r, seg_v, seg_c = ret1[b:e, j], volr[b:e, j], CV[b:e, j]
    tk = np.where((seg_v > 5) & (seg_r < 0))[0]
    panic = float(seg_c[tk[0]] / ph - 1.0) if len(tk) else np.nan
    # D 站岗比例（peak 后 24h 仍在 peak*0.7 以上的比例）
    stand = float(np.nanmean(seg_c > ph * 0.7)) if len(seg_c) else np.nan
    # E 接盘枯竭
    b50 = np.where(seg_c / ph - 1.0 <= -0.5)[0]
    if len(b50) and b50[0] > 3:
        vol_after = float(np.nanmean(seg_v[b50[0]:]))
        vol_before = float(np.nanmean(seg_v[:b50[0]]))
        t_b50 = float(b50[0] * 5 / 60.0)
    else:
        vol_after = vol_before = t_b50 = np.nan
    rec.append((s, gain, lead, peak_volr, top_share, panic, stand, vol_after, vol_before, t_b50,
                float(np.nansum(QV[:, j])), float(ev['thr'][i])))
R = pd.DataFrame(rec, columns=['symbol', 'gain', 'lead', 'peak_volr', 'top_share', 'panic',
                               'stand', 'v_after', 'v_before', 't_b50', 'qv_sum', 'thr'])
R.to_csv(os.path.join(EV, 'gambler_third.csv'), index=False, encoding='utf-8-sig')
say('有效事件 %d  %.0fs' % (len(R), time.time() - t0))

# ---- A ----
say('\n' + '=' * 100)
say('【A 跟风者上钩速度：启动后量能高峰出现在第几根】')
say('=' * 100)
say('  追涨延迟中位 = %.0f 根（%.1f 小时）；均值 %.1f 根' % (R['lead'].median(), R['lead'].median() * 5 / 60, R['lead'].mean()))
say('  延迟分布：≤1h %.1f%% ｜ 1~2h %.1f%% ｜ 2~4h %.1f%% ｜ >4h %.1f%%'
    % (100 * (R['lead'] <= 12).mean(), 100 * ((R['lead'] > 12) & (R['lead'] <= 24)).mean(),
       100 * ((R['lead'] > 24) & (R['lead'] <= 48)).mean(), 100 * (R['lead'] > 48).mean()))
say('  启动后 48 根内量能峰值（量比）中位 = %.1f×' % R['peak_volr'].median())

# ---- B ----
say('\n' + '=' * 100)
say('【B 顶部换手：峰值 ±6 根（1 小时）吃掉整段成交量的多少】')
say('=' * 100)
say('  顶部换手占比中位 = %.1f%%；均值 %.1f%%'
    % (100 * R['top_share'].median(), 100 * R['top_share'].mean()))
say('  与涨幅的相关 corr(gain, top_share) = %+.4f' % R['gain'].corr(R['top_share']))
say('  ⇒ 若这 1 小时换手 >20%%，说明**庄家就在这一小时大量出货**（也是跟风者集中接盘的时刻）')
def _pc(v):
    v = float(v)
    return 'nan' if v != v else '%.1f%%' % (100 * v)

big = R[R['top_share'] >= 0.20]; small = R[R['top_share'] < 0.20]
if len(big) and len(small):
    say('    高换手组 n=%d：涨幅中位 %.2f 倍，24h 站岗比例 %s'
        % (len(big), big['gain'].median(), _pc(big['stand'].median())))
    say('    低换手组 n=%d：涨幅中位 %.2f 倍，24h 站岗比例 %s'
        % (len(small), small['gain'].median(), _pc(small['stand'].median())))

# ---- C ----
say('\n' + '=' * 100)
say('【C 恐慌阈值：从峰值跌到多少 % 时成交量再次爆发（第三者的止损线）】')
say('=' * 100)
p = R['panic'].dropna()
if not len(p):
    say('  （本批事件里没有「放量下跌」样本，跳过 C 小节）')
else:
    say('  首次放量下跌（量比>5 且收跌）发生在峰后跌幅中位 = %s；均值 %s' % (_pc(p.median()), _pc(p.mean())))
    for lo_, hi_ in ((-1, -0.5), (-0.5, -0.3), (-0.3, -0.15), (-0.15, 0)):
        n = int(((p >= lo_) & (p < hi_)).sum())
        say('    跌幅 %5.0f%% ~ %4.0f%% 触发：%4d 次（%.1f%%）' % (100 * lo_, 100 * hi_, n, 100 * n / len(p)))
say('  ⇒ 这可以当"**踩踏点**"参考：跌破该幅度常触发第二波抛售')

# ---- D/E ----
say('\n' + '=' * 100)
say('【D 站岗比例：峰值后 24h 仍留在"高位"（peak×0.7 以上）的时间占比】')
say('=' * 100)
say('  站岗比例中位 = %.1f%%；均值 %.1f%%' % (100 * R['stand'].median(), 100 * R['stand'].mean()))
say('  ⇒ 越高说明**接盘者被套住越久**（庄家出货慢、需要更长派发期）')

say('\n' + '=' * 100)
say('【E 接盘枯竭：跌破峰值 −50% 之后的量能 vs 之前】')
say('=' * 100)
q = R.dropna(subset=['v_after', 'v_before'])
say('  样本 %d；跌破 −50%% 耗时中位 = %.1f 小时' % (len(q), q['t_b50'].median()))
say('  跌破后量能中位 = %.1f× ｜ 跌破前量能中位 = %.1f× ⇒ 枯竭比 = %.2f'
    % (q['v_after'].median(), q['v_before'].median(), q['v_after'].median() / max(q['v_before'].median(), 1e-9)))
say('  ⇒ 枯竭比 <1 越明显，说明砸下去后**没人接**（价格会继续阴跌，别抄底）')

# ---- F ----
say('\n' + '=' * 100)
say('【F 局越大，第三人越多越久？按涨幅三档看】')
say('=' * 100)
R['sc'] = pd.cut(R['gain'], [0, 1.0, 2.0, 999], labels=['1倍内', '1~2倍', '2倍以上'])
for k in ('1倍内', '1~2倍', '2倍以上'):
    s = R[R['sc'] == k]
    if len(s) < 5: continue
    say('  %-6s n=%4d  追涨延迟%4.0f根  顶部换手%5.1f%%  恐慌阈值%+6.1f%%  站岗%5.1f%%  枯竭比%5.2f'
        % (k, len(s), s['lead'].median(), 100 * s['top_share'].median(),
           100 * s['panic'].median(), 100 * s['stand'].median(),
           s['v_after'].median() / max(s['v_before'].median(), 1e-9)))

# ---- 图 ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(R['lead'].dropna(), bins=40, color='#4472c4'); ax[0].set_title('A. Follower lag (bars after start)')
ax[1].hist(100 * R['panic'].dropna(), bins=40, color='#ed7d31'); ax[1].set_title('C. Panic threshold (% below peak)')
ax[2].hist(100 * R['top_share'].dropna(), bins=40, color='#a5a5a5'); ax[2].set_title('B. Top-hour turnover share (%)')
plt.tight_layout(); plt.savefig(os.path.join(EV, 'gambler_third.png'), dpi=110); plt.close()

open(os.path.join(EV, 'gambler_third.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
