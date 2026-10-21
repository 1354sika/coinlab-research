# -*- coding: utf-8 -*-
"""_probe_third_fast.py — 抄近路版「参与者（第三者）行为侧写」
用 60 币 5m 数据（1.85GB）而非 10GB 全量；只读必要列。
测四件事（都是「参与者」而不是操盘者的行为）：
  A 跟风者上钩速度   事件启动后成交量峰值出现在第几根
  B 顶部换手         峰值 ±1h 吃掉整段拉升多少成交量（谁在接盘）
  C 恐慌阈值         峰值后首次「放量下跌」发生在跌多少 % 时（止损线）
  D 站岗比例         在峰值买进，之后 24h 还亏钱的比例
"""
import os, gc, time
import numpy as np
import pandas as pd

D = r'D:\中信证券\coinlab\data'; E = r'D:\中信证券\coinlab\evidence'
t0 = time.time()

df = pd.read_csv(os.path.join(D, 'klines_5m_3y.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'])
df = df.sort_values(['symbol', 'open_time'])
m = {}
for c in ('close', 'high', 'low', 'quote_volume'):
    m[c] = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
del df; gc.collect()
CV = m['close'].values; HV = m['high'].values; LV = m['low'].values; QV = m['quote_volume'].values
idx = m['close'].index.values
T, N = CV.shape
syms = np.array(list(m['close'].columns)); pos = {s: i for i, s in enumerate(syms)}
print('matrix %d x %d   %.0fs' % (T, N, time.time() - t0), flush=True)

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
med = pd.DataFrame(QV).rolling(96, min_periods=20).median().values        # 8h 基准
med = np.where(med > 0, med, np.nan)
volr = QV / med
print('volr 中位 %.2f' % np.nanmedian(volr), flush=True)

def _to_ms(s):
    v = pd.to_datetime(s).astype('int64').values.astype('int64')
    d = int(v[0])
    if d > 10**17: v = v // 10**6
    elif d > 10**14: v = v // 10**3
    elif d > 10**11: pass
    else: v = v * 1000
    return v

ev = pd.read_csv(os.path.join(E, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)
t0m = _to_ms(ev['start_time']); t1m = _to_ms(ev['peak_time'])
p0 = np.searchsorted(idx, t0m); p1 = np.searchsorted(idx, t1m)
keep = (p0 > 300) & (p1 < T - 300) & (p0 < p1 + 1)
ev = ev[keep].reset_index(drop=True); p0 = p0[keep]; p1 = p1[keep]
print('事件表 %d → 在本 60 币数据集里可对齐 %d 个   %.0fs' % (len(t1m), len(ev), time.time() - t0), flush=True)

rec = []
for i in range(len(ev)):
    s = ev['symbol'][i]; j = pos.get(s)
    if j is None: continue
    a, b = int(p0[i]), int(p1[i])
    if b <= a: continue
    seg = QV[a:b, j]
    if not np.isfinite(seg).any() or seg.sum() <= 0: continue
    # A 跟风上钩速度：拉升段内成交量峰值在第几根
    lead = int(np.nanargmax(seg))
    # B 顶部换手：峰值 ±6 根(1h) 的成交量 / 整段
    lo_, hi_ = max(a, b - 6), min(T - 1, b + 6)
    top_share = float(np.nansum(QV[lo_:hi_ + 1, j]) / max(seg.sum(), 1e-12))
    # C 恐慌阈值：峰后首次 量比>5 且收跌 的跌幅
    e = min(b + 288, T - 1)
    panic = np.nan
    for k in range(b + 1, e):
        if np.isfinite(volr[k, j]) and volr[k, j] > 5 and np.isfinite(ret1[k, j]) and ret1[k, j] < 0:
            panic = CV[k, j] / HV[b, j] - 1.0 if np.isfinite(HV[b, j]) and HV[b, j] > 0 else np.nan
            break
    # D 站岗：峰后 24h 内是否曾亏
    seg_c = CV[b:e, j]
    pk = HV[b, j]
    stand = float(np.nanmean(seg_c < pk)) if np.isfinite(pk) and pk > 0 and len(seg_c) else np.nan
    rec.append((s, float(ev['gain'][i]), float(ev['min_to_peak'][i]), lead, lead * 5 / 60.0,
                top_share, panic, stand))

R = pd.DataFrame(rec, columns=['symbol', 'gain', 'mins', 'lead', 'lead_h', 'top_share', 'panic', 'stand'])
R.to_csv(os.path.join(E, 'gambler_third_fast.csv'), index=False, encoding='utf-8-sig')
print('有效事件 %d   %.0fs' % (len(R), time.time() - t0), flush=True)

def pc(v):
    v = float(v)
    return 'nan' if v != v else '%.1f%%' % (100 * v)

print('\n' + '=' * 90)
print('【A 跟风者上钩速度】启动后成交量峰值出现在第几根（=参与者被钓上来的延迟）')
print('=' * 90)
print('  延迟中位 %.1f 小时（%.0f 根）  均值 %.1f 小时' % (R['lead_h'].median(), R['lead'].median(), R['lead_h'].mean()))
for lo_, hi_, lbl in [(0, 1, '≤1h'), (1, 2, '1~2h'), (2, 4, '2~4h'), (4, 99, '>4h')]:
    s = (R['lead_h'] > lo_) & (R['lead_h'] <= hi_)
    if lo_ == 0: s = (R['lead_h'] <= 1)
    print('    %-5s %4d (%.1f%%)' % (lbl, s.sum(), 100 * s.sum() / len(R)))

print('\n' + '=' * 90)
print('【B 顶部换手】峰值 ±1h 成交量 / 整段拉升成交量（谁在顶部接盘）')
print('=' * 90)
print('  顶部换手占比：中位 %s  均值 %s' % (pc(R['top_share'].median()), pc(R['top_share'].mean())))
print('  与涨幅相关 corr(gain, top_share) = %+.4f' % R['gain'].corr(R['top_share']))
hi_grp = R[R['top_share'] >= 0.30]; lo_grp = R[R['top_share'] < 0.30]
if len(hi_grp) and len(lo_grp):
    print('  高换手组 n=%d：涨幅中位 %.2f 倍，24h 站岗比例 %s' % (len(hi_grp), hi_grp['gain'].median(), pc(hi_grp['stand'].median())))
    print('  低换手组 n=%d：涨幅中位 %.2f 倍，24h 站岗比例 %s' % (len(lo_grp), lo_grp['gain'].median(), pc(lo_grp['stand'].median())))

print('\n' + '=' * 90)
print('【C 恐慌阈值】峰后首次「放量下跌(量比>5)」发生在跌多少 %（参与者的止损线）')
print('=' * 90)
p = R['panic'].dropna()
if len(p):
    print('  中位 %s  均值 %s  样本 %d' % (pc(p.median()), pc(p.mean()), len(p)))
    for lo_, hi_ in ((-1, -0.5), (-0.5, -0.3), (-0.3, -0.15), (-0.15, 0)):
        n = int(((p >= lo_) & (p < hi_)).sum())
        print('    跌幅 %4.0f%%~%4.0f%% 触发：%4d 次（%.1f%%）' % (100 * lo_, 100 * hi_, n, 100 * n / len(p)))
else:
    print('  （无样本）')

print('\n' + '=' * 90)
print('【D 站岗比例】峰值买进、之后 24h 内亏钱的时间占比（参与者的下场）')
print('=' * 90)
print('  中位 %s   均值 %s  样本 %d' % (pc(R['stand'].median()), pc(R['stand'].mean()), R['stand'].notna().sum()))
print('  经历「峰后亏 >50%% 时间」的事件占比：%s' % pc((R['stand'] > 0.5).mean()))
print('\nTOTAL %.0fs' % (time.time() - t0))
