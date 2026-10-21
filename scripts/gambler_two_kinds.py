# -*- coding: utf-8 -*-
"""gambler · 群体操盘 vs 个人操盘：两种不同的局，分开刻画

用户指出：操盘手是「群体」还是「个人」是两种不同的情况 —— 必须分开建模。

可操作判据（单靠 K 线可算，无前视）：
  群体型 = 该币异动时，**同一时刻**全市场多个币也在异动（跨币同步度高）
  个体型 = 孤立异动（它动的时候别人安静）

做法：
  1 对 pump_hunt_events.csv 里每个"局"，量出它启动时刻的全市场同步度
    sync_start = 事件起点 ±2 根内，全市场"放量上涨异动"的币数（量比≥5 且 ret≥+5%）
    sync_peak  = 峰值时刻 ±2 根内，全市场"放量下跌异动"的币数（撤走是否一起撤）
  2 按 sync_start 中位数分两组：群体型 / 个体型
  3 两组对照：涨幅规模、到峰值时长、之后回撤、撤走同步度、币的流动性
  4 关键对照：跟盘收益（事件起点入场 + 移动止损 25%，持有到 24h）
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler · 群体操盘 vs 个人操盘（同一时刻有多少别的币在陪它动）')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
piv = {}
for c in ('close', 'high', 'low', 'quote_volume'):
    m = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
    piv[c] = m
close, high, low, qv = piv['close'], piv['high'], piv['low'], piv['quote_volume']
del df; gc.collect()
close = close.where(close > 0).ffill(limit=3)
qv = qv.where(qv >= 0)
idx = close.index.values
T, N = close.shape
syms = np.array(list(close.columns)); pos_of = {s: i for i, s in enumerate(syms)}
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('%d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
med = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
med = np.where(med > 0, med, np.nan)
volr = QV / med
ret1[np.isfinite(volr) & (volr > 200)] = np.nan
M_UP = np.isfinite(volr) & np.isfinite(ret1) & (volr >= 2) & (ret1 >= 0.03)
M_DN = np.isfinite(volr) & np.isfinite(ret1) & (volr >= 2) & (ret1 <= -0.03)
n_up = M_UP.sum(axis=1).astype(np.int32)          # 每桶：全市场放量上涨异动币数
n_dn = M_DN.sum(axis=1).astype(np.int32)
say('每桶同时异动币数中位：上涨 %.1f / 下跌 %.1f ；≥5 币的桶占 %.2f%% / ≥3 币占 %.2f%%  %.0fs'
    % (np.median(n_up), np.median(n_dn), 100 * (n_up >= 5).mean(), 100 * (n_dn >= 3).mean(), time.time() - t0))

ev = pd.read_csv(os.path.join(EV, 'pump_hunt_events.csv'))
ev['symbol'] = ev['symbol'].astype(str)
def _to_ms(s):
    v = pd.to_datetime(s).astype('int64').values.astype('int64')
    d = int(v[0])
    if d > 10**17:
        v = v // 10**6      # 纳秒 → 毫秒
    elif d > 10**14:
        v = v // 10**3      # 微秒 → 毫秒
    elif d > 10**11:
        pass                # 已是毫秒
    else:
        v = v * 1000        # 秒 → 毫秒
    return v
tm = _to_ms(ev['start_time'])
tp = _to_ms(ev['peak_time'])
p0 = np.searchsorted(idx, tm); p1 = np.searchsorted(idx, tp)
ok = (p0 > 10) & (p0 < T - 300) & (p1 < T - 10)
ev, p0, p1 = ev[ok].reset_index(drop=True), p0[ok], p1[ok]
say('事件 %d（+50%%门槛 %d / +100%% %d / +200%% %d），覆盖 %d 个币'
    % (len(ev), int((ev['thr'] == 0.5).sum()), int((ev['thr'] == 1.0).sum()),
       int((ev['thr'] == 2.0).sum()), ev['symbol'].nunique()))

lo = np.maximum(p0 - 2, 0); hi = np.minimum(p0 + 3, T)
sync_start = np.array([n_up[lo[i]:hi[i]].mean() for i in range(len(p0))])
lo2 = np.maximum(p1 - 2, 0); hi2 = np.minimum(p1 + 3, T)
sync_peak = np.array([n_dn[lo2[i]:hi2[i]].mean() for i in range(len(p1))])
ev['sync_start'] = sync_start; ev['sync_peak'] = sync_peak
med_s = np.median(sync_start)
ev['kind'] = np.where(sync_start >= med_s, '群体', '个体')
say('\n启动同步度中位 = %.2f 币 ⇒ 分组：群体 %d 事件 / 个体 %d 事件'
    % (med_s, int((ev['kind'] == '群体').sum()), int((ev['kind'] == '个体').sum())))

# 之后 24h 回撤 & 跟盘收益
fwd24 = np.full((T, N), np.nan, np.float32); fwd24[:-288] = CV[288:] / CV[:-288] - 1.0
rows = []
for i in range(len(ev)):
    s = ev['symbol'][i]; j = pos_of.get(s)
    if j is None: continue
    # 峰值后 24h 最大回撤（撤走的狠度）
    if p1[i] + 288 < T:
        seg = CV[p1[i]:p1[i] + 288, j]
        peak = HV[p1[i], j]
        dd = np.nanmin(seg / peak - 1.0) if np.isfinite(peak) and peak > 0 else np.nan
    else:
        dd = np.nan
    # 跟盘：起点入场，-25% 移动止损，24h 强平（maker 双边 0.04%）
    entry = ev['start_close'][i]
    end = min(p0[i] + 288, T - 1)
    seg_l, seg_h = LV[p0[i]:end, j], HV[p0[i]:end, j]
    ret = CV[end, j] / entry - 1.0
    stop_hit = np.argmax(seg_l / entry - 1.0 <= -0.25) if (seg_l / entry - 1.0 <= -0.25).any() else -1
    trail = -1
    if stop_hit >= 0:
        ret = -0.25
    elif len(seg_h) > 0:
        peak_trail = np.maximum.accumulate(seg_h)
        hit = np.argmax(seg_l / peak_trail - 1.0 <= -0.25) if (seg_l / peak_trail - 1.0 <= -0.25).any() else -1
        if hit >= 0:
            ret = np.nanmax(seg_h[:hit + 1]) / entry - 1.0 - 0.25
    rows.append((s, ev['kind'][i], ev['thr'][i], ev['gain'][i], ev['min_to_peak'][i],
                 sync_start[i], sync_peak[i], dd, ret - 0.0008,
                 np.nansum(QV[:, j])))
R = pd.DataFrame(rows, columns=['symbol', 'kind', 'thr', 'gain', 'mins', 'sync_start', 'sync_peak', 'dd24', 'net', 'qv_sum'])
R.to_csv(os.path.join(EV, 'gambler_two_kinds.csv'), index=False, encoding='utf-8-sig')

say('\n' + '=' * 100)
say('【对照表】群体型 vs 个体型')
say('=' * 100)
say('  %-6s %7s %9s %9s %9s %9s %9s %9s %9s'
    % ('类型', '事件数', '最大涨幅', '到峰值min', '撤走同步', '24h回撤', '跟盘净收益', '胜率', '成交额中位'))
for k in ('群体', '个体'):
    s = R[R['kind'] == k]
    if not len(s): continue
    say('  %-6s %7d %8.1f%% %9.0f %9.2f %8.1f%% %8.2f%% %8.1f%% %9.2e'
        % (k, len(s), 100 * s['gain'].median(), s['mins'].median(), s['sync_peak'].median(),
           100 * s['dd24'].median(), 100 * s['net'].mean(), 100 * (s['net'] > 0).mean(),
           s['qv_sum'].median()))

say('\n【按门槛细看】')
for thr in (0.5, 1.0, 2.0):
    for k in ('群体', '个体'):
        s = R[(R['thr'] == thr) & (R['kind'] == k)]
        if len(s) < 5: continue
        say('  +%d%% %-4s n=%4d  最大涨幅中位%6.1f%%  跟盘净收益均值%+7.2f%%  胜率%5.1f%%  ≥+100%%占%4.1f%%'
            % (int(thr * 100), k, len(s), 100 * s['gain'].median(), 100 * s['net'].mean(),
               100 * (s['net'] > 0).mean(), 100 * (s['net'] >= 1.0).mean()))

say('\n【相关性】sync_start 与各量的相关（事件级）')
for c in ('gain', 'mins', 'dd24', 'net'):
    say('  sync_start vs %-5s corr = %+.4f' % (c, R['sync_start'].corr(R[c])))
say('  sync_peak  vs %-5s corr = %+.4f（撤走同步度 vs 24h 回撤）' % ('dd24', R['sync_peak'].corr(R['dd24'])))

open(os.path.join(EV, 'gambler_two_kinds.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
