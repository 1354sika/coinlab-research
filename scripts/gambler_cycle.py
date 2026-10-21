# -*- coding: utf-8 -*-
"""gambler_cycle.py — 按用户定义重新切「周期」（不是任意点看未来48h）

【用户定义（2026-10-17 口述，逐字）】
    「周期最长48h，上涨到百分之三十这个记录点之后还在涨就继续记录，
      直到跌出现，那么这个周期结束。不是到百分之30就结束，
      30只是这个曲线的起点，当斜率让曲线下降，那么周期结束。」

【据此实现的规则】
    1. 起点：从当前低点 L 起涨（低点用 low 滚动更新，未确认前一直下移）
    2. 确认：价格首次触及 L×1.30 → 该周期被「记录」（记录点 = 确认点）
    3. 记录：确认后继续涨就一直记，peak 用 high 滚动更新
    4. 结束：从 peak 回落 ≥ R（R = 斜率转负的阈值，本脚本分别跑 3%/5%/10%/20%）
             或  周期总长 > 48h（超时截断）
    5. 未确认就超过 48h → 该候选作废，从当前点重新起算
    6. 周期之间不重叠（结束后从结束点继续找下一个起点）

【输出】这是「普查」，不做预测，只回答：
    · 一共有多少个周期？每年多少？
    · 周期涨幅分布（中位/分位/最大）——用户要的「翻几倍」在哪个分位
    · 周期时长分布（确认用时、总时长）——用户说的「4~5 小时」对不对
    · 确认点之后还能涨多少（剩余空间）——跟盘的真正空间
    · 确认点时刻可观测的量（用了多久、量比、市场氛围）→ 后续做预测的输入
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
BAR_MIN = 5
MAX_BARS = int(48 * 60 / BAR_MIN)      # 48h = 576 根
TRIG = 0.30                            # 确认门槛 +30%
RS = [0.03, 0.05, 0.10, 0.20]          # 四种"斜率转负"阈值

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 120)
log('gambler_cycle · 按「周期」重新切数据（30% 是门槛不是终点，回落才算结束）')
log('=' * 120)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume', 'trades'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32',
                             'trades': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64'),
                    g['trades'].values.astype('float64')))
    del ch; gc.collect()
log(f'读入 {len(RAW)} 币   耗时 {time.time()-t0:.0f}s')

rows = []
for (sym, ot, c, h, l, q, tr) in RAW:
    n = len(c)
    if n < 600:
        continue
    qs = pd.Series(q)
    base_q = qs.rolling(288 * 30, min_periods=288).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values

    for R in RS:
        i = 0
        while i < n - 1:
            # ---- 找起点低点 ----
            lo_idx = i; lo = l[i]
            conf = -1; peak = -np.inf; peak_i = -1
            j = i
            while j < n - 1:
                j += 1
                if conf < 0:
                    if l[j] < lo:
                        lo = l[j]; lo_idx = j
                    if j - lo_idx > MAX_BARS:          # 未确认就超 48h → 作废重起
                        i = lo_idx + 1
                        break
                    if h[j] >= lo * (1 + TRIG):
                        conf = j
                        peak = h[j]; peak_i = j
                        if j - lo_idx > MAX_BARS:      # 确认时已超 48h
                            i = j + 1
                            break
                else:
                    if h[j] > peak:
                        peak = h[j]; peak_i = j
                    if j - conf > MAX_BARS:            # 周期总长超 48h（从确认算）
                        rows.append(dict(sym=sym, R=R, lo_i=lo_idx, conf_i=conf, peak_i=peak_i,
                                         end_i=j, exit='timeout',
                                         lo=lo, peak=peak, gain=peak / lo - 1,
                                         conf_min=(conf - lo_idx) * BAR_MIN,
                                         conf_gain=(peak / (lo * 1.3) - 1),
                                         span_min=(j - lo_idx) * BAR_MIN,
                                         peak_to_end=c[j] / peak - 1,
                                         volr6_conf=volr6[conf] if np.isfinite(volr6[conf]) else np.nan,
                                         t_conf=int(ot[conf])))
                        i = j + 1
                        break
                    if l[j] <= peak * (1 - R):         # 斜率转负 → 周期结束
                        rows.append(dict(sym=sym, R=R, lo_i=lo_idx, conf_i=conf, peak_i=peak_i,
                                         end_i=j, exit='drop',
                                         lo=lo, peak=peak, gain=peak / lo - 1,
                                         conf_min=(conf - lo_idx) * BAR_MIN,
                                         conf_gain=(peak / (lo * 1.3) - 1),
                                         span_min=(j - lo_idx) * BAR_MIN,
                                         peak_to_end=c[j] / peak - 1,
                                         volr6_conf=volr6[conf] if np.isfinite(volr6[conf]) else np.nan,
                                         t_conf=int(ot[conf])))
                        i = j + 1
                        break
            else:
                break   # 数据走完
del RAW; gc.collect()

DF = pd.DataFrame(rows)
DF.to_csv(os.path.join(OUT, 'cycles.csv'), index=False, encoding='utf-8-sig')
log(f'周期总数 {len(DF):,}   耗时 {time.time()-t0:.0f}s')
DF['year'] = pd.to_datetime(DF['t_conf'], unit='ms').dt.year

for R in RS:
    d = DF[DF.R == R]
    if len(d) == 0:
        continue
    log('\n' + '=' * 120)
    log(f'【回落阈值 R={R:.0%} 判周期结束】周期数 {len(d):,}   涉及币 {d.sym.nunique()}   '
        f'结束原因：回落 {100*(d.exit=="drop").mean():.1f}% / 超时 {100*(d.exit=="timeout").mean():.1f}%')
    log('=' * 120)
    g = d['gain']
    log(f'  周期涨幅（从低点到峰值）：中位 {100*g.median():.0f}%   '
        f'p25 {100*g.quantile(.25):.0f}%   p75 {100*g.quantile(.75):.0f}%   '
        f'p90 {100*g.quantile(.90):.0f}%   p99 {100*g.quantile(.99):.0f}%   最大 {100*g.max():.0f}%')
    for th in (0.5, 1.0, 2.0, 3.0):
        log(f'    涨幅 ≥{th*100:.0f}% 的周期：{int((g>=th).sum()):,} 个（{100*(g>=th).mean():.1f}%）')
    log(f'  确认用时（低点→首次+30%）：中位 {d.conf_min.median():.0f} 分钟   '
        f'p75 {d.conf_min.quantile(.75):.0f}   p90 {d.conf_min.quantile(.90):.0f}')
    log(f'  周期总时长（低点→结束）：中位 {d.span_min.median():.0f} 分钟（{d.span_min.median()/60:.1f}h）  '
        f'p75 {d.span_min.quantile(.75)/60:.1f}h   p90 {d.span_min.quantile(.90)/60:.1f}h')
    log(f'  ★ 确认点(+30%)之后还能涨多少：中位 {100*d.conf_gain.median():.0f}%   '
        f'p75 {100*d.conf_gain.quantile(.75):.0f}%   p90 {100*d.conf_gain.quantile(.90):.0f}%   '
        f'最大 {100*d.conf_gain.max():.0f}%')
    log(f'    确认后 ≤0（即 30% 基本见顶）：{100*(d.conf_gain<=0.02).mean():.1f}%')
    log(f'  按年：' + '  '.join(f'{y}:{int(c)}' for y, c in d.groupby('year').size().items()))
    topc = d.groupby('sym').size().sort_values(ascending=False).head(8)
    log(f'  最常见的币：' + '  '.join(f'{s.replace("USDT","")}:{int(c)}' for s, c in topc.items()))

out = os.path.join(OUT, 'cycle_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
