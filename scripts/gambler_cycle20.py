# -*- coding: utf-8 -*-
"""gambler_cycle20 · 按用户 2026-10-17 口径重新切「周期」，并分「局 / 散户博弈」两类

用户定义（逐字）：
  「定义一个周期为真实收益率为 0.2，也就是我投下去一块钱，赚 1.2 块钱甚至更多，
    直到收益率到达峰值开始下跌，或感知下跌即将开始为结束一个周期。」

实现：
  1. 起点：滚动低点 L（未确认前一直下移）
  2. 记录点：价格首次触及 L*1.20（= 投 1 块能拿回 1.2 块）
  3. 峰值：记录点后 high 滚动更新
  4. 结束：从峰值回落 >= R；**"感知下跌"在数据里不可得，用回落 R 代理**（诚实标注）
  5. 周期不重叠

分类（沿用 pump_hunt 的局定义）：
  局   = 峰值涨幅 >= 50% 且 峰值后回落 >= 30%
  散户 = 其余（峰值涨幅 < 50%，或回落 < 30%）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
TRIG = 0.20
RS = [0.20, 0.30, 0.40]   # 小 R 会把大局切碎：拉升途中的正常回调就会结束周期
MAX_WAIT = 720          # 1h 根 = 30 天

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_cycle20 · 用户口径：真实收益 0.2 起算的周期 + 局/散户二分')
log('=' * 116)

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
log(f'币 {len(RAW)}  读盘 {time.time()-t0:.0f}s')

H1 = []
for (sym, ot, c, h, l, q) in RAW:
    key = ot // 3600000
    uniq, st = np.unique(key, return_index=True)
    en = np.append(st[1:], len(key)) - 1
    H1.append((sym, uniq * 3600000, c[en],
               np.maximum.reduceat(h, st), np.minimum.reduceat(l, st),
               np.add.reduceat(q, st)))
log(f'1h根 {sum(len(x[1]) for x in H1):,}  {time.time()-t0:.0f}s')
del RAW; gc.collect()

def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else np.nan

for R in RS:
    rows = []
    for (sym, ot, c, h, l, q) in H1:
        n = len(c)
        if n < 200:
            continue
        qma = pd.Series(q).rolling(24, min_periods=6).mean().values
        i = 0
        while i < n - 2:
            lo = l[i]; lo_i = i
            conf = -1
            j = i
            while j < n - 1 and j - lo_i <= MAX_WAIT:
                j += 1
                if l[j] < lo:
                    lo = l[j]; lo_i = j
                if h[j] >= lo * (1 + TRIG):
                    conf = j; break
            if conf < 0:
                break
            peak = h[conf]; pk = conf
            k = conf
            while k < n - 1:
                k += 1
                if h[k] > peak:
                    peak = h[k]; pk = k
                if l[k] <= peak * (1 - R):
                    break
            end = min(k, n - 1)
            if pk <= conf:
                i = end + 1
                continue
            volr = q[conf] / qma[conf] if qma[conf] > 0 else np.nan
            rows.append(dict(
                sym=sym,
                reach_h=conf - lo_i,                    # 低点→+20%
                conf2peak_h=pk - conf,                  # +20%→峰值
                total_h=end - lo_i,                     # 周期总长
                peak_gain=peak / lo - 1,                # 低点→峰值
                rest_after20=peak / (lo * (1 + TRIG)) - 1,   # +20% 后还剩多少
                dd=(peak - c[end]) / peak,              # 结束时回落深度
                volr=volr,
                t_lo=int(ot[lo_i]), t_peak=int(ot[pk]),
            ))
            i = end + 1
    if not rows:
        log(f'R={R:.0%}: 无周期')
        continue
    df = pd.DataFrame(rows)
    df['is_op'] = (df.peak_gain >= 0.50) & (df.dd >= 0.30)
    log('\n' + '=' * 116)
    log(f'【R={R:.0%} 判结束】周期 {len(df):,} 个  涉及币 {df.sym.nunique()}')
    log('=' * 116)
    log(f'{"指标":22s}{"全部":>12s}{"局":>14s}{"散户":>14s}')
    log('-' * 116)
    def line(name, f):
        a, b, c_ = f(df), f(df[df.is_op]), f(df[~df.is_op])
        log(f'{name:22s}{a:>12}{b:>14}{c_:>14}')
    line('周期数', lambda d: f'{len(d):,}')
    line('低点→+20% 用时h 中位', lambda d: f'{d.reach_h.median():.1f}')
    line('  用时 P10/P90', lambda d: f'{pct(d.reach_h,10):.1f}/{pct(d.reach_h,90):.1f}')
    line('+20%→峰值 用时h 中位', lambda d: f'{d.conf2peak_h.median():.1f}')
    line('周期总长h 中位', lambda d: f'{d.total_h.median():.1f}')
    line('峰值涨幅 中位', lambda d: f'{d.peak_gain.median():.1%}')
    line('+20%后还剩 中位', lambda d: f'{d.rest_after20.median():.1%}')
    line('+20%后还剩 P75', lambda d: f'{pct(d.rest_after20,75):.1%}')
    line('结束回落 中位', lambda d: f'{d.dd.median():.1%}')
    line('确认时量比 中位', lambda d: f'{d.volr.median():.2f}')
    log('')
    log(f'局占比 {df.is_op.mean():.1%}（{df.is_op.sum():,} / {len(df):,}）')
    # 局内部再分快慢
    op = df[df.is_op]
    if len(op):
        fast = op[op.reach_h <= op.reach_h.median()]
        slow = op[op.reach_h > op.reach_h.median()]
        log(f'  局内：快速起涨(<=中位 {op.reach_h.median():.1f}h) {len(fast):,} 个，峰值涨幅中位 {fast.peak_gain.median():.1%}，+20%后剩 {fast.rest_after20.median():.1%}')
        log(f'        慢速起涨(>中位)              {len(slow):,} 个，峰值涨幅中位 {slow.peak_gain.median():.1%}，+20%后剩 {slow.rest_after20.median():.1%}')
    # 按峰值涨幅分档（这才是「局/散户」的真实分布）
    bins = [(0.20, 0.35, '散户·小 20~35%'), (0.35, 0.50, '散户·中 35~50%'),
            (0.50, 1.00, '局·小 50~100%'), (1.00, np.inf, '局·大 >=100%')]
    log('  ── 按峰值涨幅分档 ──')
    log(f'  {"档":16s}{"个数":>7s}{"占比":>7s}{"到+20%用时h":>12s}{"+20%后剩":>10s}{"总长h":>8s}{"量比":>7s}')
    for lo_, hi_, name in bins:
        d = df[(df.peak_gain >= lo_) & (df.peak_gain < hi_)]
        if len(d) == 0:
            continue
        log(f'  {name:16s}{len(d):>7,}{len(d)/len(df):>7.1%}'
            f'{d.reach_h.median():>12.1f}{d.rest_after20.median():>10.1%}'
            f'{d.total_h.median():>8.1f}{d.volr.median():>7.2f}')
    # 时间 vs 幅度：时间是不是隐性
    if len(df) > 100:
        for col in ['reach_h', 'conf2peak_h', 'volr']:
            s = df[[col, 'peak_gain']].dropna()
            if len(s) > 100:
                rho = s[col].corr(s.peak_gain, method='spearman')
                log(f'  相关性：{col} vs 峰值涨幅  Spearman={rho:+.3f}')
    df.to_csv(os.path.join(OUT, f'cycle20_rows_R{int(R*100)}.csv'), index=False)

log('\n' + '=' * 116)
log('【判读要点】')
log('=' * 116)
log('  · "时间隐性"的证据：看 用时 P10/P90 的跨度，以及 用时 vs 峰值涨幅 的相关性（若 |rho|<0.2 → 时间不由幅度决定）')
log('  · 局与散户的差别若只在"幅度"不在"时间/量能"，说明两者是同一机制的强弱版本，不是两种物种')
log('  · "感知下跌"不可得，用回落 R 代理 —— 这是代理误差，不是发现')

out = os.path.join(OUT, 'cycle20_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
