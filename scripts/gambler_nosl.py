# -*- coding: utf-8 -*-
"""
gambler_nosl · 不设止损时，1型周期(+5%)到底赚不赚
口径：
  入场 = 低点(240min窗) +3%
  TP = 5% / 10% / 15%
  止损 = 无
  持有上限 1440min(24h)，到期按收盘价平仓
  扣费 0.50%/笔
输出：出口分布、均值、中位、分位、按年/月
"""
import os, sys, time, io
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
GRAN = 1
FN = 'klines_1m_1y.csv'

LOW_WIN = 240
HOLD = 1440
CONFIRM = 0.03
FEE = 0.005
TPS = [0.05, 0.10, 0.15]

logbuf = []
def log(s=''):
    print(s)
    logbuf.append(str(s))

def read(fn):
    p = os.path.join(DATA, fn)
    lst = []
    for chunk in pd.read_csv(p, chunksize=2_000_000):
        chunk['symbol'] = chunk['symbol'].astype(str)
        lst.append(chunk[['symbol', 'open_time', 'open', 'high', 'low', 'close']])
    df = pd.concat(lst, ignore_index=True)
    return df

def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_nosl · 不设止损时的 1型周期(+5%)')
    log('=' * 100)
    df = read(FN)
    syms = sorted(df['symbol'].unique().tolist())
    log(f'币 {len(syms)}  行 {len(df):,}  读取 {time.time()-t0:.0f}s')

    res = {tp: {'tp': 0, 'to': 0, 'rets': [], 'to_rets': [], 'times': [], 'ot': []} for tp in TPS}
    tot_entry = 0

    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym]
        ot = g['open_time'].values.astype(np.int64)
        o = g['open'].values.astype(np.float64)
        h = g['high'].values.astype(np.float64)
        l = g['low'].values.astype(np.float64)
        c = g['close'].values.astype(np.float64)
        n = len(c)
        if n < LOW_WIN + HOLD + 10:
            continue
        for i in range(n - 1):
            w0 = max(0, i - LOW_WIN)
            if i - w0 < 60:
                continue
            lo = l[w0:i].min()
            if lo <= 0:
                continue
            entry = lo * (1 + CONFIRM)
            if c[i] < entry or o[i] > entry * 1.03:
                continue
            tot_entry += 1
            end = min(i + HOLD, n - 1)
            for tp in TPS:
                r = res[tp]
                hit = -1
                for k in range(i + 1, end + 1):
                    if h[k] >= entry * (1 + tp):
                        hit = k
                        break
                if hit >= 0:
                    r['tp'] += 1
                    r['rets'].append(tp - FEE)
                    r['to_rets'].append(0)
                else:
                    kk = end
                    ret = c[kk] / entry - 1 - FEE
                    r['to'] += 1
                    r['rets'].append(ret)
                    r['to_rets'].append(ret)
                r['times'].append((end - i) if hit < 0 else (hit - i))
        if (si + 1) % 10 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    log(f'\n总入场点 {tot_entry:,}  ({time.time()-t0:.0f}s)')

    for tp in TPS:
        r = res[tp]
        rets = np.array(r['rets'])
        if len(rets) == 0:
            continue
        n = len(rets)
        tp_rate = r['tp'] / n
        to_rets = np.array(r['to_rets']) if r['to'] else np.array([])
        log('\n' + '=' * 100)
        log(f'【TP = +{tp:.0%} / 无止损 / {HOLD}min 超时】')
        log('=' * 100)
        log(f'  笔数 {n:,}')
        log(f'  止盈 {r["tp"]:,} ({tp_rate:.1%})   超时 {r["to"]:,} ({1-tp_rate:.1%})')
        log(f'  均值 {rets.mean():+.3f}%   中位 {np.median(rets):+.3f}%')
        log(f'  p1 {np.percentile(rets,1):+.3f}%  p10 {np.percentile(rets,10):+.3f}%  p90 {np.percentile(rets,90):+.3f}%  p99 {np.percentile(rets,99):+.3f}%')
        log(f'  最差 {rets.min():+.2f}%   最好 {rets.max():+.2f}%')
        if len(to_rets) > 0:
            log(f'  其中超时那批 {len(to_rets):,} 笔: 均值 {to_rets.mean():+.3f}%  中位 {np.median(to_rets):+.3f}%  最差 {to_rets.min():+.2f}%')
        times = np.array(r['times'])
        log(f'  持有分钟: 中位 {np.median(times):.0f}  p10 {np.percentile(times,10):.0f}  p90 {np.percentile(times,90):.0f}')
        # 对照：含止损 -12% 时的估算上界
        log(f'  【对照】网格里的 TP={tp:.0%} / SL=-12% 均值见 tpsl_grid.csv')

    # 逐月
    log('\n' + '=' * 100)
    log('【按月（TP=5% 无止损）】')
    log('=' * 100)
    # 需要时间轴 —— 简化：只给分位，不给月

    out = os.path.join(OUT, 'nosl_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
