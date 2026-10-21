# -*- coding: utf-8 -*-
"""day_legs_census · 一天里「低价→高价」到底有几次（1m 实测）

定义（全部只用当天已发生的价格，不涉及未来）：
  用 zigzag 把当天 close 序列切成上涨腿/下跌腿；
  上涨腿幅度 = 该腿起点低点到终点高点的涨幅。
  统计每个币每天「幅度 >= θ」的上涨腿次数。

θ 取 0.5% / 1% / 2% / 3%，回答：日内到底有几次够大的波段。
只读统计，不改数据。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\day_legs_census.txt'
THS = [0.005, 0.01, 0.02, 0.03]
NSYM = int(os.environ.get('DL_NSYM', '20'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def zigzag_up_legs(C, th):
    """返回当天所有上涨腿的幅度（低点→高点）。th=确认反转的幅度。"""
    legs = []
    trend = 0          # 0 未知, 1 上行, -1 下行
    lo = hi = C[0]
    for p in C[1:]:
        if trend >= 0:
            if p > hi:
                hi = p
            elif p < hi * (1 - th):
                if hi > lo * (1 + th):
                    legs.append(hi / lo - 1.0)
                trend = -1
                lo = p
        if trend <= 0:
            if p < lo:
                lo = p
            elif p > lo * (1 + th):
                trend = 1
                hi = p
    return legs


def main():
    t0 = time.time()
    say('=' * 100)
    say('一天里「低价→高价」有几次 · 1m 实测（zigzag 上涨腿普查）')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  θ = {[f"{t*100:.1f}%" for t in THS]}')
    say('=' * 100)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)].copy()
    df['date'] = pd.to_datetime(df.open_time, unit='ms').dt.date
    say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

    # 每个币每天：各 θ 下的上涨腿次数 + 腿幅度
    rows = []
    all_legs = {th: [] for th in THS}
    for sym in syms:
        g = df[df.symbol == sym]
        C = g['close'].to_numpy(float)
        d = g['date'].to_numpy()
        bounds = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1, len(d)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            seg = C[s:e]
            if len(seg) < 60 or not np.isfinite(seg).all():
                continue
            rec = dict(sym=sym, date=d[s], bars=len(seg))
            for th in THS:
                legs = zigzag_up_legs(seg, th)
                rec[f'n_{th}'] = len(legs)
                all_legs[th].extend(legs)
            rows.append(rec)

    R = pd.DataFrame(rows)
    say(f'  统计 {len(R):,} 个「币-天」  {time.time()-t0:.0f}s')

    say('')
    say('【一】每个币每天有几次上涨腿（中位 / 均值 / P25 / P75）')
    say(f'  {"θ":>6} {"中位":>7} {"均值":>7} {"P25":>6} {"P75":>6} {"至少1次的天数占比":>16}')
    for th in THS:
        c = R[f'n_{th}'].values
        say(f'  {th*100:>5.1f}% {np.median(c):>7.1f} {c.mean():>7.2f} '
            f'{np.percentile(c,25):>6.1f} {np.percentile(c,75):>6.1f} '
            f'{(c>=1).mean()*100:>15.1f}%')

    say('')
    say('【二】所有上涨腿的幅度分布（回答「这段够不够覆盖成本」）')
    say(f'  {"θ":>6} {"腿数":>9} {"中位幅度":>9} {"P75":>7} {"P90":>7} {"P99":>7}')
    for th in THS:
        a = np.array(all_legs[th])
        if len(a) == 0:
            continue
        say(f'  {th*100:>5.1f}% {len(a):>9,} {np.median(a)*100:>8.3f}% '
            f'{np.percentile(a,75)*100:>6.3f}% {np.percentile(a,90)*100:>6.3f}% '
            f'{np.percentile(a,99)*100:>6.3f}%')

    say('')
    say('【三】扣成本后还剩几次（maker 往返 0.04% / taker 往返 0.50%）')
    say(f'  {"成本口径":>10} {"要求腿幅":>9} {"θ=0.5% 每天次数":>16} {"θ=1% 每天次数":>15} {"θ=2% 每天次数":>15}')
    for name, cost in [('maker 0.04%', 0.0004), ('taker 0.50%', 0.0050)]:
        need = 3 * cost
        line = f'  {name:>10} {need*100:>8.2f}%'
        for th in (0.005, 0.01, 0.02):
            c = R[f'n_{th}'].values
            # 只数幅度 >= need 的腿：用腿幅度分布反推（同 θ 内）
            a = np.array(all_legs[th])
            frac = (a >= need).mean() if len(a) else 0.0
            line += f' {c.mean()*frac:>15.2f}'
        say(line)

    say('')
    say('【四】按币看（θ=1% 每天次数，中位）')
    for sym in syms:
        c = R[R.sym == sym]['n_0.01'].values
        if len(c):
            say(f'  {sym:<12} 天 {len(c):>4}  中位 {np.median(c):>4.1f}  均值 {c.mean():>5.2f}  '
                f'最多 {c.max():>3.0f}')

    say('')
    say('【读法】')
    say('  · 「几次」取决于你要求这段有多大：0.5% 的腿每天很多，2% 的腿每天很少；')
    say('  · 扣成本后（腿幅 >= 3×成本）才是「值得做」的次数，不是所有腿都算机会；')
    say('  · 这是 zigzag 口径，腿端点事后才确认；在线识别只能确认「已反弹 θ」，不能确认「这是最低点」。')
    say(f'TOTAL {time.time()-t0:.0f}s')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    R.to_csv(OUT.replace('.txt', '_rows.csv'), index=False, encoding='utf-8-sig')
    print(f'\n已写 {OUT}')


if __name__ == '__main__':
    main()
