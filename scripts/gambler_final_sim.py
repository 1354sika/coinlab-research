"""
gambler_final_sim.py · 完整资金曲线模拟（不是单笔期望）

入场：240min 低点 × 1.03，当根收盘价
出场：浮盈 < 1% 就走 / 1440 根超时
成本：0.50%/笔
仓位：10%/笔，同时最多持 MAX_POS 个币，总仓位上限 MAX_USE
本金：100
"""
import sys, io, os, time, heapq
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

SEQ = 60
A1_WIN = 240
A1_CONFIRM = 1.03
HOLD = 1440
FEE = 0.005
EXIT_PNL = 0.01          # 浮盈 < 1% 就走
MAX_POS = 5              # 同时最多 5 个币
MAX_USE = 0.50           # 总仓位上限 50%
POS_PCT = 0.10           # 每笔 10%
START = 100.0

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

log = lambda s='': print(s, flush=True)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_final_sim · 完整资金曲线模拟')
    log('=' * 100)
    log(f'  入场：{A1_WIN}min 低点 × {A1_CONFIRM}  当根收盘价')
    log(f'  出场：浮盈 < {EXIT_PNL:.0%} 就走 / {HOLD} 根超时')
    log(f'  成本 {FEE:.2%}/笔   仓位 {POS_PCT:.0%}/笔  最多持 {MAX_POS} 个  总仓位 {MAX_USE:.0%}')
    log(f'  本金 {START:.0f} 元')
    log()

    log('读数据...')
    dtypes = {'symbol': 'category', 'open_time': 'int64',
              'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
              'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())

    log('\n[1/3] 找 A1 触发点 + 预计算每笔出场...')
    CACHE = {}
    trades = []   # (t_in, t_out, ret, sym)
    n_trig = 0
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < A1_WIN + SEQ + 300:
            continue
        h = g.high.values.astype(np.float32)
        l = g.low.values.astype(np.float32)
        c = g.close.values.astype(np.float32)
        t = g.open_time.values

        for i in range(A1_WIN + SEQ + 1, n - 2):
            if c[i] < l[i-A1_WIN:i].min() * A1_CONFIRM:
                continue
            n_trig += 1
            if n_trig % 100 != 0:      # 采样 1/100：全量 328 万触发点跑不动
                continue
            entry = float(c[i])
            if not np.isfinite(entry) or entry <= 0:
                continue
            end = min(i + 1 + HOLD, n)
            if end - i < 30:
                continue
            ret = None
            t_out = None
            for j in range(i + 1, end):
                un = c[j] / entry - 1.0
                if un < EXIT_PNL:
                    ret = un - FEE
                    t_out = int(t[j])
                    break
            if ret is None:
                ret = c[end-1] / entry - 1.0 - FEE
                t_out = int(t[end-1])
            trades.append((int(t[i]), t_out, ret, sym))

        if (si+1) % 15 == 0:
            log(f'  {si+1}/{len(syms)}  触发 {n_trig:,}  有效 {len(trades):,}  {time.time()-t0:.0f}s')

    log(f'  总触发 {n_trig:,}  有效交易 {len(trades):,}')
    trades.sort(key=lambda x: x[0])

    log('\n[2/3] 资金池模拟...')
    cash = START
    held = []       # heap: (t_out, size, ret, sym)
    equity = []     # (t, equity, cash, used)
    n_taken = 0
    n_skip_full = 0
    n_skip_cash = 0

    for (t_in, t_out, ret, sym) in trades:
        # 释放到期持仓
        while held and held[0][0] <= t_in:
            _, size, r, s = heapq.heappop(held)
            cash += size * (1.0 + r)

        # 检查仓位
        used = sum(x[1] for x in held)
        total = cash + used
        want = total * POS_PCT

        if len(held) >= MAX_POS:
            n_skip_full += 1
            continue
        if used + want > total * MAX_USE:
            n_skip_cash += 1
            continue
        if cash < want:
            n_skip_cash += 1
            continue

        cash -= want
        heapq.heappush(held, (t_out, want, ret, sym))
        n_taken += 1

        if n_taken % 500 == 0:
            cur = cash + sum(x[1] for x in held)
            equity.append((t_in, cur, cash, used))

    # 释放剩余
    while held:
        _, size, r, s = heapq.heappop(held)
        cash += size * (1.0 + r)

    final = cash
    log(f'  开仓 {n_taken:,}  因满仓跳过 {n_skip_full:,}  因资金跳过 {n_skip_cash:,}')

    log('\n[3/3] 资金曲线统计...')
    # 重跑一遍记录每笔结算后的权益
    cash = START
    held = []
    curve = []   # (t, equity)
    for (t_in, t_out, ret, sym) in trades:
        while held and held[0][0] <= t_in:
            tt, size, r, s = heapq.heappop(held)
            cash += size * (1.0 + r)
            curve.append((tt, cash + sum(x[1] for x in held)))
        used = sum(x[1] for x in held)
        total = cash + used
        want = total * POS_PCT
        if len(held) >= MAX_POS or used + want > total * MAX_USE or cash < want:
            continue
        cash -= want
        heapq.heappush(held, (t_out, want, ret, sym))
    while held:
        tt, size, r, s = heapq.heappop(held)
        cash += size * (1.0 + r)
        curve.append((tt, cash))

    log('\n' + '=' * 100)
    log('【结果】')
    log('=' * 100)
    log(f'  本金 {START:.0f} → 最终 {final:,.2f} 元  （{final/START:.2f}x，{final/START-1:+.1%}）')

    if curve:
        curve.sort(key=lambda x: x[0])
        eq = np.array([c[1] for c in curve])
        ts = np.array([c[0] for c in curve])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        log(f'  最高权益 {eq.max():,.2f}   最低 {eq.min():,.2f}')
        log(f'  最大回撤 {dd.min():.1%}')

        # 按月
        dt = pd.to_datetime(ts, unit='ms', utc=True)
        months = pd.Series(dt).dt.to_period('M').values
        df_c = pd.DataFrame({'m': months, 'eq': eq})
        log(f'\n【按月】')
        log(f'  {"月份":<10}{"月末资金":>14}{"月收益%":>10}')
        prev = START
        for m, g2 in df_c.groupby('m', sort=True):
            last = g2['eq'].iloc[-1]
            log(f'  {str(m):<10}{last:>14,.2f}{(last/prev-1)*100:>+9.1f}%')
            prev = last

    log('\n【判读】')
    log('  · 最终 > 100 ⇒ 这套算法在历史模拟里是赚的')
    log('  · 最大回撤 < 50% ⇒ 风险可接受')
    log('  · 按月若多数为负 ⇒ 只是某几个月运气好')

    log(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
