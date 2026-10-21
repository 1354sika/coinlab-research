"""
gambler_hold_sim.py · 出场规则对比：固定 TP/SL vs 无固定线（纯浮盈规则）

消融已证明：「该持模型」≈ 浮盈距离计（AUC 0.9186 只剩 2 维浮盈特征）。
所以本次直接对比纯浮盈规则，不经过模型。

方案：
  A 固定 TP+5% / SL-2% / 1440            （基线）
  B 无 TP / 浮盈<0 就走 / 1440
  C 无 TP / 浮盈<1% 就走 / 1440
  D 无 TP / 移动止盈(峰值回落1%) / 浮盈<0 / 1440
  E 固定 TP+5% / 浮盈<0 就走 / 1440       （止损收紧，止盈保留）
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

SEQ = 60
A1_WIN = 240
A1_CONFIRM = 1.03
TP = 0.05
SL = 0.02
HOLD = 1440
FEE = 0.005
N_TRIGS = 10000

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

log = lambda s='': print(s, flush=True)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_hold_sim · 出场规则对比（纯浮盈规则，无模型）')
    log('=' * 100)
    log(f'  入场：{A1_WIN}min 低点 × {A1_CONFIRM}  当根收盘价')
    log(f'  持有上限 {HOLD} 根（24h）  成本 {FEE:.2%}/笔')
    log()

    log('读数据...')
    dtypes = {'symbol': 'category', 'open_time': 'int64',
              'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
              'volume': 'float32', 'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dtypes.keys()), dtype=dtypes)
    log(f'  {len(df):,} 行  {df.symbol.nunique()} 币')

    syms = sorted(df.symbol.unique())
    rng = np.random.default_rng(41)

    log('\n[1/3] 找 A1 触发点（全时段，不切分——这是纯规则对照，无训练）...')
    TRIGS = []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < A1_WIN + SEQ + 200:
            continue
        l = g.low.values.astype(np.float32)
        c = g.close.values.astype(np.float32)
        for i in range(A1_WIN + SEQ + 1, n - 1):
            if c[i] >= l[i-A1_WIN:i].min() * A1_CONFIRM:
                TRIGS.append((sym, i))
        if (si+1) % 15 == 0:
            log(f'  {si+1}/{len(syms)}  累计 {len(TRIGS):,}  {time.time()-t0:.0f}s')
    log(f'  共 {len(TRIGS):,}')

    if len(TRIGS) > N_TRIGS:
        sel = rng.choice(len(TRIGS), size=N_TRIGS, replace=False)
        TRIGS = [TRIGS[k] for k in sel]
        log(f'  采样 {len(TRIGS):,}')

    log('\n[2/3] 模拟（5 种出场规则并行）...')
    MODES = ['A', 'B', 'C', 'D', 'E']
    DESC = {
        'A': '固定 TP+5% / SL-2%',
        'B': '无TP / 浮盈<0 走',
        'C': '无TP / 浮盈<1% 走',
        'D': '无TP / 移动止盈(回落1%) / 浮盈<0',
        'E': 'TP+5% / 浮盈<0 走',
    }
    res = {m: {'n': 0, 'tp': 0, 'sl': 0, 'exit': 0, 'trail': 0, 'to': 0, 'ret': []} for m in MODES}

    CACHE = {}
    for k, (sym, i) in enumerate(TRIGS):
        if sym not in CACHE:
            g = df[df.symbol == sym].sort_values('open_time')
            CACHE[sym] = (g.high.values.astype(np.float32),
                          g.low.values.astype(np.float32),
                          g.close.values.astype(np.float32))
        h, l, c = CACHE[sym]
        n = len(c)
        entry = float(c[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(i + 1 + HOLD, n)
        if end - i < 30:
            continue

        tp_px = entry * (1 + TP)
        sl_px = entry * (1 - SL)

        for m in MODES:
            ret = None; why = None
            peak = -1e9
            for j in range(i + 1, end):
                un = c[j] / entry - 1.0
                if un > peak:
                    peak = un

                if m == 'A':
                    if h[j] >= tp_px:
                        ret = TP - FEE; why = 'tp'; break
                    if l[j] <= sl_px:
                        ret = -SL - FEE; why = 'sl'; break
                elif m == 'B':
                    if c[j] < entry:
                        ret = un - FEE; why = 'exit'; break
                elif m == 'C':
                    if un < 0.01:
                        ret = un - FEE; why = 'exit'; break
                elif m == 'D':
                    if c[j] < entry:
                        ret = un - FEE; why = 'exit'; break
                    if peak - un >= 0.01:
                        ret = un - FEE; why = 'trail'; break
                elif m == 'E':
                    if h[j] >= tp_px:
                        ret = TP - FEE; why = 'tp'; break
                    if c[j] < entry:
                        ret = un - FEE; why = 'exit'; break

            if ret is None:
                ret = c[end-1] / entry - 1.0 - FEE
                why = 'to'

            r = res[m]
            r['n'] += 1
            r[why] += 1
            r['ret'].append(ret)

        if (k+1) % 2000 == 0:
            log(f'  {k+1}/{len(TRIGS)}  {time.time()-t0:.0f}s')

    log('\n[3/3] 汇总...')
    log('\n' + '=' * 100)
    log('【结果】')
    log('=' * 100)
    log(f'  {"方案":<38}{"笔数":>7}{"止盈%":>7}{"止损/出%":>9}{"移动%":>7}{"超时%":>7}'
        f'{"均值%":>9}{"中位%":>9}{"胜率%":>8}{"判定":>5}')
    for m in MODES:
        r = res[m]
        if r['n'] == 0:
            continue
        rets = np.array(r['ret']) * 100
        mean = rets.mean(); med = np.median(rets); win = (rets > 0).mean() * 100
        nm = f'{m} {DESC[m]}'
        mark = '✓' if mean > 0 else '✗'
        log(f'  {nm:<38}{r["n"]:>7,}{r["tp"]/r["n"]*100:>6.1f}{r["exit"]/r["n"]*100:>8.1f}'
            f'{r["trail"]/r["n"]*100:>6.1f}{r["to"]/r["n"]*100:>6.1f}'
            f'{mean:>9.2f}{med:>9.2f}{win:>8.1f}{mark:>5}')

    log('\n【判读】')
    log('  · 若 B/C/D 均值 > A ⇒ 「不固定线」确实更优')
    log('  · 若全负 ⇒ 出场规则改不动问题，问题在入场')
    log('  · 看「超时%」：太高说明规则没在主动出场')

    log(f'\n【分布（A vs 最好方案）】')
    for m in MODES:
        rets = np.array(res[m]['ret']) * 100
        if len(rets) == 0:
            continue
        log(f'  {m}: p1={np.percentile(rets,1):+.2f}% p25={np.percentile(rets,25):+.2f}% '
            f'中位={np.median(rets):+.2f}% p75={np.percentile(rets,75):+.2f}% '
            f'p99={np.percentile(rets,99):+.2f}%')

    log(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
