# -*- coding: utf-8 -*-
"""gambler_longrun_both · 大小周期各一条线 · 本金 100 · 连续 25 个月

数据：klines_5m_3y.csv（2023-10~2026-10）
大周期：止盈=低点×1.30，止损=低点×0.95，持有 1440 根(5天)
小周期：止盈=入场×1.05，止损=入场×0.98，持有 12 根(1小时)
筛选用各自口径的 h3>=1.0（最近 3 次同口径 TP）
训练 2023-10~2024-09，测试 2024-10~2026-10（连续 25 个月）
本金 100 元，每笔 10% 仓位，成本 0.50%/笔
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
START = 100.0
POS = 0.10
MAXSYM = int(os.environ.get('SYMS', '60'))

VW = 240
CONF = 2880
HOLD_BIG = 1440
HOLD_SML = 12
BIG_TP, BIG_SL = 0.30, 0.95
SML_TP, SML_SL = 0.05, 0.02

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_longrun_both · 大小周期各一条线 · 本金 100 · 连续 25 个月')
log('=' * 100)
log(f'大周期：低点×1.30 / 低点×0.95 / 持有 {HOLD_BIG} 根 5m(5天)')
log(f'小周期：入场×{(1+SML_TP):.2f} / 入场×{(1-SML_SL):.2f} / 持有 {HOLD_SML} 根 5m(1小时)')
log(f'本金 {START:.0f} 元，每笔 {POS:.0%} 仓位，成本 {FEE*100:.2f}%/笔')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_5m_3y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = pd.Series(l).rolling(121, min_periods=1).min().values
    lo_i = []
    i = 120
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 30
        else:
            i += 1
    pairs = []
    for i in lo_i:
        if i + CONF >= n: continue
        for j in range(i + 1, min(i + CONF + 1, n)):
            if c[j] >= l[i] * 1.03:
                pairs.append((i, j)); break
    return pairs


def judge(seg_h, seg_l, tp_px, sl_px, e, hold):
    hit_tp = seg_h >= tp_px
    hit_sl = seg_l <= sl_px
    i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
    if i_tp >= 0 and i_sl >= 0:
        if i_tp < i_sl:
            return tp_px / e - 1, 1
        return sl_px / e - 1, 0
    if i_tp >= 0:
        return tp_px / e - 1, 1
    if i_sl >= 0:
        return sl_px / e - 1, 0
    return None, 0


ROWS_B, ROWS_S = [], []
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    hb, hs = [], []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD_BIG >= n: continue
        e = c[j]; lo_p = l[li]
        ts = int(t[j])

        # 大周期
        seg_h = h[j + 1:j + 1 + HOLD_BIG]
        seg_l = l[j + 1:j + 1 + HOLD_BIG]
        rb, tpb = judge(seg_h, seg_l, lo_p * (1 + BIG_TP), lo_p * BIG_SL, e, HOLD_BIG)
        if rb is None:
            rb = c[min(j + HOLD_BIG, n - 1)] / e - 1; tpb = 0
        h3b = float(np.mean(hb[-3:])) if hb else -1.0
        ROWS_B.append((ts, rb, tpb, h3b))
        hb.append(tpb)

        # 小周期
        seg_h2 = h[j + 1:j + 1 + HOLD_SML]
        seg_l2 = l[j + 1:j + 1 + HOLD_SML]
        rs, tps = judge(seg_h2, seg_l2, e * (1 + SML_TP), e * (1 - SML_SL), e, HOLD_SML)
        if rs is None:
            rs = c[min(j + HOLD_SML, n - 1)] / e - 1; tps = 0
        h3s = float(np.mean(hs[-3:])) if hs else -1.0
        ROWS_S.append((ts, rs, tps, h3s))
        hs.append(tps)
log(f'大周期入场 {len(ROWS_B):,}  小周期入场 {len(ROWS_S):,}  耗时 {time.time()-t0:.0f}s')

TEST_FROM = '2024-10'


def run(ROWS, name):
    d = pd.DataFrame(ROWS, columns=['t', 'ret', 'tp', 'h3'])
    d['ym'] = pd.to_datetime(d['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
    train_m = [m for m in sorted(d['ym'].unique()) if m < TEST_FROM]
    te = d[d['ym'] >= TEST_FROM].copy()
    log('')
    log('=' * 100)
    log(f'【{name}】训练 {len(train_m)} 月，测试 {te["ym"].nunique()} 月连续')
    log(f'  测试全样本 {len(te):,} 笔  TP率 {te["tp"].mean()*100:.1f}%')
    # h3>=1.0 筛选（不训练模型，纯动量）
    sel = te[te['h3'] >= 1.0].sort_values('t')
    if len(sel) == 0:
        log('  h3>=1.0 无交易')
        return
    cap = START; peak = START; dd = 0.0; curve = []
    for _, row in sel.iterrows():
        cap *= (1 + POS * (row['ret'] - FEE))
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
        curve.append((row['t'], cap, row['ret']))
    log(f'  h3>=1.0 交易 {len(sel)} 笔  均值 {sel["ret"].mean()*100:>6.2f}%  '
        f'胜率 {(sel["ret"]>0).mean()*100:>5.1f}%  {START:.0f} → {cap:>16,.2f} 元  回撤 {dd*100:>5.1f}%')
    log('')
    log(f"  {'月份':<10}{'笔数':>6}{'均值%':>9}{'月末资金':>18}{'月收益%':>10}")
    cv = pd.DataFrame(curve, columns=['t', 'cap', 'ret'])
    cv['ym'] = pd.to_datetime(cv['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
    g = cv.groupby('ym').agg(n=('ret', 'count'), mean=('ret', 'mean'), last=('cap', 'last'))
    prev = START
    for ym, row in g.iterrows():
        log(f'  {ym:<10}{int(row["n"]):>6}{row["mean"]*100:>9.2f}{row["last"]:>18,.2f}{(row["last"]/prev-1)*100:>10.1f}%')
        prev = row['last']


run(ROWS_B, '大周期')
run(ROWS_S, '小周期')

log('')
log('=' * 100)
log('【判读】')
log('  · 本金 100，每笔 10% 仓位，成本 0.50%/笔')
log('  · 大小周期都用各自的 h3>=1.0 动量筛选')
log('  · 连续 25 个月不中断，包含 2024-11 牛市、2025 熊市、2026 牛市月')
log('  · 复利数字是数学放大；看单笔均值与胜率是否稳定')

out = os.path.join(OUT, 'longrun_both_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
