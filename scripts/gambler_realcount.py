# -*- coding: utf-8 -*-
"""gambler_realcount.py — "真局"到底有多少个？六种口径普查
用户意见（2026-10-16）：分档应为 无/微/弱/真 —— 与实现一致（Stage1 判"无"，Stage2 判微/弱/真）。
  但"真局"一档在三轮实验中始终 0 报出，根因疑为**真局定义过严 + 起点去重过狠**：
  现行口径下 3 年 64 币只找到 13 个真局起点，样本根本不够训练。

本脚本只做计数（不训练），对比 6 种口径下"真局"的样本量，找出哪种口径
  既保持"操盘局"语义、又有足够样本可训。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_realcount.txt')
WMAX = 288

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_realcount · "真局"到底有多少个？（6 种口径普查）')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

def year_of(ms):
    return pd.Timestamp(int(ms), unit='ms').year

def scan(name, horizon, gain_min, need_low, dedup, dd_min=None):
    """扫一个口径，返回事件列表"""
    ev = []
    for (sym, ot, c, h, l) in RAW:
        n = len(c)
        if n < horizon + 96: continue
        fmax = pd.Series(h).rolling(horizon, min_periods=1).max().shift(-horizon).values
        base = l if need_low else c
        ratio = fmax / np.maximum(base, 1e-12) - 1.0
        cand = np.flatnonzero(np.isfinite(ratio) & (ratio >= gain_min))
        if need_low:
            W = np.lib.stride_tricks.sliding_window_view(np.pad(l, (96, 96), constant_values=np.inf), 193)
            is_low = l <= W.min(axis=1) + 1e-12
            cand = cand[is_low[cand]]
        last = -10 ** 9
        for t0 in cand:
            if dedup and t0 - last < horizon: continue
            i1 = t0 + int(np.argmax(h[t0:min(n, t0 + horizon)]))
            if dd_min is not None:
                k2 = min(n, i1 + 97)
                if k2 - i1 < 5: continue
                dd = 1.0 - l[i1:k2].min() / max(h[i1], 1e-12)
                if dd < dd_min: continue
            ev.append((sym, int(ot[t0]), year_of(ot[t0]), float(ratio[t0])))
            last = i1
    return ev

CASES = [
    ('A 现行(贴低点&288根涨>=100%&峰后崩>=50%)', 288, 1.00, True,  True,  0.50),
    ('B 同A但不去重',                          288, 1.00, True,  False, 0.50),
    ('C 288根涨>=100%(不要求贴低点)',           288, 1.00, False, True,  None),
    ('D 48根(4h)涨>=100%',                     48, 1.00, False, True,  None),
    ('E 48根(4h)涨>=50%',                      48, 0.50, False, True,  None),
    ('F 288根涨>=50%',                         288, 0.50, False, True,  None),
    ('G 288根涨>=50% 且 峰后崩>=30%',           288, 0.50, False, True,  0.30),
]

log('')
log('=' * 100)
log('【各口径下的事件数】')
log('=' * 100)
log(f'  {"口径":<42} {"事件数":>8} {"涉及币":>7} {"涨幅中位":>9}   逐年分布')
rows = []
for (nm, hz, gm, nl, dd, ddm) in CASES:
    ev = scan(nm, hz, gm, nl, dd, ddm)
    if not ev:
        log(f'  {nm:<42} {0:>8}'); continue
    syms = set(e[0] for e in ev)
    gains = np.array([e[3] for e in ev])
    yrs = {}
    for e in ev: yrs[e[2]] = yrs.get(e[2], 0) + 1
    ystr = ' '.join(f'{y}:{c}' for y, c in sorted(yrs.items()))
    log(f'  {nm:<42} {len(ev):>8,} {len(syms):>7} {100*np.median(gains):>8.1f}%   {ystr}')
    rows.append((nm, len(ev), len(syms)))
    del ev; gc.collect()

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · A 口径（现行）= 训练"真局"档时实际用的样本来源')
log('  · 若 A 远小于 D/E，说明"贴低点 + 峰后崩 50%"这两个附加条件把绝大多数'
    '真实拉升事件都排除掉了，不是市场上没有真局')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
