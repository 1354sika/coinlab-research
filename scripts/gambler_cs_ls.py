# -*- coding: utf-8 -*-
"""gambler_cs_ls · 截面多空 × 分化度条件（唯一还没试的变体）

问题：
  · 真截面多空（按币排名：多 top / 空 bottom）已做过，全负（backtest_ls）
  · 但「只在截面分化度 disp 高时开仓」这个条件，从未试过

设计：
  · 因子：过去 MOM=240 根（20h）动量，逐币排名
  · 组合：多 top 20% / 空 bottom 20%，等权、美元中性（各 50% 本金）
  · 条件：disp240（60 币 48 根动量的截面 std）分 低/中/高 三档
  · 持有 H=12 根，每 12 根决策，入场 t+1 开盘，出场 t+1+H 收盘
  · 成本：组合往返 0.04%（两腿按 50% 资金加权）
  · 对照：--shuffle 按天块重排 log 收益

用法：
  python -X utf8 gambler_cs_ls.py
  python -X utf8 gambler_cs_ls.py --shuffle
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CS_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CS_NSYM', '60'))
MOM = int(os.environ.get('CS_MOM', '240'))
H = int(os.environ.get('CS_H', '12'))
STEP = int(os.environ.get('CS_STEP', '12'))
Q = float(os.environ.get('CS_Q', '0.2'))
COST = 0.0004
BPD = 288
SHUF = '--shuffle' in sys.argv
TAG = 'SHUF' if SHUF else 'REAL'
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def shuffle_panel(C, rng):
    T, S = C.shape
    r = np.zeros_like(C)
    ok = np.isfinite(C[1:]) & np.isfinite(C[:-1]) & (C[1:] > 0) & (C[:-1] > 0)
    r[1:][ok] = np.log(C[1:][ok] / C[:-1][ok])
    r = np.nan_to_num(r)
    nb = T // BPD
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * BPD, (b + 1) * BPD) for b in order]
                         + [np.arange(nb * BPD, T)]).astype(int)
    v0 = C[0][np.isfinite(C[0])]
    c0 = np.full(S, v0.mean() if len(v0) else 1.0)
    return np.exp(np.log(np.maximum(c0, 1e-12)) + np.cumsum(r[idx], axis=0))


def main():
    t0 = time.time()
    say('=' * 100)
    say(f'gambler_cs_ls · 截面多空 × 分化度条件 [{TAG}]')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  因子 mom{MOM}  持有 {H}  成本 {COST:.2%}')
    say('=' * 100)
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    O = df.pivot_table(index='t', columns='symbol', values='open').sort_index()[syms].values.astype(np.float64)
    C = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms].values.astype(np.float64)
    O[~np.isfinite(O) | (O <= 0)] = np.nan
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    tidx = df.pivot_table(index='t', columns='symbol', values='close').sort_index().index
    rng = np.random.default_rng(20261020)
    if SHUF:
        C = shuffle_panel(C, rng)
        O = C.copy()
        say('  ⚠ 已打乱：按天块重排 log 收益')
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    mom = np.full((T, S), np.nan)
    mom[MOM:] = C[MOM:] / C[:-MOM] - 1.0
    disp = np.nanstd(mom, axis=1)
    say(f'  因子/分化度算完  {time.time()-t0:.0f}s')

    starts = np.arange(MOM, T - H - 1, STEP)
    rows = []
    for t in starts:
        if not np.isfinite(disp[t]):
            continue
        v = mom[t]
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(v[ok])]
        k = max(1, int(len(order) * Q))
        bot = order[:k]; top = order[-k:]
        # 入场 t+1 开盘，出场 t+1+H 收盘
        if not (np.isfinite(O[t + 1, top]).all() and np.isfinite(C[t + 1 + H, top]).all()
                and np.isfinite(O[t + 1, bot]).all() and np.isfinite(C[t + 1 + H, bot]).all()):
            continue
        lr = np.mean(C[t + 1 + H, top] / O[t + 1, top] - 1.0)
        sr = np.mean(O[t + 1, bot] / C[t + 1 + H, bot] - 1.0)
        net = 0.5 * lr + 0.5 * sr - COST
        rows.append((t, disp[t], lr, sr, net, len(top)))
    A = np.asarray(rows, np.float64)
    n = len(A)
    say(f'  决策点 {n:,}  {time.time()-t0:.0f}s')

    split = int(n * 0.7)
    # disp 分位用训练段
    qs = np.quantile(A[:split, 1], [1/3, 2/3])
    lab = np.searchsorted(qs, A[:, 1], side='right')
    names = ['低分化', '中分化', '高分化']

    say('\n【一】按分化度分档（训练段分位）')
    say(f'  {"档":<8}{"n":>10}{"多头均值":>12}{"空头均值":>12}{"组合净均值":>12}{"中位":>10}{"胜率":>8}')
    for b in range(3):
        for seg, nm in ((np.arange(n) < split, '训练'), (np.arange(n) >= split, '测试')):
            m = (lab == b) & seg
            if m.sum() < 50:
                continue
            say(f'  {names[b]+"-"+nm:<8}{m.sum():>10,}{A[m,2].mean():>11.4%}{A[m,3].mean():>11.4%}'
                f'{A[m,4].mean():>11.4%}{np.median(A[m,4]):>9.4%}{(A[m,4]>0).mean():>7.1%}')

    say('\n【二】只用「高分化」档，按自然季度（测试段）')
    qt = pd.PeriodIndex(pd.to_datetime(tidx[A[:, 0].astype(int)]), freq='Q')
    te = np.arange(n) >= split
    for q in sorted(set(qt[te])):
        m = (qt == q) & te & (lab == 2)
        if m.sum() < 30:
            continue
        say(f'  {q}  n={m.sum():,}  组合净 {A[m,4].mean():+.4%}  中位 {np.median(A[m,4]):+.4%}  胜率 {(A[m,4]>0).mean():.1%}')

    say('\n【三】全样本（所有分化档）')
    for seg, nm in ((np.arange(n) < split, '训练'), (np.arange(n) >= split, '测试')):
        m = seg
        say(f'  {nm}  n={m.sum():,}  组合净 {A[m,4].mean():+.4%}  中位 {np.median(A[m,4]):+.4%}  胜率 {(A[m,4]>0).mean():.1%}')

    say('\n【判据】高分化档扣费后是否为正 + 季度是否稳定 + SHUF 是否更差')
    say(f'TOTAL {time.time()-t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'gambler_cs_ls_{TAG}.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT) + '\n')
    print(f'\n已写 {p}')


if __name__ == '__main__':
    main()
