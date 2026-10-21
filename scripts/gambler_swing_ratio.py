# -*- coding: utf-8 -*-
"""gambler_swing_ratio · 「低买高卖」的机会，跟历史波动有关还是跟「来回摆动」有关？

要回答：
  · 按「历史波动」分档，未来摆幅是否更大？
  · 按「摆动比 Σ|r|/|Σr|」分档，未来摆幅是否更大？
  · 按「VR 方差比」分档，未来摆幅是否更大？
  · 三者与未来摆幅的秩相关，谁更强？

口径（无前视）：
  · 事前指标只用过去 288 根（24 小时）
  · 未来摆幅 = log(max(high[t+1..t+H]) / min(low[t+1..t+H]))，H=12 根（60 分钟）
  · 分档边界用训练段（前 70%）分位，测试段沿用
  · 成本 maker 往返 0.04%

用法：
  python -X utf8 gambler_swing_ratio.py
  python -X utf8 gambler_swing_ratio.py --shuffle
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('SR_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('SR_NSYM', '60'))
STEP = int(os.environ.get('SR_STEP', '12'))
H = int(os.environ.get('SR_H', '12'))
LOOK = int(os.environ.get('SR_LOOK', '288'))
COST = 0.0004
SHUF = '--shuffle' in sys.argv
TAG = 'SHUF' if SHUF else 'REAL'
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def shuffle_panel(C, rng, bpd=288):
    T, S = C.shape
    r = np.zeros_like(C)
    ok = np.isfinite(C[1:]) & np.isfinite(C[:-1]) & (C[1:] > 0) & (C[:-1] > 0)
    r[1:][ok] = np.log(C[1:][ok] / C[:-1][ok])
    r = np.nan_to_num(r)
    nb = T // bpd
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * bpd, (b + 1) * bpd) for b in order]
                         + [np.arange(nb * bpd, T)]).astype(int)
    v0 = C[0][np.isfinite(C[0])]
    c0 = np.full(S, v0.mean() if len(v0) else 1.0)
    return np.exp(np.log(np.maximum(c0, 1e-12)) + np.cumsum(r[idx], axis=0))


def spearman(a, b):
    a = pd.Series(a); b = pd.Series(b)
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def main():
    t0 = time.time()
    say('=' * 100)
    say(f'gambler_swing_ratio · 低买高卖的机会来自「波动大」还是「来回摆动」？ [{TAG}]')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  回看 {LOOK} 根  未来 {H} 根  maker {COST:.2%}')
    say('=' * 100)
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    pv = df.pivot_table(index='t', columns='symbol', values=['high', 'low', 'close']).sort_index()
    Hi = pv['high'][syms].values.astype(np.float64)
    Lo = pv['low'][syms].values.astype(np.float64)
    C = pv['close'][syms].values.astype(np.float64)
    tidx = pv.index
    rng = np.random.default_rng(20261020)
    if SHUF:
        C = shuffle_panel(C, rng)
        Hi = C.copy(); Lo = C.copy()
        say('  ⚠ 已打乱：按天块重排 log 收益')
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    # ---- 事前指标（只用过去 LOOK 根） ----
    r1 = np.zeros((T, S))
    ok = np.isfinite(C[1:]) & np.isfinite(C[:-1]) & (C[1:] > 0) & (C[:-1] > 0)
    r1[1:][ok] = np.log(C[1:][ok] / C[:-1][ok])
    r1 = np.nan_to_num(r1)
    d1 = pd.DataFrame(r1)
    vol288 = d1.rolling(LOOK, min_periods=LOOK // 2).std().values
    path = d1.abs().rolling(LOOK, min_periods=LOOK // 2).sum().values
    net = d1.rolling(LOOK, min_periods=LOOK // 2).sum().values
    swing = path / np.maximum(np.abs(net), 1e-6)
    # VR(12)：用过去 LOOK 根
    r12 = np.zeros((T, S)); r12[12:] = np.log(np.maximum(C[12:], 1e-12) / np.maximum(C[:-12], 1e-12))
    vr = pd.DataFrame(r12).rolling(LOOK, min_periods=LOOK // 2).var().values / \
         np.maximum(12.0 * pd.DataFrame(r1).rolling(LOOK, min_periods=LOOK // 2).var().values, 1e-12)
    say(f'  事前指标算完  {time.time()-t0:.0f}s')

    # ---- 未来摆幅（事后上界，仅作为「机会大小」度量） ----
    Hf = pd.DataFrame(Hi).rolling(H, min_periods=H).max().shift(-H).values
    Lf = pd.DataFrame(Lo).rolling(H, min_periods=H).min().shift(-H).values
    rng_fwd = np.log(np.maximum(Hf, 1e-12) / np.maximum(Lf, 1e-12))
    up_fwd = np.log(np.maximum(Hf, 1e-12) / np.maximum(C, 1e-12))
    say(f'  未来摆幅算完  {time.time()-t0:.0f}s')

    # ---- 决策点 ----
    starts = np.arange(LOOK, T - H - 1, STEP)
    rows = []
    for s in range(S):
        for t in starts:
            if not (np.isfinite(vol288[t, s]) and np.isfinite(swing[t, s]) and np.isfinite(vr[t, s])
                    and np.isfinite(rng_fwd[t, s]) and np.isfinite(up_fwd[t, s])):
                continue
            rows.append((t, vol288[t, s], swing[t, s], vr[t, s], rng_fwd[t, s], up_fwd[t, s]))
    A = np.asarray(rows, np.float64)
    n = len(A)
    say(f'  决策点 {n:,}  {time.time()-t0:.0f}s')

    split = int(n * 0.7)
    tr, te = np.arange(split), np.arange(split, n)

    # ---- 分档表 ----
    def table(x, name):
        qs = np.quantile(x[tr], [0.2, 0.4, 0.6, 0.8])
        lab = np.searchsorted(qs, x, side='right')
        say(f'\n  【{name}】按训练段分位分 5 档')
        say(f'    {"档":<6}{"n":>10}{"摆幅中位":>12}{"买后上界中位":>14}{"摆幅>3×成本占比":>16}')
        for b in range(5):
            m = (lab == b) & (np.arange(n) >= split)
            if m.sum() < 100:
                continue
            rr = A[m, 4]; uu = A[m, 5]
            say(f'    {b:<6}{m.sum():>10,}{np.median(rr):>11.3%}{np.median(uu):>13.3%}{np.mean(rr > 3*COST):>15.1%}')
        # 训练段也报一遍
        mtr = np.arange(n) < split
        say(f'    (训练段整体) 摆幅中位 {np.median(A[mtr & (lab>=0),4]):.3%}')

        corr_tr = spearman(x[tr], A[tr, 4])
        corr_te = spearman(x[te], A[te, 4])
        say(f'    Spearman(指标, 未来摆幅)：训练 {corr_tr:+.3f}  测试 {corr_te:+.3f}')

    table(A[:, 1], '历史波动 vol288')
    table(A[:, 2], '摆动比 Σ|r|/|Σr|')
    table(A[:, 3], 'VR12（方差比）')

    say('\n【判读】')
    say('  · 波动档越高 ≠ 摆幅越大 ⇒ 「波动大」不是低买高卖的充分条件')
    say('  · 摆动比/VR 的 Spearman 若明显高于 vol288 ⇒ 条件应取「来回摆动」而非「波动」')
    say(f'TOTAL {time.time()-t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'gambler_swing_ratio_{TAG}.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT) + '\n')
    print(f'\n已写 {p}')


if __name__ == '__main__':
    main()
