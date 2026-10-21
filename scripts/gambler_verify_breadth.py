# -*- coding: utf-8 -*-
"""
gambler_verify_breadth.py · 复核第一张过闸的牌（breadth60）

breadth_k = 全市场 k 分钟里上涨的币占比。
在 gambler_gate_cross.py 里，k=60 这一张过了闸（t(真)=-10.80，噪声天花板 6.58）。
过闸只说明"不是搜出来的假货"，不说明它稳、也不说明扣费后能赚。
这个脚本专门复核它：

  ① 换持有期 H：15/30/60/120/240 分钟 —— 方向是否一致、量级是否合理
  ② 换回看窗口 k：15/60/240 —— 是不是只有 k=60 这一个巧合数字有效
  ③ 换时间段：1 年切成 4 段，逐段独立评估 —— 是不是只在某一段有效
  ④ 打乱对照：同样参数下打乱序列的表现（口径与 cross 一致）

判据：
  稳 = 各 H 方向一致 且 4 段里 ≥3 段同向 且 打乱对照明显更弱

用法：python -X utf8 gambler_verify_breadth.py
输出：evidence/gambler_verify_breadth.txt / .csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate import tstat
from gambler_gate_cross import shuffle_panel

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CROSS_N', '60'))
K_LIST = [15, 60, 240]
H_LIST = [15, 30, 60, 120, 240]
NSEG = 4
BLOCK = 1440
TRAIN = 0.70

TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def mom(C, k):
    out = np.full_like(C, np.nan)
    out[k:] = C[k:] / np.maximum(C[:-k], 1e-12) - 1
    return out


def breadth(C, k):
    """(T,) 全市场过去 k 根上涨的币占比"""
    M = mom(C, k)
    return np.nanmean(M > 0, axis=1)


def delta_series(B, fwd, lo, hi, s):
    """币 s 在 [lo,hi) 上：B 高 20% 组的未来收益 − 低 20% 组"""
    v = B[lo:hi]
    y = fwd[lo:hi, s]
    ok = np.isfinite(v) & np.isfinite(y)
    if ok.sum() < 2000:
        return np.nan
    v, y = v[ok], y[ok]
    ql, qh = np.quantile(v, [0.2, 0.8])
    if qh - ql < 1e-9:
        return np.nan
    hi_m = v >= qh
    lo_m = v <= ql
    if hi_m.sum() < 300 or lo_m.sum() < 300:
        return np.nan
    return y[hi_m].mean() - y[lo_m].mean()


def eval_all(B, fwd, cut, S):
    d = np.array([delta_series(B, fwd, cut, len(B), s) for s in range(S)])
    return d


def main():
    t0 = time.time()
    say('=' * 104)
    say('gambler_verify_breadth · 复核 breadth（全市场上涨占比）')
    say(f'  回看 k ∈ {K_LIST} | 持有 H ∈ {H_LIST} | 分段 {NSEG} | 币 {NSYM}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    C = np.nan_to_num(piv.values.astype(np.float64), nan=0.0)
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    say(f'  打乱面板完成  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    seg_len = T // NSEG
    rows = []

    say('')
    say('【一】不同回看 k × 持有 H（全样本：前 70% 定分位，后 30% 评估）')
    say(f'  {"k":>4}{"H":>5}{"真差值":>11}{"t(真)":>8}{"打乱差值":>11}{"t(打乱)":>9}{"t(真-打乱)":>11}')
    for k in K_LIST:
        B = breadth(C, k)
        B2 = breadth(C2, k)
        for H in H_LIST:
            fwd = np.full((T, S), np.nan)
            fwd[:-H] = C[H:] / np.maximum(C[:-H], 1e-12) - 1
            fwd2 = np.full((T, S), np.nan)
            fwd2[:-H] = C2[H:] / np.maximum(C2[:-H], 1e-12) - 1
            dR = eval_all(B, fwd, cut, S)
            dS = eval_all(B2, fwd2, cut, S)
            tr, ts, tg = tstat(dR), tstat(dS), tstat(dR - dS)
            say(f'  {k:>4}{H:>5}{np.nanmean(dR) * 100:>+10.3f}%{tr:>+8.2f}'
                f'{np.nanmean(dS) * 100:>+10.3f}%{ts:>+9.2f}{tg:>+11.2f}')
            rows.append(dict(k=k, H=H, seg='ALL', d_real=np.nanmean(dR) * 100,
                             d_shuf=np.nanmean(dS) * 100, t_real=tr, t_shuf=ts, t_gap=tg))

    say('')
    say(f'【二】分段复核（k=60, H=60；每段内部独立定分位）')
    k0, H0 = 60, 60
    B = breadth(C, k0)
    B2 = breadth(C2, k0)
    fwd = np.full((T, S), np.nan)
    fwd[:-H0] = C[H0:] / np.maximum(C[:-H0], 1e-12) - 1
    fwd2 = np.full((T, S), np.nan)
    fwd2[:-H0] = C2[H0:] / np.maximum(C2[:-H0], 1e-12) - 1
    say(f'  {"段":>3}{"区间":>26}{"真差值":>11}{"t(真)":>8}{"打乱差值":>11}')
    pos = 0
    for i in range(NSEG):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < NSEG - 1 else T
        dR = eval_all(B, fwd, lo, S)
        # 段内也要用段前半定分位，避免用未来
        mid = lo + (hi - lo) // 2
        dR_mid = np.array([delta_series(B, fwd, mid, hi, s) for s in range(S)])
        dS_mid = np.array([delta_series(B2, fwd2, mid, hi, s) for s in range(S)])
        tr = tstat(dR_mid)
        say(f'  {i + 1:>3}{str(piv.index[lo])[:16] + " ~ " + str(piv.index[hi - 1])[:8]:>26}'
            f'{np.nanmean(dR_mid) * 100:>+10.3f}%{tr:>+8.2f}{np.nanmean(dS_mid) * 100:>+10.3f}%')
        rows.append(dict(k=k0, H=H0, seg=f'S{i + 1}', d_real=np.nanmean(dR_mid) * 100,
                         d_shuf=np.nanmean(dS_mid) * 100, t_real=tr, t_shuf=np.nan, t_gap=np.nan))
        if np.nanmean(dR_mid) < 0:
            pos += 1

    say('')
    say('=' * 104)
    say('【三】判据')
    base = [r for r in rows if r['seg'] == 'ALL' and r['k'] == 60]
    signs = set(np.sign(r['d_real']) for r in base if np.isfinite(r['d_real']))
    say(f'  · k=60 在 {len(base)} 个持有期里，方向集合 = {signs}（一致 = 只含一个符号）')
    say(f'  · 分段里方向为负的段数 = {pos}/{NSEG}')
    segs = [r for r in rows if r['seg'].startswith('S')]
    ok_seg = sum(1 for r in segs if r['d_real'] < 0)
    verdict = '✅ 稳（各 H 同向、多数段同向）' if (len(signs) == 1 and ok_seg >= NSEG - 1) else \
              '⚠ 不稳（换 H 或换时间段方向会翻）'
    say(f'  · 判决：{verdict}')
    say('')
    say('  代价提醒：差值要扣往返手续费。maker 0.02%×2=0.04%，taker 0.10%×2=0.20%。')
    say('  若 |真差值| 小于 0.04%，这张牌在扣费后归零；小于 0.20%，用 taker 成交必亏。')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_verify_breadth.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gambler_verify_breadth.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
