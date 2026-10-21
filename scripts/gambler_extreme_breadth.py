# -*- coding: utf-8 -*-
"""
gambler_extreme_breadth.py · 极端市场宽度下的收益（只在人群极度一致时下注）

上一轮结论：breadth60（全市场 60 分钟上涨占比）方向稳定为负（人群一致上涨 → 后续跌），
但普通分位差只有 0.05%/笔，扣 maker 费 0.04% 后基本归零。

所以这个脚本问的是赌徒真正该问的问题：
    如果只在**极度一致**的时刻下注（宽度 ≥ 0.9 或 ≤ 0.1），差值会不会放大到扣费后仍有肉？

同时给"精准避开必输局"的答案：
    宽度极端时的未来收益分布（均值/中位/胜率/最差尾部），与全样本对比。

判据：
    可交易 = 极端条件下的 |均值| ≥ 0.20%（覆盖 taker 来回 0.20%）或 ≥ 0.10%（覆盖 maker 来回 0.04% 后仍有 0.06%）

用法：python -X utf8 gambler_extreme_breadth.py
输出：evidence/gambler_extreme_breadth.txt / .csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CROSS_N', '60'))
K = 60
H_LIST = [15, 30, 60, 120]
THR = [0.60, 0.70, 0.80, 0.90, 0.95]
NSEG = 4

TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def mom(C, k):
    """过去 k 根收益率，nan 安全（缺价不参与）"""
    out = np.full_like(C, np.nan)
    a = C[k:]
    b = C[:-k]
    ok = np.isfinite(a) & np.isfinite(b)
    v = np.full_like(a, np.nan)
    v[ok] = a[ok] / b[ok] - 1
    out[k:] = v
    return out


def fwd_ret(C, H):
    """未来 H 根收益率，nan 安全"""
    T, S = C.shape
    out = np.full((T, S), np.nan)
    a = C[H:]
    b = C[:-H]
    ok = np.isfinite(a) & np.isfinite(b)
    v = np.full_like(a, np.nan)
    v[ok] = a[ok] / b[ok] - 1
    out[:-H] = v
    return out


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_extreme_breadth · 极端市场宽度下注')
    say(f'  breadth 回看 {K} 根 1m | 阈值 {THR} | 持有 {H_LIST} | 币 {NSYM}')
    say('=' * 100)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    C = piv.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币（保留缺失：不填 0、不填未来价）  {time.time() - t0:.0f}s')

    Mk = mom(C, K)
    valid = np.isfinite(Mk)
    nv = valid.sum(axis=1)
    ups = ((Mk > 0) & valid).sum(axis=1)
    B = np.where(nv >= 30, ups / np.maximum(nv, 1), np.nan)
    say(f'  宽度分布： 中位 {np.nanmedian(B):.3f}  均值 {np.nanmean(B):.3f}  '
        f'p10 {np.nanpercentile(B, 10):.3f}  p90 {np.nanpercentile(B, 90):.3f}')

    rows = []
    for H in H_LIST:
        fwd = fwd_ret(C, H)
        say('')
        say(f'【H={H} 分钟】  {"宽度条件":<16}{"样本":>9}{"均值":>10}{"中位":>10}{"胜率":>8}{"最差5%":>10}')
        # 全样本基线
        allv = fwd[np.isfinite(fwd)]
        say(f'  {"全样本":<16}{len(allv):>9,}{allv.mean() * 100:>+9.3f}%'
            f'{np.median(allv) * 100:>+9.3f}%{(allv > 0).mean() * 100:>7.1f}%'
            f'{np.percentile(allv, 5) * 100:>+9.3f}%')
        rows.append(dict(H=H, cond='ALL', n=len(allv), mean=allv.mean() * 100,
                         med=np.median(allv) * 100, win=(allv > 0).mean() * 100,
                         p5=np.percentile(allv, 5) * 100))
        for q in THR:
            for side, mask_b in (('宽≥%.2f' % q, B >= q), ('宽≤%.2f' % (1 - q), B <= 1 - q)):
                m = np.repeat(mask_b[:, None], S, axis=1)
                v = fwd[np.isfinite(fwd) & m]
                if len(v) < 200:
                    continue
                say(f'  {side:<16}{len(v):>9,}{v.mean() * 100:>+9.3f}%'
                    f'{np.median(v) * 100:>+9.3f}%{(v > 0).mean() * 100:>7.1f}%'
                    f'{np.percentile(v, 5) * 100:>+9.3f}%')
                rows.append(dict(H=H, cond=side, n=len(v), mean=v.mean() * 100,
                                 med=np.median(v) * 100, win=(v > 0).mean() * 100,
                                 p5=np.percentile(v, 5) * 100))

    say('')
    say('=' * 100)
    say('【二】分段稳定性（宽 ≥ 0.90 做空一侧，H=60）')
    H = 60
    fwd = fwd_ret(C, H)
    seg_len = T // NSEG
    say(f'  {"段":>3}{"样本":>9}{"均值":>10}{"中位":>10}{"胜率(跌)":>11}')
    for i in range(NSEG):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < NSEG - 1 else T
        m = np.repeat((B >= 0.90)[:, None], S, axis=1)[lo:hi]
        v = fwd[lo:hi][np.isfinite(fwd[lo:hi]) & m]
        if len(v) < 100:
            say(f'  {i + 1:>3}{len(v):>9,}   样本不足')
            continue
        say(f'  {i + 1:>3}{len(v):>9,}{v.mean() * 100:>+9.3f}%{np.median(v) * 100:>+9.3f}%'
            f'{(v < 0).mean() * 100:>10.1f}%')

    say('')
    say('=' * 100)
    say('【三】判据')
    say('  · 只做空一侧（宽度高 = 人群一致看涨）：均值应为负；只做多一侧应为正')
    say('  · 可交易门槛：|均值| ≥ 0.20%（覆盖 taker 来回）或 ≥ 0.10%（覆盖 maker 来回后仍有 0.06%）')
    df_out = pd.DataFrame(rows)
    best = df_out[df_out.cond != 'ALL'].reindex(
        df_out[df_out.cond != 'ALL'].mean().abs().sort_values(ascending=False).index).head(6)
    say('  · 绝对值最大的 6 个条件：')
    for _, r in best.iterrows():
        cover = '✅ 覆盖 taker' if abs(r['mean']) >= 0.20 else ('⚠ 只够 maker' if abs(r['mean']) >= 0.10 else '❌ 扣费后归零')
        say(f'    H={int(r.H):>3} {r.cond:<16} 均值 {r["mean"]:+.3f}%  n={int(r.n):,}  {cover}')

    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_extreme_breadth.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    df_out.to_csv(os.path.join(OUT, 'gambler_extreme_breadth.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
