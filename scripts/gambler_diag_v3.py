# -*- coding: utf-8 -*-
"""
gambler_diag_v3.py · 诊断 v3 为什么亏

v3 全量 60 币结果：42,388 笔，单笔 −0.0297%（t=−3.61），资金 0.1036。
在调参之前必须先回答三个问题，否则就是瞎调：

  ① **阈值有没有意义**：v3 用 thr ∈ [0.55, 0.70] 作用在 [0,1] 的合成分数上。
     如果分数实际全挤在 0.45~0.55，那 thr=0.6 就是"几乎从不开仓"或"几乎总开仓"，
     阈值形同虚设。先看分数分布。
  ② **信号还在不在**：样本外把分数分 10 档，看每档未来收益是否仍单调。
     单调 ⇒ 信号活着，是用法（阈值/持有/仓位）错了；
     不单调或反向 ⇒ 信号在样本外已衰减，调参无用。
  ③ **是不是某几段扛全场**：样本外分 4 段，逐段看最高档的收益方向。

输出：evidence/gambler_diag_v3.txt
用法：python -X utf8 gambler_diag_v3.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_adaptive_v3 import scale_scores, SCALES, SIGMA

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('AD_NSYM', '60'))
TRAIN = 0.70
TAUS = [183.0, 300.0, 474.0, 761.0]
H_LIST = [60, 240, 480]
NSEG = 4
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def mix_score(scores, tau):
    ln = np.log(np.array(SCALES, dtype=np.float64))
    w = np.exp(-0.5 * ((ln - np.log(tau)) / SIGMA) ** 2)
    w /= w.sum()
    T, S = scores[SCALES[0]].shape
    out = np.zeros((T, S), dtype=np.float32)
    for wi, sc in zip(w, SCALES):
        if wi > 1e-4:
            out += float(wi) * scores[sc]
    return out


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_diag_v3 · 诊断：阈值是否有效 / 信号是否还在 / 是否只靠某几段')
    say(f'  币 {NSYM} | 训练定界 {TRAIN:.0%} | τ 候选 {TAUS} | H {H_LIST}')
    say('=' * 100)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    C = piv.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    T, S = C.shape
    dow = piv.index.dayofweek.values
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    say('  计算多尺度牌 ...')
    SR = scale_scores(C, dow)
    say(f'  完成  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    fwd = {}
    for H in H_LIST:
        f = np.full((T, S), np.nan)
        ok = np.isfinite(C[H:]) & np.isfinite(C[:-H])
        v = np.full_like(C[H:], np.nan)
        v[ok] = C[H:][ok] / C[:-H][ok] - 1
        f[:-H] = v
        fwd[H] = f

    for tau in TAUS:
        sc = mix_score(SR, tau)
        say('')
        say('=' * 100)
        say(f'【τ = {tau:.0f} 分（σ={SIGMA}）】')
        s_tr = sc[:cut]
        s_tr = s_tr[np.isfinite(s_tr)]
        s_te = sc[cut:]
        s_te = s_te[np.isfinite(s_te)]
        qs = [1, 10, 50, 90, 99]
        say('  ① 分数分布（训练段 / 样本外）')
        say('     分位     ' + ''.join(f'p{q:<8}' for q in qs))
        say('     训练段   ' + ''.join(f'{np.percentile(s_tr, q):<9.4f}' for q in qs))
        say('     样本外   ' + ''.join(f'{np.percentile(s_te, q):<9.4f}' for q in qs))
        say(f'     训练段落在 [0.55,0.70] 之外的比例：'
            f'{(s_tr < 0.55).mean() * 100:.1f}% / {(s_tr > 0.70).mean() * 100:.1f}%'
            f'  ⇒ 阈值区间的样本占比 {(np.sum((s_tr >= 0.55) & (s_tr <= 0.70)) / len(s_tr)) * 100:.1f}%')

        for H in H_LIST:
            say(f'  ② 分 10 档的未来收益（H={H} 分，样本外）')
            f = fwd[H][cut:]
            ss = sc[cut:]
            ok = np.isfinite(f) & np.isfinite(ss)
            fv, sv = f[ok], ss[ok]
            edges = np.quantile(sv, np.linspace(0, 1, 11))
            line = []
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                m = (sv >= lo) & (sv <= hi) if i == 9 else (sv >= lo) & (sv < hi)
                if m.sum() < 1000:
                    line.append(np.nan)
                else:
                    line.append(fv[m].mean() * 100)
            say('     档位      ' + ''.join(f'{i + 1:<9}' for i in range(10)))
            say('     均值%     ' + ''.join(f'{x:<+9.3f}' if np.isfinite(x) else 'nan      ' for x in line))
            hi_m = fv[sv >= edges[8]]
            lo_m = fv[sv <= edges[1]]
            say(f'     顶档−底档 = {(hi_m.mean() - lo_m.mean()) * 100:+.3f}%   '
                f'顶档 {hi_m.mean() * 100:+.3f}%  底档 {lo_m.mean() * 100:+.3f}%')

        say('  ③ 样本外分 4 段：顶档（前 20%）均值')
        seg_len = (T - cut) // NSEG
        vals = []
        for i in range(NSEG):
            lo = cut + i * seg_len
            hi = cut + (i + 1) * seg_len if i < NSEG - 1 else T
            ss = sc[lo:hi]
            ff = fwd[60][lo:hi]
            ok = np.isfinite(ss) & np.isfinite(ff)
            ss, ff = ss[ok], ff[ok]
            if len(ss) < 1000:
                vals.append(np.nan)
                continue
            q80 = np.quantile(ss, 0.8)
            vals.append(ff[ss >= q80].mean() * 100)
        say('     段均值%   ' + ''.join(f'{x:<+9.3f}' if np.isfinite(x) else 'nan      ' for x in vals))

    say('')
    say('=' * 100)
    say('【读法】')
    say('  · 分数分布挤在 0.5 附近 + [0.55,0.70] 样本占比 < 5% ⇒ 阈值没意义，v3 的"开仓"其实是极端少数点')
    say('  · 分 10 档单调递增 ⇒ 信号活着，问题在用法；不单调/反向 ⇒ 信号在样本外已衰减')
    say('  · 分段符号全负 ⇒ 不是某一段拖累，是整体方向错了')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_diag_v3.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))


if __name__ == '__main__':
    main()
