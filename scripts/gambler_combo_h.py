# -*- coding: utf-8 -*-
"""
gambler_combo_h.py · 合成分数 × 多持有期：信号能不能累积

已知（gambler_combo.py，H=1 小时）：
  三张牌合成后 5 档单调递增：-0.049 → -0.034 → +0.007 → +0.040 → +0.079 %
  最高档单边 +0.079%/笔，打乱对照 -0.010% —— 信号是真的，但扣 maker 0.04% 后只剩 0.04%。

这里问一个关键问题：
  手续费是**按笔**收的，不是按时间收的。
  如果同一笔持有 4 小时 / 8 小时 / 1 天，信号会不会累积到 0.3~1.0%？
  若能，成本占比就从 50% 降到 5%，这套信号才真的能用。

做法：
  score 不变（disp240+ / breadth60− / dow_cos−），对 H ∈ {60,240,480,1440} 各算一遍：
  5 档均值 + 最高档 − 最低档 + 最高档扣费后净值（maker 0.04% / taker 0.20%）

用法：python -X utf8 gambler_combo_h.py
输出：evidence/gambler_combo_h.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate_cross import mom, shuffle_panel
from gambler_combo import rank01, build_score

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CROSS_N', '60'))
H_LIST = [60, 240, 480, 1440]
TRAIN = 0.70
NSEG = 4
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def fwd_of(C, H):
    T, S = C.shape
    out = np.full((T, S), np.nan)
    ok = np.isfinite(C[H:]) & np.isfinite(C[:-H])
    v = np.full_like(C[H:], np.nan)
    v[ok] = C[H:][ok] / C[:-H][ok] - 1
    out[:-H] = v
    return out


def main():
    t0 = time.time()
    say('=' * 96)
    say('gambler_combo_h · 合成分数 × 多持有期（信号累积 vs 手续费）')
    say(f'  H ∈ {H_LIST} 分钟 | 币 {NSYM} | 训练定分位 {TRAIN:.0%}')
    say('=' * 96)

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

    score = build_score(C, dow)
    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    score2 = build_score(C2, dow)
    say(f'  score 完成（真/打乱）  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    say('')
    say(f'{"H(分)":>6}{"档":>4}{"样本":>9}{"均值":>10}{"中位":>10}{"胜率":>7}{"打乱均值":>10}')
    rows = []
    for H in H_LIST:
        fwd = fwd_of(C, H)
        fwd2 = fwd_of(C2, H)
        mkt = np.nanmean(fwd, axis=1)
        mkt2 = np.nanmean(fwd2, axis=1)
        # 注意：H 越大，末尾 H 根无 fwd，评估段也要相应缩短
        end = T - H
        qs = np.nanquantile(score[:cut], [0.2, 0.4, 0.6, 0.8])
        for i in range(5):
            lo = -np.inf if i == 0 else qs[i - 1]
            hi = np.inf if i == 4 else qs[i]
            m = (score[cut:end] > lo) & (score[cut:end] <= hi)
            mv = mkt[cut:end][m]
            ms = mkt2[cut:end][m]
            mv = mv[np.isfinite(mv)]
            ms = ms[np.isfinite(ms)]
            if len(mv) < 200:
                continue
            flag = '  ← 最高档' if i == 4 else ('  ← 最低档' if i == 0 else '')
            say(f'{H:>6}{i + 1:>4}{len(mv):>9,}{mv.mean() * 100:>+9.3f}%'
                f'{np.median(mv) * 100:>+9.3f}%{(mv > 0).mean() * 100:>6.1f}%'
                f'{ms.mean() * 100:>+9.3f}%{flag}')
            rows.append(dict(H=H, tier=i + 1, n=len(mv), mean=mv.mean() * 100,
                             med=np.median(mv) * 100, win=(mv > 0).mean() * 100,
                             shuf=ms.mean() * 100))
        d = pd.DataFrame([r for r in rows if r['H'] == H])
        if len(d) >= 5:
            spread = d.iloc[-1]['mean'] - d.iloc[0]['mean']
            top = d.iloc[-1]['mean']
            say(f'        最高档−最低档 = {spread:+.3f}%   最高档单边 {top:+.3f}%   '
                f'扣maker(0.04%)后 {top - 0.04:+.3f}%   扣taker(0.20%)后 {top - 0.20:+.3f}%')
        say('')

    say('=' * 96)
    say('【分段稳定性】只取最高档（top 20%），各 H 下 4 段均值')
    say(f'  {"H":>6}' + ''.join(f'{"段" + str(i + 1):>12}' for i in range(NSEG)))
    for H in H_LIST:
        fwd = fwd_of(C, H)
        mkt = np.nanmean(fwd, axis=1)
        end = T - H
        seg_len = (end - cut) // NSEG
        vals = []
        for i in range(NSEG):
            lo = cut + i * seg_len
            hi = cut + (i + 1) * seg_len if i < NSEG - 1 else end
            ss = score[lo:hi]
            ff = mkt[lo:hi]
            q = np.nanquantile(ss, 0.8)
            m = ss >= q
            v = ff[m]
            v = v[np.isfinite(v)]
            vals.append(v.mean() * 100 if len(v) > 50 else np.nan)
        say(f'  {H:>6}' + ''.join(f'{x:>+11.3f}%' for x in vals))

    say('')
    say('【读法】')
    say('  · 若均值随 H 增长（60→1440）⇒ 信号在累积，成本占比下降，可交易性上升')
    say('  · 若均值不增长或衰减 ⇒ 这是一次性冲击，不是持续漂移，拉长持有只会摊薄收益')
    say('  · 分段同向段数为 4/4 才算稳')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_combo_h.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gambler_combo_h.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
