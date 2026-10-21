# -*- coding: utf-8 -*-
"""
gambler_combo_h_indep.py · 去重叠：每个持有期只取一个不重叠的样本点

上一轮 gambler_combo_h.py 的发现（1 年 1m 数据）：
  H=60 分：最高档 +0.079%/笔（扣 taker 后 -0.121%）
  H=1440 分：最高档 +1.103%/笔（扣 taker 后 +0.903%），4 段全正
  打乱对照在所有 H 上都接近 0（-0.01 ~ -0.07）

但有一个致命的技术问题：**样本重叠**。
用 1 分钟数据算 1 天持有期，相邻两分钟的两个"1 天收益"有 1439 分钟是同一段，
等于把同一个观察数了 1440 遍 —— t 值会被大幅虚高。

这个脚本做干净版本：
  每 H 分钟只取一个样本点（不重叠），重算：
    ① 各 H 下 top 20% / bottom 20% 的真实均值
    ② 独立样本数（1 年数据下 H=1440 只有约 365 个点）
    ③ 分段方向

判据：
  不重叠样本下仍显著、方向一致 ⇒ 这份信号值得上更长的历史数据去确认
  不重叠后消失 ⇒ 之前是重叠造成的假象

用法：python -X utf8 gambler_combo_h_indep.py
输出：evidence/gambler_combo_h_indep.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate_cross import shuffle_panel
from gambler_combo import build_score

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


def tstat(a):
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return np.nan
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def main():
    t0 = time.time()
    say('=' * 96)
    say('gambler_combo_h_indep · 去重叠检验（每 H 分钟取一个不重叠样本）')
    say(f'  H ∈ {H_LIST} | 币 {NSYM} | 训练定分位 {TRAIN:.0%}')
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
    say(f'  score 完成  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    mkt = np.nanmean(np.where(np.isfinite(C), C, np.nan), axis=1)  # 占位，稍后覆盖
    say('')
    say(f'{"H(分)":>6}{"独立样本":>9}{"top均值":>10}{"top t":>8}{"bot均值":>10}{"bot t":>8}{"差":>9}{"打乱top":>10}')
    rows = []
    for H in H_LIST:
        fwd = np.full((T, S), np.nan)
        ok = np.isfinite(C[H:]) & np.isfinite(C[:-H])
        v = np.full_like(C[H:], np.nan)
        v[ok] = C[H:][ok] / C[:-H][ok] - 1
        fwd[:-H] = v
        fwd2 = np.full((T, S), np.nan)
        ok2 = np.isfinite(C2[H:]) & np.isfinite(C2[:-H])
        v2 = np.full_like(C2[H:], np.nan)
        v2[ok2] = C2[H:][ok2] / C2[:-H][ok2] - 1
        fwd2[:-H] = v2

        mkt = np.nanmean(fwd, axis=1)
        mkt2 = np.nanmean(fwd2, axis=1)

        # 不重叠采样：起点 cut，步长 H
        idx = np.arange(cut, T - H, H)
        sc = score[idx]
        ff = mkt[idx]
        ff2 = mkt2[idx]
        okm = np.isfinite(sc) & np.isfinite(ff)
        sc, ff, ff2 = sc[okm], ff[okm], ff2[okm]
        if len(sc) < 50:
            say(f'{H:>6}  样本不足（{len(sc)}）')
            continue
        q = np.nanquantile(score[:cut], [0.2, 0.8])
        mt = sc >= q[1]
        mb = sc <= q[0]
        vt, vb = ff[mt], ff[mb]
        vt2 = ff2[mt]
        say(f'{H:>6}{len(sc):>9,}{vt.mean() * 100:>+9.3f}%{tstat(vt):>+8.2f}'
            f'{vb.mean() * 100:>+9.3f}%{tstat(vb):>+8.2f}'
            f'{(vt.mean() - vb.mean()) * 100:>+8.3f}%{vt2.mean() * 100:>+9.3f}%')
        rows.append(dict(H=H, n=len(sc), top=vt.mean() * 100, t_top=tstat(vt),
                         bot=vb.mean() * 100, t_bot=tstat(vb),
                         spread=(vt.mean() - vb.mean()) * 100, shuf_top=vt2.mean() * 100,
                         n_top=int(mt.sum()), n_bot=int(mb.sum())))

    say('')
    say('=' * 96)
    say('【分段】不重叠样本下，top 20% 各段均值')
    say(f'  {"H":>6}' + ''.join(f'{"段" + str(i + 1):>11}' for i in range(NSEG)) + f'{"样本/段":>10}')
    for H in H_LIST:
        fwd = np.full((T, S), np.nan)
        ok = np.isfinite(C[H:]) & np.isfinite(C[:-H])
        v = np.full_like(C[H:], np.nan)
        v[ok] = C[H:][ok] / C[:-H][ok] - 1
        fwd[:-H] = v
        mkt = np.nanmean(fwd, axis=1)
        idx = np.arange(cut, T - H, H)
        sc = score[idx]
        ff = mkt[idx]
        okm = np.isfinite(sc) & np.isfinite(ff)
        sc, ff = sc[okm], ff[okm]
        if len(sc) < 50:
            continue
        seg = len(sc) // NSEG
        vals, ns = [], []
        for i in range(NSEG):
            lo = i * seg
            hi = (i + 1) * seg if i < NSEG - 1 else len(sc)
            s_i, f_i = sc[lo:hi], ff[lo:hi]
            qq = np.nanquantile(s_i, 0.8)
            m = s_i >= qq
            vals.append(f_i[m].mean() * 100 if m.sum() > 10 else np.nan)
            ns.append(int(m.sum()))
        say(f'  {H:>6}' + ''.join(f'{x:>+10.3f}%' for x in vals) + f'{ns[0]:>10}')

    say('')
    say('=' * 96)
    say('【判据】')
    if rows:
        for r in rows:
            tag = '✅ 独立样本下仍显著' if (r['t_top'] >= 2 and r['top'] > 0.20) else \
                  ('⚠ 方向对但不显著/不够厚' if r['top'] > 0 else '❌ 不成立')
            say(f'  H={int(r["H"]):>4} 独立样本 {int(r["n"]):>4}  top {r["top"]:+.3f}%  '
                f't={r["t_top"]:+.2f}  打乱 top {r["shuf_top"]:+.3f}%   {tag}')
    say('')
    say('  注意：1 年数据下 H=1440 只有约 365 个独立样本，统计功效很低；')
    say('  要确认这个信号，需要多年历史（E 盘正在补的 502 币 1m 数据）。')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_combo_h_indep.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gambler_combo_h_indep.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
