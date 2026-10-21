# -*- coding: utf-8 -*-
"""
gambler_combo.py · 把过闸的牌合成一手（叠加检验）

gambler_gate_cross.py（修正版）里过闸 3 张：
  disp240   真 +0.071%  打乱 +0.007%  t(差) +7.83   截面分化大 → 后续涨
  breadth60 真 -0.073%  打乱 -0.008%  t(差) -7.67   宽度高（人群一致涨）→ 后续跌
  dow_cos   真 -0.073%  打乱 -0.023%  t(差) -5.65   星期效应

单张只有 0.07%/笔，扣 maker 0.04% 后剩 0.03%，不够。
这个脚本问：三张叠在一起，能不能到 0.15~0.20%（覆盖 taker）？

做法：
  ① 每个特征按训练段分位做秩归一（-1..1），按过闸方向对齐成同一符号
  ② 合成分数 score = 三个对齐后秩的等权和
  ③ 按 score 分 5 档，看每档未来收益；再看最高档 − 最低档
  ④ 分段稳定性 + 打乱对照
  ⑤ 提醒：这三个特征都是"全市场统一"的（对每个币取值相同），
    所以真做起来是**大盘择时**（同时开 60 币），不是选币。

用法：python -X utf8 gambler_combo.py
输出：evidence/gambler_combo.txt / .csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate import tstat
from gambler_gate_cross import mom, shuffle_panel

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CROSS_N', '60'))
H = 60
TRAIN = 0.70
NSEG = 4
BLOCK = 1440
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def rank01(v):
    """把向量映射到 0..1 的经验累积分布（nan 保持 nan）"""
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.sum() < 100:
        return out
    r = np.argsort(np.argsort(v[ok]))
    out[ok] = r / (ok.sum() - 1)
    return out


def build_score(C, dow):
    """返回 (T,) 合成分数：disp240 正、breadth60 负、dow_cos 负"""
    M240 = mom(C, 240)
    valid = np.isfinite(M240)
    nv = valid.sum(axis=1)
    enough = nv >= 30
    disp = np.where(enough, np.nanstd(M240, axis=1), np.nan)
    M60 = mom(C, 60)
    valid6 = np.isfinite(M60)
    nv6 = valid6.sum(axis=1)
    br = np.where(nv6 >= 30, ((M60 > 0) & valid6).sum(axis=1) / np.maximum(nv6, 1), np.nan)
    dc = np.cos(2 * np.pi * dow / 7)
    a = rank01(disp)
    b = rank01(br)
    c = rank01(dc)
    return (a - b - c) / 3.0


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_combo · 三张过闸牌合成一手')
    say(f'  disp240(+) + breadth60(−) + dow_cos(−) | H={H} 分钟 | 币 {NSYM}')
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
    say(f'  矩阵 {T:,} × {S} 币  缺失 {np.isnan(C).mean():.2%}  {time.time() - t0:.0f}s')

    fwd = np.full((T, S), np.nan)
    a = C[H:]
    b = C[:-H]
    ok = np.isfinite(a) & np.isfinite(b)
    v = np.full_like(a, np.nan)
    v[ok] = a[ok] / b[ok] - 1
    fwd[:-H] = v

    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    fwd2 = np.full((T, S), np.nan)
    a2 = C2[H:]
    b2 = C2[:-H]
    ok2 = np.isfinite(a2) & np.isfinite(b2)
    v2 = np.full_like(a2, np.nan)
    v2[ok2] = a2[ok2] / b2[ok2] - 1
    fwd2[:-H] = v2
    say(f'  打乱面板完成  {time.time() - t0:.0f}s')

    score = build_score(C, dow)
    score2 = build_score(C2, dow)
    cut = int(T * TRAIN)

    say('')
    say(f'【一】按合成分数分 5 档（训练段定分位，后 {1 - TRAIN:.0%} 评估）')
    say(f'  {"档位":<10}{"样本":>10}{"均值":>10}{"中位":>10}{"胜率":>8}{"打乱均值":>11}')
    qs = np.nanquantile(score[:cut], [0.2, 0.4, 0.6, 0.8])
    mkt_fwd = np.nanmean(fwd, axis=1)          # 全市场平均收益（大盘代理）
    mkt_fwd2 = np.nanmean(fwd2, axis=1)
    tiers = []
    for i in range(5):
        lo = -np.inf if i == 0 else qs[i - 1]
        hi = np.inf if i == 4 else qs[i]
        m = (score[cut:] > lo) & (score[cut:] <= hi)
        mv = mkt_fwd[cut:][m]
        ms = mkt_fwd2[cut:][m]
        mv = mv[np.isfinite(mv)]
        ms = ms[np.isfinite(ms)]
        if len(mv) < 100:
            continue
        tiers.append((i + 1, len(mv), mv.mean() * 100, np.median(mv) * 100,
                      (mv > 0).mean() * 100, ms.mean() * 100))
        say(f'  {(i + 1):<10}{len(mv):>10,}{mv.mean() * 100:>+9.3f}%'
            f'{np.median(mv) * 100:>+9.3f}%{(mv > 0).mean() * 100:>7.1f}%{ms.mean() * 100:>+10.3f}%')

    if len(tiers) == 5:
        top = tiers[4][2]
        bot = tiers[0][2]
        say(f'  最高档 − 最低档 = {top - bot:+.3f}%   （单张牌约 0.07%）')
        say(f'  最高档单边 = {top:+.3f}%（打乱 {tiers[4][5]:+.3f}%）')

    say('')
    say('【二】只做最高档（top 20%）vs 最低档（bottom 20%），分段稳定性')
    say(f'  {"段":>3}{"top均值":>10}{"top样本":>10}{"bottom均值":>12}{"bottom样本":>11}')
    seg_len = T // NSEG
    tops, bots = [], []
    for i in range(NSEG):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < NSEG - 1 else T
        ss = score[lo:hi]
        ff = mkt_fwd[lo:hi]
        q = np.nanquantile(ss[:max(1000, (hi - lo) // 2)], [0.2, 0.8])
        mt = ss >= q[1]
        mb = ss <= q[0]
        vt = ff[mt]
        vb = ff[mb]
        vt = vt[np.isfinite(vt)]
        vb = vb[np.isfinite(vb)]
        if len(vt) < 100:
            continue
        tops.append(vt.mean() * 100)
        bots.append(vb.mean() * 100)
        say(f'  {i + 1:>3}{vt.mean() * 100:>+9.3f}%{len(vt):>10,}'
            f'{vb.mean() * 100:>+11.3f}%{len(vb):>11,}')

    say('')
    say('=' * 100)
    say('【三】判据')
    if tops:
        say(f'  · top 档在 {len(tops)} 段里的均值：' + '  '.join(f'{x:+.3f}%' for x in tops))
        say(f'  · 同向段数 = {sum(1 for x in tops if x > 0)}/{len(tops)}（做多为正）')
        best = max(abs(x) for x in tops)
        say(f'  · 单段最好 |均值| = {best:.3f}%')
        if best >= 0.20:
            say('  ✅ 至少在某段能覆盖 taker 来回（0.20%）')
        elif best >= 0.10:
            say('  ⚠ 只够覆盖 maker 来回（0.04%）后剩一点')
        else:
            say('  ❌ 扣费后基本归零')
    say('')
    say('  提醒：这三张牌都是全市场统一取值 ⇒ 实际是"大盘择时"（同时开 60 币），')
    say('  成本要按 60 笔算，且各币高度相关（等于押一个方向），仓位必须按整体风险算。')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_combo.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))


if __name__ == '__main__':
    main()
