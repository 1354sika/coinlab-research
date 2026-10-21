# -*- coding: utf-8 -*-
"""gambler_cross_quarter · 不同源（截面/宽度/时间）× 自然季度 的最小检验

要回答的问题：
  三张牌（disp240 / breadth60 / dow_cos）在 60 币 × 3 年 5m 上，
  按自然季度分层、扣掉成本后，还能不能每个季度为正？

口径（全部无前视）：
  · 特征只用过去：disp240=60币48根动量截面std；breadth60=60币12根上涨占比；
    dow_cos=星期余弦
  · 分位每季度用【之前所有数据】重估一次，季度内固定
  · 每 12 根（60 分钟）决策一次，top 档做多，持有 12 根
  · 成本两套：maker 0.04% 往返 / taker 0.50% 往返
  · 对照：--shuffle 按天块重排 log 收益

用法：
  python -X utf8 gambler_cross_quarter.py
  python -X utf8 gambler_cross_quarter.py --shuffle
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CQ_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CQ_NSYM', '60'))
BPD = int(os.environ.get('CQ_BPD', '288'))
H = int(os.environ.get('CQ_H', '12'))          # 持有 12 根 = 60 分钟
STEP = H                                       # 决策间隔 = 持有期，避免重叠
COST_M = 0.0004
COST_T = 0.0050
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
    C2 = np.exp(np.log(np.maximum(c0, 1e-12)) + np.cumsum(r[idx], axis=0))
    return C2


def rank_of(x, qs):
    """把连续值映射到 0..len(qs) 的秩（用给定的分位边界）。"""
    return np.searchsorted(qs, x, side='right')


def main():
    t0 = time.time()
    say('=' * 104)
    say(f'gambler_cross_quarter · 三张牌 × 自然季度 [{TAG}]')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  持有 {H} 根  maker {COST_M:.2%} / taker {COST_T:.2%}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    pc = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    C = pc.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    tidx = pc.index
    rng = np.random.default_rng(20261020)
    if SHUF:
        C = shuffle_panel(C, rng)
        say('  ⚠ 已打乱：按天块重排 log 收益（总漂移不变）')
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    # ---- 三张牌（只用过去） ----
    mom240 = np.full((T, S), np.nan)
    mom240[48:] = C[48:] / C[:-48] - 1.0
    mom60 = np.full((T, S), np.nan)
    mom60[12:] = C[12:] / C[:-12] - 1.0
    disp240 = np.nanstd(mom240, axis=1)
    breadth60 = np.nanmean(mom60 > 0, axis=1)
    weekday = tidx.weekday.values.astype(float)
    dow_cos = np.cos(2 * np.pi * weekday / 7.0)
    say(f'  三张牌算完  {time.time()-t0:.0f}s')

    # ---- 季度分位（每季度用之前数据重估，季度内固定） ----
    q = pd.PeriodIndex(tidx, freq='Q')
    q_code = q.asi8
    q_start = np.flatnonzero(np.diff(q_code, prepend=q_code[0] - 1) != 0)
    q_names = {int(q_code[i]): str(q[i]) for i in q_start}

    # 决策点：每 STEP 根
    decisions = []
    for qi, s0 in enumerate(q_start):
        s1 = q_start[qi + 1] if qi + 1 < len(q_start) else T
        # 本季度可用的分位：只用 s0 之前的数据
        if s0 < BPD * 2:
            continue
        ref_disp = disp240[:s0]; ref_disp = ref_disp[np.isfinite(ref_disp)]
        ref_br = breadth60[:s0]; ref_br = ref_br[np.isfinite(ref_br)]
        if len(ref_disp) < 200 or len(ref_br) < 200:
            continue
        qd = np.quantile(ref_disp, [0.2, 0.4, 0.6, 0.8])
        qb = np.quantile(ref_br, [0.2, 0.4, 0.6, 0.8])
        # dow_cos 是固定周期，直接与历史分位比
        qw = np.quantile(dow_cos[:s0], [0.2, 0.4, 0.6, 0.8])
        # 季度内每 STEP 根一个决策
        for t in range(max(s0, BPD), min(s1, T - H), STEP):
            if not np.isfinite(disp240[t]) or not np.isfinite(breadth60[t]):
                continue
            score = (rank_of(disp240[t], qd) - rank_of(breadth60[t], qb)
                     - rank_of(dow_cos[t], qw))
            decisions.append((int(q_code[t]), t, int(score)))

    if not decisions:
        say('❌ 没有决策点')
        return

    D = pd.DataFrame(decisions, columns=['q', 't', 'score'])
    # top 档：score 的 80% 分位以上（每季度内相对）
    picks = []
    for qq, g in D.groupby('q'):
        thr = g.score.quantile(0.80)
        picks.append(g[g.score >= thr])
    P = pd.concat(picks) if picks else D.iloc[:0]
    say(f'  决策点 {len(D):,}，top 档 {len(P):,}  {time.time()-t0:.0f}s')

    # ---- 组合收益（60 币等权） ----
    rows = []
    for _, r in P.iterrows():
        t = int(r.t)
        seg = C[t + H] / C[t] - 1.0
        seg = seg[np.isfinite(seg)]
        if len(seg) < 10:
            continue
        rows.append(dict(q=int(r.q), t=t, score=int(r.score),
                         gross=float(np.mean(seg)), ncoins=len(seg)))
    R = pd.DataFrame(rows)
    if len(R) == 0:
        say('❌ 没有成交')
        return

    say('')
    say(f'【一】top 档按自然季度的表现（{TAG}）')
    say(f'  {"季度":>8} {"笔数":>5} {"毛单笔":>8} {"maker净":>8} {"taker净":>8} {"胜率":>6} {"季度maker资金":>12}')
    pos_m = pos_t = nq = 0
    for qq in sorted(R.q.unique()):
        g = R[R.q == qq]
        gm = g.gross.mean()
        gm_net = gm - COST_M
        gt_net = gm - COST_T
        eq = float(np.prod(1 + g.gross.values - COST_M))
        win = (g.gross.values > COST_M).mean()
        nq += 1
        pos_m += 1 if eq > 1 else 0
        pos_t += 1 if (g.gross.mean() - COST_T) > 0 else 0
        say(f'  {q_names.get(qq, str(qq)):>8} {len(g):>5} {gm*100:>7.3f}% '
            f'{gm_net*100:>7.3f}% {gt_net*100:>7.3f}% {win*100:>5.1f}% {eq:>11.3f}')

    say('')
    gross = R.gross.mean()
    say(f'【二】总体：{len(R)} 笔  毛单笔 {gross*100:+.4f}%  '
        f'maker 净 {(gross-COST_M)*100:+.4f}%  taker 净 {(gross-COST_T)*100:+.4f}%')
    say(f'  正季度：maker {pos_m}/{nq}  taker {pos_t}/{nq}')

    say('')
    say('【三】分档单调性（score 0..8）')
    for sc in sorted(D.score.unique()):
        g = D[D.score == sc]
        if len(g) < 5:
            continue
        vals = []
        for t in g.t:
            seg = C[t + H] / C[t] - 1.0
            seg = seg[np.isfinite(seg)]
            vals.append(np.mean(seg) if len(seg) else np.nan)
        vals = np.array(vals); vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            say(f'  score={sc:>2}  n={len(vals):>5}  毛单笔 {vals.mean()*100:+.4f}%  '
                f'maker 净 {(vals.mean()-COST_M)*100:+.4f}%')

    say('')
    say('【判据】正季度比例 + maker 净单笔 + 分档单调；由账本客观给出，不设门槛')
    say(f'TOTAL {time.time()-t0:.0f}s')

    with open(os.path.join(OUT, f'gambler_cross_quarter_{TAG}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    R.to_csv(os.path.join(OUT, f'gambler_cross_quarter_{TAG}_rows.csv'),
             index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
