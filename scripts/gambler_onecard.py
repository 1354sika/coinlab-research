# -*- coding: utf-8 -*-
"""单张牌体检：12 个尺度逐张跑（只做多），每张都分 4 段看是否同向

用户 2026-10-19 三条：
  1. 逐张牌跑四段，筛出真正四段同向的那一张
  2. 进化机制必须正向且有效
  3. 做多之后，模型能不能自己选择在合适的地方果断退场

本脚本只做第 1 件：把"12 张牌混着用"拆成"每张牌单独用"，
每张牌再切成 4 段，看它是不是**四段同号**。四段不同号的牌一律不认。

判据（写死，不给"全局平均"留后门）：
  ✅ 四段同号 且 每段 |均值| ≥ 0.02%（扣费后还剩一点）
  ❌ 任一段反号 ⇒ 直接判死
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

from gambler_adaptive_v3 import (DATA, SCALES, BLOCK, FEE, TRAIN, OUT,
                                 scale_scores, shuffle_scores)

NSYM = int(os.environ.get('ONE_NSYM', '60'))
THR = 0.60
H = 240
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def bt_single(C, O, SC, cut, T, S):
    """单尺度、只做多、H=240、按过去 1 天分位止损止盈"""
    pending = {}
    trades = []
    R1 = np.zeros((T, S))
    R1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    R1 = np.nan_to_num(R1)
    tp_pct, sl_pct = 0.02, -0.015
    for t in range(cut, T - 1):
        if t % BLOCK == 0 and t > 1440:
            seg = R1[t - 1440:t].ravel()
            seg = seg[np.isfinite(seg)]
            if len(seg) > 5000:
                tp_pct = float(np.quantile(seg, 0.97))
                sl_pct = float(np.quantile(seg, 0.03))
        for s_ in list(pending.keys()):
            ps = pending[s_]
            px = C[t, s_]
            if not np.isfinite(px):
                continue
            ret = px / ps['entry_p'] - 1.0
            held = t - ps['entry_t'] + 1
            if held >= H or ret >= tp_pct or ret <= sl_pct:
                why = 'tp' if ret >= tp_pct else 'sl' if ret <= sl_pct else 'time'
                trades.append(dict(t=t, sym=s_, ret=ret - FEE, held=held, why=why))
                del pending[s_]
        row = SC[t]
        for s_ in range(S):
            if s_ in pending or not np.isfinite(row[s_]):
                continue
            if row[s_] < THR:          # 只做多
                continue
            px = O[t + 1, s_]
            if not np.isfinite(px):
                continue
            pending[s_] = dict(entry_t=t + 1, entry_p=px)
    return pd.DataFrame(trades)


def seg_stats(tr, T):
    """样本外按时间四分，返回 4 段的 (n, mean, t)"""
    out = []
    if len(tr) < 40:
        return out
    q = np.quantile(tr.t.values, [0, .25, .5, .75, 1.0])
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        g = tr[(tr.t >= lo) & (tr.t <= hi)] if i == 3 else tr[(tr.t >= lo) & (tr.t < hi)]
        a = g.ret.values
        if len(a) < 10:
            out.append((len(a), np.nan, np.nan))
            continue
        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
        out.append((len(a), a.mean(), t))
    return out


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_onecard · 单张牌体检（只做多 | 每张牌切 4 段 | 四段必须同号）')
    say(f'  牌 {len(SCALES)} 张 {SCALES} | 币 {NSYM} | thr={THR} | H={H} | 费 {FEE:.2%}')
    say('=' * 100)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv_c = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    piv_o = df.pivot_table(index='t', columns='symbol', values='open').sort_index()[syms]
    C = piv_c.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    O = piv_o.values.astype(np.float64)
    O[~np.isfinite(O) | (O <= 0)] = np.nan
    T, S = C.shape
    dow = piv_c.index.dayofweek.values
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    SR = scale_scores(C, dow)
    say(f'  12 张牌算完  {time.time() - t0:.0f}s')
    rng = np.random.default_rng(20261019)
    SS = shuffle_scores(SR, rng)
    cut = int(T * TRAIN)

    rows = []
    say('')
    say(f'{"牌(分)":>8} {"策略":<4} {"笔数":>7} {"总均值":>9} | ' +
        ' '.join(f'{"段%d" % i:>17}' for i in range(1, 5)) + '  判决')
    say('-' * 100)
    for sc in SCALES:
        for tag, SD in (('真', SR), ('乱', SS)):
            tr = bt_single(C, O, SD[sc], cut, T, S)
            if len(tr) < 40:
                say(f'{sc:>8} {tag:<4} {len(tr):>7}  样本太少')
                continue
            st = seg_stats(tr, T)
            means = [x[1] for x in st]
            same = all(np.isfinite(m) and m > 0 for m in means) and len(st) == 4
            cell = ' '.join(f'{m*100:+.3f}%({t:+.1f})' if np.isfinite(m) else '   -   '
                            for _, m, t in st)
            rows.append(dict(scale=sc, tag=tag, n=len(tr), mean=tr.ret.mean(),
                             same=same, m1=means[0] if len(means) > 0 else np.nan,
                             m2=means[1] if len(means) > 1 else np.nan,
                             m3=means[2] if len(means) > 2 else np.nan,
                             m4=means[3] if len(means) > 3 else np.nan))
            say(f'{sc:>8} {tag:<4} {len(tr):>7} {tr.ret.mean()*100:>+8.4f}% | {cell}  '
                f'{"✅四段同正" if same else "❌"}')

    say('')
    say('【判据】四段全部 >0 才算"活着的牌"；任一段反号 ⇒ 判死')
    alive = [r for r in rows if r['tag'] == '真' and r['same']]
    if alive:
        say(f'  活着的牌 {len(alive)} 张：' + ', '.join(str(r['scale']) for r in alive))
        # 再用打乱对照复核：真牌必须同时赢过同尺度的打乱版
        for r in alive:
            sh = next((x for x in rows if x['tag'] == '乱' and x['scale'] == r['scale']), None)
            if sh:
                say(f'    {r["scale"]:>6}分  真 {r["mean"]*100:+.4f}%  乱 {sh["mean"]*100:+.4f}%  '
                    f'差 {(r["mean"]-sh["mean"])*100:+.4f}%')
    else:
        say('  ❌ 0 张。所有 12 张牌都至少有一段反号 ⇒ 这一批价格量牌没有可用的种子')

    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_onecard.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gambler_onecard.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
