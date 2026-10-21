# -*- coding: utf-8 -*-
"""gambler_exit · 退出决策独立学习（反事实账本）

用户 2026-10-19 第 3 点：
  "关键是做多之后，赌徒模型能不能够自己选择在合适的地方果断退场"

当前 v3 的问题：76% 的平仓是 time（持满 240 根到期），tp/sl 是死阈值。
本脚本把"退出"拆成独立的学习问题：

  第一遍（收集）：模拟所有入场，记录每笔之后每一根的收益路径
  反事实：对每个决策点 k，比较"现在退"vs"继续拿到 H"的收益差
  学习：按 (已持有比例, 当前浮盈档) 分桶，统计"继续更好"的比例
  第二遍（应用）：持仓中每根查表，若该桶"继续更好"的比例 < 阈值 ⇒ 立刻退

关键实验：入场用 SHUF（一副没信息的假牌）——如果退出规则能在假牌上赚钱，
就说明**退出本身是独立的 alpha 来源**，不依赖入场牌。

用法：EX_NSYM=6 python -X utf8 gambler_exit.py
输出：evidence/gambler_exit.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

from gambler_adaptive_v3 import (DATA, SCALES, BLOCK, FEE, TRAIN, OUT,
                                 scale_scores, shuffle_scores)

NSYM = int(os.environ.get('EX_NSYM', '6'))
THR = 0.60
H = 240
SCALE = 240          # 用哪一张牌做入场（240 分）
COST = FEE
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def collect(C, O, SC, cut, T, S):
    """收集每笔入场 + 之后 H 根的收益路径（单币单持仓）"""
    pending = {}
    out = []
    for t in range(cut, T - 2):
        for s_ in list(pending.keys()):
            ps = pending[s_]
            px = C[t, s_]
            if not np.isfinite(px):
                continue
            ps['path'].append(px / ps['entry_p'] - 1.0)
            if len(ps['path']) >= H:
                out.append(dict(sym=s_, t0=ps['entry_t'], path=np.array(ps['path'])))
                del pending[s_]
        row = SC[t]
        for s_ in range(S):
            if s_ in pending or not np.isfinite(row[s_]) or row[s_] < THR:
                continue
            px = O[t + 1, s_]
            if not np.isfinite(px):
                continue
            pending[s_] = dict(entry_t=t + 1, entry_p=px, path=[])
    return out


def learn_exit(trades, ks=(30, 60, 120, 180)):
    """反事实：在持有 k 根时，继续拿到 H 是否更好

    返回 table[(k, hold_bin, pnl_bin)] = (n, mean_delta, win_rate)
      hold_bin: 已持有比例档 0..4
      pnl_bin : 当前浮盈档 0..4（相对 −2%..+2% 分位）
    """
    tab = {}
    for tr in trades:
        p = tr['path']
        for k in ks:
            if k >= len(p):
                continue
            early = p[k - 1]
            late = p[-1]
            hb = min(4, int(k / H * 5))
            pb = int(np.clip((early + 0.02) / 0.04, 0, 0.999) * 5)
            key = (k, hb, pb)
            d = late - early
            a = tab.setdefault(key, [])
            a.append(d)
    out = {}
    for key, v in tab.items():
        a = np.array(v)
        out[key] = (len(a), float(a.mean()), float((a > 0).mean()))
    return out


def apply_exit(C, O, SC, cut, T, S, tab, min_n=50, p_thr=0.52):
    """第二遍：每根查表，若"继续更好"的概率 < p_thr ⇒ 立刻退"""
    pending = {}
    trades = []
    n_check = n_lookup = 0
    for t in range(cut, T - 2):
        for s_ in list(pending.keys()):
            ps = pending[s_]
            px = C[t, s_]
            if not np.isfinite(px):
                continue
            ret = px / ps['entry_p'] - 1.0
            k = t - ps['entry_t'] + 1
            if k >= H:
                trades.append(dict(t=t, ret=ret - COST, why='time', held=k))
                del pending[s_]
                continue
            pb = int(np.clip((ret + 0.02) / 0.04, 0, 0.999) * 5)
            n_check += 1
            # 找最近的 k 桶（注意：表里的 hold_bin 是用 kk 算的，不是用当前 k）
            best, best_kk = None, -1
            for kk in (30, 60, 120, 180):
                if kk > k:
                    continue
                hb_k = min(4, int(kk / H * 5))
                if (kk, hb_k, pb) in tab and kk > best_kk:
                    best, best_kk = tab[(kk, hb_k, pb)], kk
            if best is not None:
                n_lookup += 1
            if best and best[0] >= min_n and best[2] < p_thr:
                trades.append(dict(t=t, ret=ret - COST, why='exit', held=k))
                del pending[s_]
        row = SC[t]
        for s_ in range(S):
            if s_ in pending or not np.isfinite(row[s_]) or row[s_] < THR:
                continue
            px = O[t + 1, s_]
            if not np.isfinite(px):
                continue
            pending[s_] = dict(entry_t=t + 1, entry_p=px)
    n_exit = sum(1 for x in trades if x['why'] == 'exit')
    say(f'    [诊断] 查表 {n_check} 次，有桶 {n_lookup} 次，触发早退 {n_exit} 笔')
    return pd.DataFrame(trades)


def stat(tr, name):
    if len(tr) < 20:
        say(f'  {name:<16} 样本不足 n={len(tr)}')
        return
    a = tr.ret.values
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    say(f'  {name:<16} n={len(a):>6}  均值={a.mean()*100:+.4f}%  中位={np.median(a)*100:+.4f}%  '
        f'胜率={(a>0).mean()*100:.1f}%  t={t:+.2f}')


def main():
    t0 = time.time()
    say('=' * 96)
    say('gambler_exit · 退出决策独立学习（反事实账本）')
    say(f'  入场牌 {SCALE} 分 >= {THR} | H={H} | 币 {NSYM} | 费 {COST:.2%}')
    say('=' * 96)

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
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    SR = scale_scores(C, dow)
    rng = np.random.default_rng(20261020)
    SS = shuffle_scores(SR, rng)
    cut = int(T * TRAIN)

    say('')
    say('【一】收集反事实账本')
    trR = collect(C, O, SR[SCALE], cut, T, S)
    trS = collect(C, O, SS[SCALE], cut, T, S)
    say(f'  真牌 {len(trR)} 笔 / 乱牌 {len(trS)} 笔  {time.time()-t0:.0f}s')

    say('')
    say('【二】学到什么（继续拿到 H 是否更好；>0.5 表示该继续）')
    tabR = learn_exit(trR)
    tabS = learn_exit(trS)
    for name, tab in (('真牌', tabR), ('乱牌', tabS)):
        keys = sorted(tab.keys())
        if not keys:
            say(f'  {name} 账本为空')
            continue
        ps = np.array([tab[k][2] for k in keys if tab[k][0] >= 50])
        if len(ps) == 0:
            say(f'  {name} 可用桶 0 个')
            continue
        say(f'  {name} 可用桶 {len(ps)} 个  继续更好比例：中位 {np.median(ps):.3f}  '
            f'最小 {ps.min():.3f}  最大 {ps.max():.3f}  '
            f'<0.5 的桶 {int((ps<0.5).sum())} 个')
        say('    最该早退的 5 个桶（继续更好比例最低）：')
        worst = sorted([k for k in keys if tab[k][0] >= 50], key=lambda k: tab[k][2])[:5]
        for k in worst:
            n, md, wr = tab[k]
            say(f'      k={k[0]:>3} 持有档{k[1]} 浮盈档{k[2]}  n={n:>4}  继续更好={wr:.3f}  平均差={md*100:+.4f}%')

    say('')
    say('【三】应用退出规则（每根查表，概率 < 0.52 就退）')
    # 用训练段学、样本外应用：把 tab 学在 cut 之前
    cut2 = int(T * 0.55)
    trR_tr = collect(C, O, SR[SCALE], cut, cut2, S)
    trS_tr = collect(C, O, SS[SCALE], cut, cut2, S)
    tabR2 = learn_exit(trR_tr)
    tabS2 = learn_exit(trS_tr)

    say('  —— 真牌入场 ——')
    a1 = apply_exit(C, O, SR[SCALE], cut2, T, S, tabR2)
    stat(a1, '固定H(基线)')     # 同数据、同入场，仅退出规则不同，需对照
    b1 = collect(C, O, SR[SCALE], cut2, T, S)
    base1 = pd.DataFrame([dict(ret=p[-1] - COST) for p in [x['path'] for x in b1]])
    stat(base1, '固定H 到 240')
    stat(a1, '学会的退出')

    say('  —— 乱牌入场（检验退出是否独立 alpha）——')
    a2 = apply_exit(C, O, SS[SCALE], cut2, T, S, tabS2)
    b2 = collect(C, O, SS[SCALE], cut2, T, S)
    base2 = pd.DataFrame([dict(ret=p[-1] - COST) for p in [x['path'] for x in b2]])
    stat(base2, '固定H 到 240')
    stat(a2, '学会的退出')

    say('')
    say('【判据】')
    say('  · 若"学会的退出"在乱牌上也 > 固定H，说明退出本身有价值（独立 alpha）')
    say('  · 若只在真牌上好，说明退出依赖入场信息')
    say('  · 两边都没改善 ⇒ 这个状态表学不到东西，需换状态定义')
    say('')
    say(f'TOTAL {time.time()-t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_exit.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))


if __name__ == '__main__':
    main()
