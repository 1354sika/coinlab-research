# -*- coding: utf-8 -*-
"""gambler_synthetic · 仪器校准：在已知答案的合成市场上验证整条机制

用户 2026-10-19 的方法质疑：
  "为什么不能机制全都写好，再去找牌？"
  —— 对。在真数据上调机制，永远分不清"机制坏"还是"牌烂"。

本脚本造一个**埋了确定规律**的合成市场：
  规律：过去 20 根涨幅 > +1% ⇒ 之后 60 根平均额外涨 +0.4%
  （这个规律是写进去的，必然存在；机制若抓不到，就是机制有 bug）

要验证的四件事：
  ① 入场牌能不能把这段规律识别成"高分"（score 分层单调）
  ② 回测能不能赚到（单笔均值应显著 > 0）
  ③ judge（四段同号 + 赢 SHUF）能不能判过
  ④ 退出学习（学会的退出 vs 持满 H）能不能改善

任何一项不过 ⇒ 机制不合格，先修机制，不许拿真数据结论。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

from gambler_judge import judge, fmt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evidence')
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def make_market(T=150000, S=6, seed=42, mu=0.004, lag=60, look=20, thr=0.01, pulse=False):
    """合成市场：过去 look 根涨幅 > thr ⇒ 之后 lag 根累计额外 +mu

    pulse=True 时改成"前一半涨、后一半跌回"（净 0）——用来验证退出学习：
    这种市场的最优退出就是恰好在中点，模型必须能学到"该早退"。
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.002, (T, S))            # 基础波动 0.2%/根
    P = np.exp(np.cumsum(r, axis=0))
    mom = np.zeros((T, S))
    mom[look:] = P[look:] / P[:-look] - 1.0
    sig = (mom > thr) & np.isfinite(mom)         # 触发条件（只用过去，无前视）
    # 把信号平移到"触发之后 lag 根"，均匀摊到每根
    add = np.zeros((T, S))
    for k in range(1, lag + 1):
        sh = np.zeros_like(sig, dtype=float)
        sh[k:] = sig[:-k]
        if pulse and k > lag // 2:
            add -= sh                      # 后半段回落 ⇒ 最优退出在中点
        else:
            add += sh
    r2 = r + add * (mu / lag)
    P2 = np.exp(np.cumsum(r2, axis=0)) * 100.0
    O2 = np.vstack([P2[0:1], P2[:-1]])           # 开盘=上一根收盘（简化）
    return P2, O2, mom, sig


def score_from_mom(mom, cut, T, S, look=20, win=1440):
    """把 mom(look) 变成 [0,1] 分数：用**过去 win 根**的滚动秩（无前视）"""
    out = np.full((T, S), np.nan)
    step = 1440
    for i in range(look, T, step):
        lo = max(look, i - win)
        ref = mom[lo:i].ravel()
        ref = ref[np.isfinite(ref)]
        if len(ref) < 100:
            continue
        order = np.sort(ref)
        hi = min(T, i + step)
        seg = mom[i:hi]
        ok = np.isfinite(seg)
        pos = np.full(seg.shape, np.nan)
        if ok.any():
            pos[ok] = np.searchsorted(order, seg[ok], side='right') / len(order)
        out[i:hi] = pos
    return out


def bt(C, O, SC, cut, T, S, thr=0.6, H=60):
    pending, trades = {}, []
    for t in range(cut, T - 2):
        for s in list(pending):
            ps = pending[s]
            px = C[t, s]
            if not np.isfinite(px):
                continue
            ret = px / ps['p'] - 1.0
            k = t - ps['t'] + 1
            if k >= H:
                trades.append(dict(t=t, ret=ret - 0.0004, held=k, why='time'))
                del pending[s]
        row = SC[t]
        for s in range(S):
            if s in pending or not np.isfinite(row[s]) or row[s] < thr:
                continue
            px = O[t + 1, s]
            if not np.isfinite(px):
                continue
            pending[s] = dict(t=t + 1, p=px)
    return pd.DataFrame(trades)


def main():
    t0 = time.time()
    T, S = 150000, 6
    say('=' * 96)
    say('gambler_synthetic · 仪器校准（已知答案：过去20根>+1% ⇒ 之后60根+0.4%）')
    say(f'  合成市场 {T:,} × {S} 币 | 费 0.04% | H=60')
    say('=' * 96)

    P, O, mom, sig = make_market(T=T, S=S)
    say(f'  市场造完  信号命中率 {sig.mean()*100:.2f}%  {time.time()-t0:.1f}s')

    # ---- ① 分数分层：合成规律必须能被 score 分层抓到 ----
    cut = int(T * 0.7)
    SC = score_from_mom(mom, cut, T, S)
    say('')
    say('【① 分数分层检验】按 score 分 10 档，看未来 60 根收益是否单调')
    fwd = np.full((T, S), np.nan)
    fwd[:-60] = P[60:] / P[:-60] - 1.0
    xs, ys = [], []
    for i in range(10):
        m = (SC >= i / 10) & (SC < (i + 1) / 10) & np.isfinite(fwd) & (np.arange(T)[:, None] >= cut)
        if m.sum() < 100:
            continue
        xs.append(f'{i/10:.1f}-{(i+1)/10:.1f}')
        ys.append(fwd[m].mean() * 100)
    say('  档位  ' + '  '.join(f'{x:>9}' for x in xs))
    say('  均值% ' + '  '.join(f'{y:>+9.3f}' for y in ys))
    mono = all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))
    say(f'  → {"✅ 单调递增（牌抓到了规律）" if mono else "❌ 不单调（牌抓不到规律 = 机制有 bug）"}')

    # ---- ② 回测能不能赚 ----
    say('')
    say('【② 回测】只做多，thr=0.6，H=60')
    tr = bt(P, O, SC, cut, T, S)
    if len(tr) < 20:
        say(f'  ❌ 交易太少 n={len(tr)}（机制有 bug）')
        return
    a = tr.ret.values
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    say(f'  n={len(a)}  均值={a.mean()*100:+.4f}%  中位={np.median(a)*100:+.4f}%  '
        f'胜率={(a>0).mean()*100:.1f}%  t={t:+.2f}')
    say(f'  → {"✅ 赚到了（机制能兑现规律）" if a.mean() > 0 and t > 2 else "❌ 没赚到（机制有 bug）"}')

    # ---- ③ judge ----
    say('')
    say('【③ judge 判据】四段同号 + 每段赢 SHUF')
    rng = np.random.default_rng(7)
    SCs = SC.copy()
    nb = T // 1440
    perm = rng.permutation(nb)
    idx = np.concatenate([np.arange(p * 1440, (p + 1) * 1440) for p in perm])
    SCs[:nb * 1440] = SC[idx]
    SCs[nb * 1440:] = SC[nb * 1440:]
    trs = bt(P, O, SCs, cut, T, S)
    ok, why, det = judge(tr.t.values, tr.ret.values, trs.t.values, trs.ret.values)
    say(fmt(det))
    say(f'  真牌 n={len(tr)} 均值={tr.ret.mean()*100:+.4f}% | '
        f'打乱 n={len(trs)} 均值={(trs.ret.mean()*100 if len(trs) else float("nan")):+.4f}%')
    say(f'  → {"✅ 判过（机制合格）" if ok else "❌ 判死：" + why + "（机制有 bug）"}')

    # ---- ④ 退出学习（必须用"会回落"的市场才能验证）----
    say('')
    say('【④ 退出学习】pulse 市场：信号后先涨 30 根再跌回 30 根 ⇒ 最优是持 30 根就退')
    P4, O4, _, sig4 = make_market(T=T, S=S, pulse=True)
    SC4 = sig4.astype(np.float64)
    pending, paths = {}, []
    H = 60
    for tt in range(cut, T - 2):
        for s in list(pending):
            ps = pending[s]
            px = P4[tt, s]
            if not np.isfinite(px):
                continue
            ps['path'].append(px / ps['p'] - 1.0)
            if len(ps['path']) >= H:
                paths.append(np.array(ps['path']))
                del pending[s]
        row = SC4[tt]
        for s in range(S):
            if s in pending or not np.isfinite(row[s]) or row[s] < 0.6:
                continue
            px = O4[tt + 1, s]
            if not np.isfinite(px):
                continue
            pending[s] = dict(t=tt + 1, p=px, path=[])
    if len(paths) < 100:
        say(f'  ❌ 路径太少 n={len(paths)}')
        return
    ks = (10, 20, 30, 40)
    tab = {}
    for p in paths:
        for k in ks:
            if k >= len(p):
                continue
            early, late = p[k - 1], p[-1]
            key = (k, int(np.clip((early + .02) / .04, 0, .999) * 5))
            tab.setdefault(key, []).append(late - early)
    stats = {k: (len(v), float(np.mean(v)), float(np.mean(np.array(v) > 0))) for k, v in tab.items()}
    good = [k for k, v in stats.items() if v[0] >= 50 and v[2] < 0.45]
    say(f'  学到 {len(stats)} 个桶，其中"早退更好"并且样本足够的桶 {len(good)} 个')
    base = np.array([p[-1] for p in paths])
    say(f'  持满 H={H}: 均值={base.mean()*100:+.4f}%  （pulse 市场应≈0，因为涨跌相抵）')
    for k in good[:5]:
        n, md, wr = stats[k]
        say(f'    k={k[0]:>2} 浮盈档{k[1]}  n={n}  继续更好={wr:.3f}  平均差={md*100:+.4f}%')
    say(f'  → {"✅ 学到了可用的退出规则" if good else "❌ 学不到（状态定义需要换）"}')

    say('')
    say('【总判据】①②③④ 全过 = 机制合格，可以拿去真数据找牌；任一不过 = 先修机制')
    say(f'TOTAL {time.time()-t0:.1f}s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_synthetic.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))


if __name__ == '__main__':
    main()
