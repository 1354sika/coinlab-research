# -*- coding: utf-8 -*-
"""gambler_live · 端到端反复的赌徒（自己找牌，不是我先给牌）

与之前所有版本的根本区别：
  之前：我先定义 12 张牌 → 回测 → 看哪张好            （做统计报告）
  现在：给一堆**基元**，赌徒自己组合出牌 → 在时间流里不断
        观察/决策/记账/进化 → 循环本身让它长成会赌的东西

一个循环（每根 K 线都在转）：
  ┌─────────────────────────────────────────┐
  │ ① 观察：算基元（多尺度动量/波动/位置）      │
  │ ② 决策：种群里的基因各自投票 → 买/不买      │
  │ ③ 执行：模拟成交，记下入场价与时间          │
  │ ④ 结算：到期/触发退出，写进每个基因的战绩表   │
  │ ⑤ 进化：每 7 天，淘汰最差，最好的变异/交叉    │
  └─────────────────────────────────────────┘

基因 = 一张"牌"的可执行形式：
  (cond1, cond2, H, tp, sl)
  cond = (基元名, 方向, 阈值)  —— 从基元里自己挑，不是写死
  H/tp/sl 也由基因自己带，不在外面定

无前视三条硬规矩：
  1. 基元只用过去（rolling）
  2. 进化只用**已结算**的交易（没结算的不算战绩）
  3. 每天的分界是固定的，不允许"今天用明天的数据"

用法：LIVE_NSYM=6 python -X utf8 gambler_live.py
输出：evidence/gambler_live.txt / _trades.csv / _species.csv
"""
import os, sys, time, collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('LIVE_DATA', r'D:\中信证券\coinlab\data\klines_1m_1y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('LIVE_NSYM', '6'))
FEE = 0.0004
POS = 0.10
POP = int(os.environ.get('LIVE_POP', '60'))
BARS_PER_DAY = int(os.environ.get('LIVE_BPD', '1440'))   # 1m=1440, 5m=288
EVOLVE_EVERY = BARS_PER_DAY      # 每 1 天进化一次（原来 7 天太长，全样本只进化了 2 次）
MIN_TRADES = 12                # 基因至少要这么多已结算交易才有资格被评估
TRAIN = 0.60                   # 前 60% 只用于观察，不交易（让赌徒先看牌）
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


# ---------------------------------------------------------------- 基元
def build_primitives(C, V):
    """所有基元都只用过去；返回 {名: (T,S)} 或 1 维 (T,)"""
    T, S = C.shape
    P = {}
    for k in (5, 15, 60, 240, 1440):
        m = np.full((T, S), np.nan)
        ok = np.isfinite(C[k:]) & np.isfinite(C[:-k])
        v = np.full_like(C[k:], np.nan)
        v[ok] = C[k:][ok] / C[:-k][ok] - 1.0
        m[k:] = v
        P[f'r{k}'] = m
    # 波动率（20 根实现波动，滚动）
    r1 = np.zeros((T, S))
    r1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    r1 = np.nan_to_num(r1)
    vol = pd.DataFrame(r1).rolling(60, min_periods=20).std().values
    P['vol'] = vol
    # 区间位置（240 根内）
    hi = pd.DataFrame(C).rolling(240, min_periods=60).max().values
    lo = pd.DataFrame(C).rolling(240, min_periods=60).min().values
    P['rp'] = (C - lo) / np.maximum(hi - lo, 1e-12)
    # 成交额相对（量的 60 根均值比）
    q = pd.DataFrame(np.nan_to_num(V)).rolling(60, min_periods=20).mean().values
    vq = np.nan_to_num(V) / np.maximum(q, 1e-12)
    P['volr'] = np.clip(vq, 0, 10)
    # 每个基元的横截面分位（只用过去 1 天），让阈值可跨币比较
    for k in list(P):
        A = P[k]
        Q = np.full(A.shape, np.nan)
        for i in range(BARS_PER_DAY, T, BARS_PER_DAY):
            ref = A[i - BARS_PER_DAY:i].ravel()
            ref = ref[np.isfinite(ref)]
            if len(ref) < 200:
                continue
            order = np.sort(ref)
            hi_i = min(T, i + BARS_PER_DAY)
            seg = A[i:hi_i]
            ok = np.isfinite(seg)
            pos = np.full(seg.shape, np.nan)
            if ok.any():
                pos[ok] = np.searchsorted(order, seg[ok], side='right') / len(order)
            Q[i:hi_i] = pos
        P[k] = Q
    return P


NAMES = ['r5', 'r15', 'r60', 'r240', 'r1440', 'vol', 'rp', 'volr']


# ---------------------------------------------------------------- 基因
class Gene:
    __slots__ = ('c1', 'c2', 'H', 'tp', 'sl', 'wins', 'ret', 'n', 'id')

    def __init__(self, c1, c2, H, tp, sl, gid):
        self.c1, self.c2, self.H, self.tp, self.sl = c1, c2, H, tp, sl
        self.wins, self.ret, self.n, self.id = 0, 0.0, 0, gid

    @staticmethod
    def random(rng, gid):
        def cond():
            return (NAMES[rng.integers(len(NAMES))],
                    int(rng.integers(2)),                        # 0=>  1=<
                    float(rng.uniform(0.3, 0.9)))
        return Gene(cond(), cond(), int(rng.choice([30, 60, 120, 240, 480])),
                    float(rng.uniform(0.01, 0.05)),
                    float(-rng.uniform(0.005, 0.03)), gid)

    def clone(self, gid):
        # 继承战绩：否则克隆体 n=0，进化后要重新积累，导致后面几十天再也不进化
        g = Gene(self.c1, self.c2, self.H, self.tp, self.sl, gid)
        g.wins, g.ret, g.n = self.wins, self.ret, self.n
        return g

    def mutate(self, rng, gid):
        g = self.clone(gid)
        g.wins, g.ret, g.n = 0, 0.0, 0    # 变异体是新个体：清零战绩，必须重新证明自己
        r = rng.random()
        if r < 0.35:
            g.c1 = (NAMES[rng.integers(len(NAMES))], int(rng.integers(2)),
                    float(np.clip(g.c1[2] + rng.normal(0, 0.1), 0.2, 0.95)))
        elif r < 0.7:
            g.c2 = (NAMES[rng.integers(len(NAMES))], int(rng.integers(2)),
                    float(np.clip(g.c2[2] + rng.normal(0, 0.1), 0.2, 0.95)))
        elif r < 0.85:
            g.H = int(np.clip(g.H * rng.choice([0.5, 2.0]), 15, 1440))
        else:
            g.tp = float(np.clip(g.tp + rng.normal(0, 0.01), 0.005, 0.10))
            g.sl = float(np.clip(g.sl - abs(rng.normal(0, 0.005)), 0.003, 0.05))
        return g

    def desc(self):
        f = lambda c: f'{c[0]}{">" if c[1] == 0 else "<"}{c[2]:.2f}'
        return f'{f(self.c1)}&{f(self.c2)} H{self.H} tp{self.tp*100:.1f} sl{self.sl*100:.1f}'

    def signal(self, P, t, s):
        for name, d, thr in (self.c1, self.c2):
            v = P[name][t, s]
            if not np.isfinite(v):
                return False
            if d == 0 and not (v > thr):
                return False
            if d == 1 and not (v < thr):
                return False
        return True


# ---------------------------------------------------------------- 主循环
def main():
    t0 = time.time()
    say('=' * 104)
    say('gambler_live · 端到端反复的赌徒（自己找牌 | 在线进化 | 只做多）')
    say(f'  币 {NSYM} | 种群 {POP} | 每 1 天进化 | 每笔 {POS:.2f} 资金 | 费 {FEE:.2%}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close', 'volume'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    pc = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    po = df.pivot_table(index='t', columns='symbol', values='open').sort_index()[syms]
    pv = df.pivot_table(index='t', columns='symbol', values='volume').sort_index()[syms]
    C = pc.values.astype(np.float64); C[~np.isfinite(C) | (C <= 0)] = np.nan
    O = po.values.astype(np.float64); O[~np.isfinite(O) | (O <= 0)] = np.nan
    V = pv.values.astype(np.float64)
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    P = build_primitives(C, V)
    say(f'  基元算完 {len(NAMES)} 个  {time.time()-t0:.0f}s')

    start = int(T * TRAIN)
    rng = np.random.default_rng(20261019)
    gid = 0
    pop = []
    for _ in range(POP):
        pop.append(Gene.random(rng, gid)); gid += 1

    pending = {}          # s -> dict(gene, entry_t, entry_p, path)
    trades = []           # 全部成交（含基因 id）
    curve = [1.0]
    equity = 1.0
    gen_log = []
    species = []

    say('')
    say('  开始跑时间流（训练段只观察，不交易）...')
    for t in range(start, T - 2):
        # ---- 结算 ----
        for s in list(pending):
            ps = pending[s]
            px = C[t, s]
            if not np.isfinite(px):
                continue
            r = px / ps['entry_p'] - 1.0
            held = t - ps['entry_t'] + 1
            hit = (r >= ps['gene'].tp) or (r <= ps['gene'].sl) or (held >= ps['gene'].H)
            if hit:
                net = r - FEE
                g = ps['gene']
                g.n += 1
                g.ret += net
                if net > 0:
                    g.wins += 1
                equity *= (1 + net * POS)
                curve.append(equity)
                trades.append(dict(t=t, sym=s, gid=g.id, desc=g.desc(), ret=net, held=held,
                                   why='tp' if r >= g.tp else 'sl' if r <= g.sl else 'time'))
                del pending[s]

        # ---- 决策：UCB 选基因（没试过的先试；试过的按战绩 + 探索加成）----
        # 原来的写法要求 n>=3 才参与，结果初始基因全无战绩 ⇒ 永不交易 ⇒ 死锁。
        total = sum(g.n for g in pop) + 1
        best, bv = None, -1e18
        for g in pop:
            if g.n == 0:
                best = g          # 先试没试过的，保证每个基因都有机会出场
                break
            v = g.ret / g.n + 0.02 * np.sqrt(np.log(total) / g.n)
            if v > bv:
                best, bv = g, v
        if best is not None:
            for s in range(S):
                if s in pending:
                    continue
                if not best.signal(P, t, s):
                    continue
                px = O[t + 1, s]
                if not np.isfinite(px):
                    continue
                pending[s] = dict(gene=best, entry_t=t + 1, entry_p=px)

        # ---- 进化 ----
        if (t - start) % EVOLVE_EVERY == 0 and t > start:
            scored = [g for g in pop if g.n >= MIN_TRADES]
            if len(scored) >= 4:
                scored.sort(key=lambda g: g.ret / max(g.n, 1), reverse=True)
                keep = max(4, POP // 3)
                survivors, seen = [], set()
                for g in scored:                 # 去重：防止同一种基因占满种群（早熟收敛）
                    if g.desc() in seen:
                        continue
                    seen.add(g.desc())
                    survivors.append(g.clone(gid)); gid += 1
                    if len(survivors) >= keep:
                        break
                newpop = list(survivors)
                while len(newpop) < POP:
                    parent = survivors[rng.integers(len(survivors))]
                    newpop.append(parent.mutate(rng, gid)); gid += 1
                gen_log.append(dict(
                    t=t, gen=len(gen_log) + 1,
                    best=scored[0].ret / scored[0].n, best_desc=scored[0].desc(),
                    alive=len(scored), pop=len(pop),
                    mean=float(np.mean([g.ret / g.n for g in scored])),
                    equity=equity))
                pop = newpop
                for s in list(pending):
                    pending[s]['gene'] = pop[0]   # 换了代，持仓改挂新种群头名
                species.append(collections.Counter(g.c1[0] for g in pop))
                if len(gen_log) <= 6 or len(gen_log) % 5 == 0:
                    say(f'    第 {len(gen_log):>2} 代  t={t-start:>7}  '
                        f'可评估 {len(scored):>2}  均值 {gen_log[-1]["mean"]*100:+.3f}%  '
                        f'最好 {gen_log[-1]["best"]*100:+.3f}%  资金 {equity:.3f}')

    say(f'  时间流跑完  {time.time()-t0:.0f}s')

    # ---- 汇总 ----
    say('')
    say('【一】总体')
    if not trades:
        say('  ❌ 没有成交')
        return
    tr = pd.DataFrame(trades)
    a = tr.ret.values
    tt = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    say(f'  成交 {len(a)} 笔  单笔均值 {a.mean()*100:+.4f}%  中位 {np.median(a)*100:+.4f}%  '
        f'胜率 {(a>0).mean()*100:.1f}%  t={tt:+.2f}')
    pk = np.maximum.accumulate(np.array(curve))
    say(f'  资金 {equity:.4f}  最大回撤 {(np.array(curve)/pk-1).min()*100:+.1f}%')

    say('')
    say('【二】四段检验（judge 口径）')
    q = np.quantile(tr.t.values, [0, .25, .5, .75, 1.0])
    ok4 = True
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        g = tr[(tr.t >= lo) & (tr.t <= hi)] if i == 3 else tr[(tr.t >= lo) & (tr.t < hi)]
        x = g.ret.values
        if len(x) < 10:
            say(f'   段{i+1} 样本不足'); ok4 = False; continue
        s_ = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        say(f'   段{i+1}  n={len(x):>5}  均值={x.mean()*100:+.4f}%  t={s_:+.2f}')
        if x.mean() <= 0:
            ok4 = False
    say(f'  → {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say('')
    say('【三】赌徒长成了什么（最后一代活下来的物种）')
    final = [g for g in pop if g.n >= MIN_TRADES]
    final.sort(key=lambda g: g.ret / max(g.n, 1), reverse=True)
    for g in final[:5]:
        say(f'   {g.desc():<44} n={g.n:>4} 均值={g.ret/g.n*100:+.3f}%')
    say(f'   存活可评估基因 {len(final)} / {POP}')
    if species:
        say(f'   最后一代主基元分布: {dict(species[-1].most_common(5))}')

    say('')
    say('【四】进化轨迹（每代最好的单笔均值）')
    for r in gen_log[-8:]:
        say(f'   第{r["gen"]:>2}代  可评估 {r["alive"]:>2}  最好 {r["best"]*100:+.3f}%  '
            f'均值 {r["mean"]*100:+.3f}%  资金 {r["equity"]:.3f}  {r["best_desc"]}')

    say('')
    say('【判据】四段同正 + 最后几代均值持续上升 ⇒ 这个循环在长东西；否则先修循环')
    say(f'TOTAL {time.time()-t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_live.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    tr.to_csv(os.path.join(OUT, 'gambler_live_trades.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(gen_log).to_csv(os.path.join(OUT, 'gambler_live_gens.csv'),
                                 index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
