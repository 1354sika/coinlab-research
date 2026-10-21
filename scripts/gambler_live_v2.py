# -*- coding: utf-8 -*-
"""gambler_live_v2 · 让赌徒自己决定「怎么算活下来、怎么算赚」

相对 gambler_live.py 的三个改动：
  ① 外层进化按【自然季度】，不是每天 —— 快适应在数据上更差（EPOCH TEST：每期 −0.160pp）
  ② 基因自己携带判据：凯利分数 c、回撤容忍 λ、季度目标 g_t —— 判据由模型自己进化
  ③ 仓位 = c·凯利（p,b 用该基因已结算交易估），优势大下大注，没优势不下 —— 不再固定 10%
  ④ 裁判外生：真实费率、时间因果、样本外、SHUF 打乱对照（--shuffle）

五条不可谈判的硬约束（用户 2026-10-20 认可）：
  本金不可为负 / 真实成本 / 时间因果 / 样本外 / SHUF 必须存在

用法：
  LIVE_NSYM=20 python -X utf8 gambler_live_v2.py            # 真序列
  LIVE_NSYM=20 python -X utf8 gambler_live_v2.py --shuffle  # 打乱对照
"""
import os, sys, time, collections
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('LIVE_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('LIVE_NSYM', '20'))
BPD = int(os.environ.get('LIVE_BPD', '288'))          # 5m=288, 1m=1440
POP = int(os.environ.get('LIVE_POP', '60'))
TRAIN = float(os.environ.get('LIVE_TRAIN', '0.45'))
MIN_TRADES = int(os.environ.get('LIVE_MINTR', '6'))
POS_MAX = float(os.environ.get('LIVE_POSMAX', '0.10'))  # 单笔上限（本金占比）
FEE = 0.0004
SHUF = '--shuffle' in sys.argv
TAG = 'SHUF' if SHUF else 'REAL'
SEED = 20261020

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


# ---------------------------------------------------------------- 基元
NAMES = ['r5', 'r15', 'r60', 'r240', 'r1440', 'vol', 'rp', 'volr']


def build_primitives(C, V):
    """所有基元只用过去；最后转成过去 1 天的横截面分位。"""
    T, S = C.shape
    P = {}
    for k in (5, 15, 60, 240, 1440):
        m = np.full((T, S), np.nan)
        ok = np.isfinite(C[k:]) & np.isfinite(C[:-k])
        v = np.full_like(C[k:], np.nan)
        v[ok] = C[k:][ok] / C[:-k][ok] - 1.0
        m[k:] = v
        P[f'r{k}'] = m
    r1 = np.zeros((T, S))
    r1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    r1 = np.nan_to_num(r1)
    P['vol'] = pd.DataFrame(r1).rolling(60, min_periods=20).std().values
    hi = pd.DataFrame(C).rolling(240, min_periods=60).max().values
    lo = pd.DataFrame(C).rolling(240, min_periods=60).min().values
    P['rp'] = (C - lo) / np.maximum(hi - lo, 1e-12)
    q = pd.DataFrame(np.nan_to_num(V)).rolling(60, min_periods=20).mean().values
    P['volr'] = np.clip(np.nan_to_num(V) / np.maximum(q, 1e-12), 0, 10)
    for k in list(P):
        A = P[k]
        Q = np.full(A.shape, np.nan)
        for i in range(BPD, T, BPD):
            ref = A[i - BPD:i].ravel()
            ref = ref[np.isfinite(ref)]
            if len(ref) < 200:
                continue
            order = np.sort(ref)
            hi_i = min(T, i + BPD)
            seg = A[i:hi_i]
            ok = np.isfinite(seg)
            pos = np.full(seg.shape, np.nan)
            if ok.any():
                pos[ok] = np.searchsorted(order, seg[ok], side='right') / len(order)
            Q[i:hi_i] = pos
        P[k] = Q
    return P


# ---------------------------------------------------------------- 基因
class Gene:
    __slots__ = ('c1', 'c2', 'H', 'tp', 'sl', 'c', 'lam', 'g_t',
                 'wins', 'n', 'sum_win', 'sum_loss', 'sum_sq', 'n_win', 'n_loss', 'id', 'explored',
                 'q_n', 'q_log', 'q_peak', 'q_dd')

    def __init__(self, c1, c2, H, tp, sl, c, lam, g_t, gid):
        self.c1, self.c2, self.H, self.tp, self.sl = c1, c2, H, tp, sl
        self.c, self.lam, self.g_t = c, lam, g_t
        self.wins = self.n = self.n_win = self.n_loss = 0
        self.sum_win = self.sum_loss = 0.0
        self.sum_sq = 0.0
        self.id = gid
        self.explored = False
        self.q_n, self.q_log, self.q_peak, self.q_dd = 0, 0.0, 0.0, 0.0

    @staticmethod
    def random(rng, gid):
        def cond():
            return (NAMES[rng.integers(len(NAMES))], int(rng.integers(2)),
                    float(rng.uniform(0.3, 0.9)))
        return Gene(cond(), cond(), int(rng.choice([30, 60, 120, 240, 480])),
                    float(rng.uniform(0.01, 0.05)), float(-rng.uniform(0.005, 0.03)),
                    float(rng.uniform(0.1, 1.0)),      # 凯利分数：模型自己决定打几折
                    float(rng.uniform(0.5, 5.0)),      # 回撤容忍：模型自己的风险偏好
                    float(rng.uniform(0.0, 0.15)),     # 季度目标：模型自己的 KPI
                    gid)

    def clone(self, gid):
        g = Gene(self.c1, self.c2, self.H, self.tp, self.sl, self.c, self.lam, self.g_t, gid)
        g.wins, g.n, g.sum_win, g.sum_loss = self.wins, self.n, self.sum_win, self.sum_loss
        g.sum_sq = self.sum_sq
        g.n_win, g.n_loss = self.n_win, self.n_loss
        g.explored = self.explored
        return g

    def mutate(self, rng, gid):
        g = self.clone(gid)
        g.wins = g.n = g.n_win = g.n_loss = 0
        g.sum_win = g.sum_loss = 0.0
        g.sum_sq = 0.0
        g.explored = False
        g.q_n, g.q_log, g.q_peak, g.q_dd = 0, 0.0, 0.0, 0.0
        r = rng.random()
        if r < 0.28:
            g.c1 = (NAMES[rng.integers(len(NAMES))], int(rng.integers(2)),
                    float(np.clip(g.c1[2] + rng.normal(0, 0.1), 0.2, 0.95)))
        elif r < 0.56:
            g.c2 = (NAMES[rng.integers(len(NAMES))], int(rng.integers(2)),
                    float(np.clip(g.c2[2] + rng.normal(0, 0.1), 0.2, 0.95)))
        elif r < 0.70:
            g.H = int(np.clip(g.H * rng.choice([0.5, 2.0]), 15, 1440))
        elif r < 0.82:
            g.tp = float(np.clip(g.tp + rng.normal(0, 0.01), 0.005, 0.10))
            g.sl = float(np.clip(g.sl - abs(rng.normal(0, 0.005)), 0.003, 0.05))
        elif r < 0.92:
            g.c = float(np.clip(g.c + rng.normal(0, 0.1), 0.1, 1.0))
        elif r < 0.97:
            g.lam = float(np.clip(g.lam + rng.normal(0, 0.5), 0.5, 5.0))
        else:
            g.g_t = float(np.clip(g.g_t + rng.normal(0, 0.03), 0.0, 0.20))
        return g

    def desc(self):
        f = lambda c: f'{c[0]}{">" if c[1] == 0 else "<"}{c[2]:.2f}'
        return (f'{f(self.c1)}&{f(self.c2)} H{self.H} tp{self.tp*100:.1f} sl{self.sl*100:.1f} '
                f'c{self.c:.2f} λ{self.lam:.1f} gt{self.g_t*100:.0f}%')

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

    # ---- 注码：只在「优势 > 1.5×成本」时下，优势大下大注 ----
    def size(self):
        if self.n < MIN_TRADES:
            return min(0.02, POS_MAX)                       # 探索小注，先攒样本
        m = (self.sum_win + self.sum_loss) / self.n          # 净单笔均值（已扣费）
        var = max(self.sum_sq / self.n - m * m, 1e-8)
        edge = m - 1.5 * FEE                                # 优势门槛：1.5×成本
        if edge <= 0:
            # 没有优势：只留极小注继续采样，n 多了就完全不下
            return 0.005 if self.n < 50 else 0.0
        f = edge / var                                       # 连续凯利 f* = edge/σ²
        rel = self.n / (self.n + 20.0)                       # 样本少时向 0 收缩
        return float(np.clip(self.c * f * rel, 0.0, POS_MAX))

    def q_reset(self):
        self.q_n, self.q_log, self.q_peak, self.q_dd = 0, 0.0, 0.0, 0.0

    def q_fitness(self):
        """模型的季度成绩：log 增长 − λ·回撤 − 未达自定目标的惩罚。"""
        return self.q_log - self.lam * abs(self.q_dd) - 0.5 * max(0.0, self.g_t - self.q_log)


# ---------------------------------------------------------------- 打乱对照
def shuffle_panel(C, O, V, rng):
    """按天块重排 log 收益，价格保持连续、总漂移不变（修掉旧版压到 0 的 bug）。"""
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
    O2 = np.vstack([C2[:1], C2[:-1]])
    V2 = V[idx]
    return C2, O2, V2


# ---------------------------------------------------------------- 主循环
def main():
    t0 = time.time()
    say('=' * 108)
    say(f'gambler_live_v2 · 赌徒自己定判据 | 季度外层进化 | 优势大下大注  [{TAG}]')
    say(f'  {os.path.basename(DATA)}  币 {NSYM}  种群 {POP}  训练 {TRAIN:.0%}  '
        f'单笔上限 {POS_MAX:.0%}  费 {FEE:.2%}  每天 {BPD} 根')
    say('=' * 108)

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
    tidx = pc.index
    rng = np.random.default_rng(SEED)
    if SHUF:
        C, O, V = shuffle_panel(C, O, V, rng)
        say('  ⚠ 已打乱：按天块重排 log 收益（跨币共用同一顺序，总漂移不变）')
    T, S = C.shape
    say(f'  矩阵 {T:,} × {S} 币  {time.time()-t0:.0f}s')

    P = build_primitives(C, V)
    say(f'  基元算完 {len(NAMES)} 个  {time.time()-t0:.0f}s')

    # 自然季度边界
    q = pd.PeriodIndex(tidx, freq='Q')
    q_code = q.asi8
    q_start = set(int(i) for i in np.where(np.diff(q_code, prepend=q_code[0] - 1) != 0)[0])
    q_name = {i: str(q[i]) for i in q_start}

    start = int(T * TRAIN)
    gid = 0
    pop = [Gene.random(rng, i) for i in range(POP)]
    gid = POP
    pending = {}
    trades = []
    curve = [1.0]
    equity = 1.0
    eq_log = []          # (t, equity)
    gen_log = []
    quarter_stats = {}

    say('')
    say(f'  样本外从 t={start:,}（{tidx[start]}）开始；季度边界 {len(q_start)} 个')
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
            g = ps['gene']
            hit = (r >= g.tp) or (r <= g.sl) or (held >= g.H)
            if hit:
                net = r - FEE
                g.n += 1
                g.sum_sq += net * net
                if net > 0:
                    g.wins += 1; g.n_win += 1; g.sum_win += net
                else:
                    g.n_loss += 1; g.sum_loss += net
                w = ps['w']
                equity *= (1 + net * w)
                curve.append(equity)
                eq_log.append((t, equity))
                g.q_n += 1
                g.q_log += float(np.log(max(1 + net * w, 1e-9)))
                g.q_peak = max(g.q_peak, g.q_log)
                g.q_dd = min(g.q_dd, float(np.exp(g.q_log - g.q_peak) - 1.0))
                qq = int(q_code[t])
                st = quarter_stats.setdefault(qq, dict(n=0, sum=0.0, wsum=0.0,
                                                       eq_start=equity / max(1 + net * w, 1e-9),
                                                       peak=equity, dd=0.0))
                st['n'] += 1; st['sum'] += net; st['wsum'] += w
                st['peak'] = max(st['peak'], equity)
                st['dd'] = min(st['dd'], equity / st['peak'] - 1)
                trades.append(dict(t=t, sym=s, gid=g.id, desc=g.desc(), ret=net, w=w,
                                   held=held, why='tp' if r >= g.tp else 'sl' if r <= g.sl else 'time'))
                del pending[s]

        # ---- 决策：UCB ----
        total = sum(g.n for g in pop) + 1
        best, bv = None, -1e18
        for g in pop:
            if g.n == 0 and not g.explored:
                best = g
                break
            v = 1e9 if g.n == 0 else (g.sum_win + g.sum_loss) / g.n - 1.5 * np.sqrt(max(g.sum_sq / g.n - ((g.sum_win + g.sum_loss) / g.n) ** 2, 0) / g.n) + 0.02 * np.sqrt(np.log(total) / g.n)
            if v > bv:
                best, bv = g, v
        if best is not None:
            w = best.size()
            if w > 0:
                for s in range(S):
                    if s in pending:
                        continue
                    if not best.signal(P, t, s):
                        continue
                    px = O[t + 1, s]
                    if not np.isfinite(px):
                        continue
                    pending[s] = dict(gene=best, entry_t=t + 1, entry_p=px, w=w)
            best.explored = True

        # ---- 季度外层进化 ----
        if t in q_start and t > start:
            scored = [g for g in pop if g.q_n >= MIN_TRADES]
            # 季度末：先把所有基因的季度成绩定格
            for g in pop:
                if g.q_dd == 0.0 and g.q_n > 0:
                    g.q_dd = min(0.0, g.q_dd)
            if len(scored) >= 4:
                scored.sort(key=lambda g: g.q_fitness(), reverse=True)
                keep = max(4, POP // 3)
                survivors, seen = [], set()
                for g in scored:
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
                    t=t, q=q_name.get(t, str(q[t])), gen=len(gen_log) + 1,
                    best=scored[0].q_fitness(), best_desc=scored[0].desc(),
                    best_c=scored[0].c, best_lam=scored[0].lam, best_gt=scored[0].g_t,
                    alive=len(scored), equity=equity))
                pop = newpop
                for s in list(pending):
                    pending[s]['gene'] = pop[0]
                say(f'    {q_name.get(t, str(q[t]))} 换季进化：可评估 {len(scored)}  '
                    f'最好 J={gen_log[-1]["best"]:+.3f}  {gen_log[-1]["best_desc"]}  '
                    f'资金 {equity:.3f}')
            for g in pop:
                g.q_reset()

    say(f'  时间流跑完  {time.time()-t0:.0f}s')

    # ---- 汇总 ----
    say('')
    say(f'【一】总体 [{TAG}]')
    if not trades:
        say('  ❌ 没有成交')
        return
    tr = pd.DataFrame(trades)
    a = tr.ret.values
    tt = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    say(f'  成交 {len(a)} 笔  单笔均值 {a.mean()*100:+.4f}%  中位 {np.median(a)*100:+.4f}%  '
        f'胜率 {(a>0).mean()*100:.1f}%  t={tt:+.2f}')
    pk = np.maximum.accumulate(np.array(curve))
    say(f'  资金 {equity:.4f}  最大回撤 {(np.array(curve)/pk-1).min()*100:+.1f}%  '
        f'平均注码 {tr.w.mean():.4f}')

    say('')
    say(f'【二】按自然季度（{TAG}）—— 这就是「适应每个季度」的直接账')
    keys = sorted(quarter_stats)
    pos_q = 0
    n_q = 0
    for kk in keys:
        st = quarter_stats[kk]
        if st['n'] < 5:
            continue
        n_q += 1
        qtr = tr[tr.t.map(lambda x: int(q_code[x]) == kk)]
        gq = (1 + qtr.ret.values * qtr.w.values).prod() - 1 if len(qtr) else 0.0
        if gq > 0:
            pos_q += 1
        say(f'    {pd.Period(ordinal=int(kk), freq="Q")}  笔数 {st["n"]:>5}  '
            f'单笔 {st["sum"]/st["n"]*100:+.4f}%  季度资金 {gq*100:+.2f}%  '
            f'回撤 {st["dd"]*100:+.1f}%  均注 {st["wsum"]/st["n"]:.3f}')
    say(f'  → 正季度 {pos_q} / {n_q}（客观描述，不设门槛）')

    say('')
    say('【三】赌徒长成了什么（最后一代，含它自己定的判据）')
    final = [g for g in pop if g.n >= MIN_TRADES]
    final.sort(key=lambda g: (g.sum_win + g.sum_loss) / max(g.n, 1), reverse=True)
    for g in final[:5]:
        say(f'   {g.desc():<70} n={g.n:>4} 均值={(g.sum_win+g.sum_loss)/g.n*100:+.3f}%')
    say(f'   存活可评估基因 {len(final)} / {POP}')

    say('')
    say('【四】换季进化轨迹')
    for r in gen_log[-8:]:
        say(f'   {r["q"]}  J={r["best"]:+.3f}  可评估 {r["alive"]:>2}  '
            f'c={r["best_c"]:.2f} λ={r["best_lam"]:.1f} gt={r["best_gt"]*100:.0f}%  '
            f'资金 {r["equity"]:.3f}  {r["best_desc"]}')

    say('')
    say('【判据】季度为正比例 / SHUF 是否更差 / 最差单季回撤 —— 三个数由账本客观给出')
    say(f'TOTAL {time.time()-t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'gambler_live_v2_{TAG}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    tr.to_csv(os.path.join(OUT, f'gambler_live_v2_{TAG}_trades.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(gen_log).to_csv(os.path.join(OUT, f'gambler_live_v2_{TAG}_gens.csv'),
                                 index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
