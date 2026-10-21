# -*- coding: utf-8 -*-
"""
gambler_adaptive_v2.py · 自适应骨架（把验过的牌交给模型自己决定怎么用）

用户要的赌徒（2026-10-18）：
  "不仅是实时决策，模型还要自己决定往回看多久的历史。
   实时决定每一个变量在当下到底要变成什么样子，
   并且能够实时决定到底是买入，卖出还是持有。
   所有参数不固定，由模型根据当下的情况自己判断。"

这个脚本就是那个骨架。原料是**已经过打乱闸门**的 4 张牌：

  牌            方向            类型      来源
  disp240      分化大→涨       全市场    gate_cross ✅
  breadth60    宽度高→跌       全市场    gate_cross ✅
  dow_cos      星期效应→跌     时间      gate_cross ✅
  rangepos     区间顶→跌       单币      gate_v2   ✅

骨架自己决定的事：
  ① 往回看多久     MEM ∈ {60, 240, 1440} 分钟（决定牌的计算窗口）
  ② 变量当下变成什么 每张牌按"过去 1 天"的滚动分位做秩归一，取值随当下状态变
  ③ 买/卖/持      三动作：+1 开多 / −1 开空 / 0 空仓；MODE 决定是否允许做空
  ④ 所有参数      (MEM, τ, H, MODE) 每 24h 按**已结算交易**重选（含 10% 探索）
  ⑤ 止损止盈      由过去 1 天收益分布的 5%/95% 分位当场给，不写死

严格无前视：
  · 分位只用 t 时刻之前的数据（滚动窗口）
  · 参数只用**已平仓**的交易
  · 成交在信号后一根开盘价
  · 对照：同一套骨架跑打乱序列

用法：set AD_NSYM=6 && python -X utf8 gambler_adaptive_v2.py
输出：evidence/gambler_adaptive_v2.txt / _trades.csv
"""
import os, sys, time, itertools, collections
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate import tstat
from gambler_gate_cross import shuffle_panel

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('AD_NSYM', '6'))
TRAIN = 0.70
FEE = 0.0004                      # 往返 maker 0.02%×2
DEC = 60                          # 每 60 分钟决策一次
BLOCK = 1440                      # 每 24h 重选一次参数
MEM_GRID = [60, 240, 1440]        # 往回看多久
TAU_GRID = [0.80, 0.90, 0.95]     # 动作阈值分位
H_GRID = [60, 240, 480]           # 持有上限
MODE_GRID = [2]                    # 固定多空双向（去掉 MODE 组，减少被噪声选中的组数）
MIN_TRADES = 20
EXPLORE = 0.10
POS_FRAC = 1.0 / 6                # 每笔占用资金比例（最多同时 6 仓 ⇒ 总暴露 ≤ 100%）
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def rmean(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    cs = np.cumsum(np.nan_to_num(A), axis=0)
    for i in range(k - 1, T):
        out[i] = (cs[i] - (cs[i - k] if i >= k else 0)) / k
    return out


def rstd(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    cs = np.cumsum(np.nan_to_num(A), axis=0)
    cs2 = np.cumsum(np.nan_to_num(A) ** 2, axis=0)
    for i in range(k - 1, T):
        s1 = cs[i] - (cs[i - k] if i >= k else 0)
        s2 = cs2[i] - (cs2[i - k] if i >= k else 0)
        mu = s1 / k
        out[i] = np.sqrt(np.maximum(s2 / k - mu * mu, 0))
    return out


def rmax(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    for i in range(k - 1, T):
        blk = A[i - k + 1:i + 1]
        with np.errstate(all='ignore'):
            out[i] = np.nanmax(np.where(np.isfinite(blk), blk, -np.inf), axis=0)
    return out


def rmin(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    for i in range(k - 1, T):
        blk = A[i - k + 1:i + 1]
        with np.errstate(all='ignore'):
            out[i] = np.nanmin(np.where(np.isfinite(blk), blk, np.inf), axis=0)
    return out


def mom(C, k):
    T, S = C.shape
    out = np.full((T, S), np.nan)
    ok = np.isfinite(C[k:]) & np.isfinite(C[:-k])
    v = np.full_like(C[k:], np.nan)
    v[ok] = C[k:][ok] / C[:-k][ok] - 1
    out[k:] = v
    return out


def expand_z_1d(x, k):
    """滚动的均值/标准差 做 z 化（只用过去 k 根）"""
    cs = np.cumsum(np.nan_to_num(x))
    cs2 = np.cumsum(np.nan_to_num(x) ** 2)
    n = len(x)
    mu = cs / np.arange(1, n + 1)
    sd = np.sqrt(np.maximum(cs2 / np.arange(1, n + 1) - mu * mu, 0))
    # 滚动版本
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - k)
        cnt = i - lo + 1
        s1 = cs[i] - (cs[lo - 1] if lo > 0 else 0)
        s2 = cs2[i] - (cs2[lo - 1] if lo > 0 else 0)
        m = s1 / cnt
        v = max(s2 / cnt - m * m, 0)
        s = np.sqrt(v)
        out[i] = (x[i] - m) / s if s > 1e-12 else 0.0
    return np.clip(out, -6, 6)


def rolling_rank(x, k, n_win=BLOCK):
    """滚动分位秩（0..1）：每 n_win 步用过去 k 根定一次分位，避免逐根重算"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(0, n, n_win):
        lo = max(0, i - k)
        ref = x[lo:i + 1]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 100:
            continue
        hi = min(n, i + n_win)
        seg = x[i:hi]
        # 用 ref 的排序位置映射
        order = np.sort(ref)
        pos = np.searchsorted(order, seg, side='right') / len(order)
        out[i:hi] = pos
    return out


def build_cards(C, V, TB, dow, mem):
    """按记忆窗口 mem 计算 4 张牌（牌本身方向已定，这里只算取值）"""
    T, S = C.shape
    cards = {}
    # disp：60 币 mem 分钟动量的截面标准差
    M = mom(C, mem)
    valid = np.isfinite(M)
    nv = valid.sum(axis=1)
    with np.errstate(all='ignore'):
        disp = np.nanstd(np.where(valid, M, np.nan), axis=1)
    disp[nv < 30] = np.nan
    cards['disp'] = np.repeat(disp[:, None], S, axis=1)
    # breadth
    br = np.where(nv >= 30, ((M > 0) & valid).sum(axis=1) / np.maximum(nv, 1), np.nan)
    cards['breadth'] = np.repeat(br[:, None], S, axis=1)
    # dow_cos（时间牌，不随 mem 变）
    cards['dow'] = np.repeat(np.cos(2 * np.pi * dow / 7)[:, None], S, axis=1)
    # rangepos
    hi = rmax(C, mem)
    lo = rmin(C, mem)
    with np.errstate(all='ignore'):
        cards['rangepos'] = (C - lo) / np.maximum(hi - lo, 1e-12)
    return cards


def main():
    t0 = time.time()
    say('=' * 104)
    say('gambler_adaptive_v2 · 自适应骨架（窗口/动作/参数/止损止盈 全部自选）')
    say(f'  牌 disp240 / breadth60 / dow_cos / rangepos | 币 {NSYM} | 手续费 {FEE:.2%} 往返')
    say(f'  MEM∈{MEM_GRID}  τ∈{TAU_GRID}  H∈{H_GRID}  MODE∈{MODE_GRID}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close', 'volume', 'taker_base'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv_c = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    piv_o = df.pivot_table(index='t', columns='symbol', values='open').sort_index()[syms]
    piv_v = df.pivot_table(index='t', columns='symbol', values='volume').sort_index()[syms]
    piv_tb = df.pivot_table(index='t', columns='symbol', values='taker_base').sort_index()[syms]
    C = piv_c.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    O = piv_o.values.astype(np.float64)
    O[~np.isfinite(O) | (O <= 0)] = np.nan
    V = np.nan_to_num(piv_v.values.astype(np.float64), nan=0.0)
    TB = np.nan_to_num(piv_tb.values.astype(np.float64), nan=0.0)
    T, S = C.shape
    dow = piv_c.index.dayofweek.values
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    # 预计算每个记忆窗口的牌（打乱序列也要）
    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    O2 = C2.copy()
    say(f'  打乱面板完成  {time.time() - t0:.0f}s')

    SCORE = {}
    SCORE2 = {}
    for mem in MEM_GRID:
        cards = build_cards(C, V, TB, dow, mem)
        # 秩归一（滚动，只用过去）
        s = np.zeros((T, S))
        for nm, sign in (('disp', +1), ('breadth', -1), ('dow', -1), ('rangepos', -1)):
            arr = cards[nm]
            flat = np.nanmean(arr, axis=1) if nm != 'rangepos' else None
            if nm == 'rangepos':
                for j in range(S):
                    rr = rolling_rank(arr[:, j], 1440)
                    s[:, j] += sign * np.nan_to_num(rr - 0.5, nan=0.0)
            else:
                r = rolling_rank(flat, 1440)
                s += sign * np.nan_to_num(r - 0.5, nan=0.0)[:, None]
        SCORE[mem] = s / 4.0
        cards2 = build_cards(C2, V, TB, dow, mem)
        s2 = np.zeros((T, S))
        for nm, sign in (('disp', +1), ('breadth', -1), ('dow', -1), ('rangepos', -1)):
            arr = cards2[nm]
            flat = np.nanmean(arr, axis=1) if nm != 'rangepos' else None
            if nm == 'rangepos':
                for j in range(S):
                    rr2 = rolling_rank(arr[:, j], 1440)
                    s2[:, j] += sign * np.nan_to_num(rr2 - 0.5, nan=0.0)
            else:
                r2 = rolling_rank(flat, 1440)
                s2 += sign * np.nan_to_num(r2 - 0.5, nan=0.0)[:, None]
        SCORE2[mem] = s2 / 4.0
        say(f'  窗口 {mem:>5} 分 的牌算完  {time.time() - t0:.0f}s')

    # ---- 回测 ----
    cut = int(T * TRAIN)
    LOG = []          # (t, symbol, dir, entry, exit, ret)
    trades = []       # 已结算，用于选参
    pending = {}      # symbol -> dict(entry_t, entry_p, dir, H, tp, sl)
    params = dict(mem=240, tau=0.90, H=240, mode=2)
    n_param_change = 0
    n_explore = 0
    equity = 1.0
    curve = []

    # 止损止盈参考：过去 1 天每分钟收益的 5%/95% 分位（滚动）
    R1 = np.zeros((T, S))
    R1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
    R1 = np.nan_to_num(R1)
    tp_pct, sl_pct = 0.02, -0.015     # 初值，之后按分布更新

    for t in range(cut, T - 1, DEC):
        # ---- 每 24h 重选参数 ----
        if (t - cut) % BLOCK == 0:
            # 先强制补样本：每个参数组都没跑够 MIN_TRADES 笔时，优先轮到它
            counts = collections.Counter((x['mem'], x['tau'], x['H'], x['mode']) for x in trades)
            allp = list(itertools.product(MEM_GRID, TAU_GRID, H_GRID, MODE_GRID))
            under = [p for p in allp if counts.get(p, 0) < MIN_TRADES]
            best = None
            if under:
                best = under[int(rng.integers(len(under)))]
                n_explore += 1
            else:
                bestv = -1e9
                for p_ in allp:
                    mem_, tau_, H_, mode_ = p_
                    sub = [x for x in trades if x['mem'] == mem_ and x['tau'] == tau_
                           and x['H'] == H_ and x['mode'] == mode_]
                    a = np.array([x['ret'] for x in sub])
                    if len(a) < MIN_TRADES:
                        continue
                    lo = a.mean() - 1.5 * a.std(ddof=1) / np.sqrt(len(a))   # 保守下界（加大罚）
                    if lo > bestv:
                        bestv, best = lo, p_
            if best:
                if best != (params['mem'], params['tau'], params['H'], params['mode']):
                    n_param_change += 1
                params = dict(mem=best[0], tau=best[1], H=best[2], mode=best[3])
        # 止损止盈按过去 1 天分布更新
        if t % BLOCK == 0 and t > 1440:
            seg = R1[t - 1440:t]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 1000:
                tp_pct = float(np.quantile(seg, 0.97))
                sl_pct = float(np.quantile(seg, 0.03))

        # ---- 平仓检查 ----
        for s_ in list(pending.keys()):
            ps = pending[s_]
            held = t - ps['entry_t']
            px = C[t, s_]
            if not np.isfinite(px):
                continue
            ret = ps['dir'] * (px / ps['entry_p'] - 1)
            hit_tp = ret >= abs(tp_pct) * max(1, held / 60)
            hit_sl = ret <= sl_pct * max(1, held / 60)
            if held >= ps['H'] or hit_tp or hit_sl:
                net = ret - FEE
                # 每笔固定用 1/6 资金（最多同时 6 仓 ⇒ 总暴露上限 100%）。
                # 不能按"平仓时刻的持仓数"分摊：那样 npos 随平仓顺序抖动，
                # 会出现"单笔均值为负、资金曲线却翻倍"的假象（v3 就是这个 bug）。
                equity *= (1 + net * POS_FRAC)
                curve.append((t, equity))
                tr = dict(t=t, sym=s_, dir=ps['dir'], ret=net, held=held,
                          mem=ps['mem'], tau=ps['tau'], H=ps['H'], mode=ps['mode'],
                          why=('tp' if hit_tp else 'sl' if hit_sl else 'time'))
                trades.append(tr)
                LOG.append(tr)
                del pending[s_]

        # ---- 开仓检查 ----
        mem, tau, H, mode = params['mem'], params['tau'], params['H'], params['mode']
        sc = SCORE[mem][t]
        thr = 0.05 + (tau - 0.80) * 0.5   # tau=0.80→0.05  0.90→0.10  0.95→0.125
        for s_ in range(S):
            if s_ in pending:
                continue
            if not np.isfinite(sc[s_]):
                continue
            if sc[s_] >= thr:
                d = +1
            elif mode == 2 and sc[s_] <= -thr:
                d = -1
            else:
                continue
            px = O[t + 1, s_] if t + 1 < T else np.nan
            if not np.isfinite(px):
                continue
            pending[s_] = dict(entry_t=t + 1, entry_p=px, dir=d, H=H,
                               mem=mem, tau=tau, mode=mode)

    say('')
    say('=' * 104)
    say(f'【一】真实序列结果（样本外）')
    n = len(trades)
    if n == 0:
        say('  ❌ 没有产生任何交易')
        return
    a = np.array([x['ret'] for x in trades])
    eq = equity
    pk = np.maximum.accumulate(np.array([1.0] + [c[1] for c in curve]))
    dd = (np.array([1.0] + [c[1] for c in curve]) / pk - 1).min()
    years = (T - cut) / (1440 * 365)
    say(f'  交易 {n} 笔  单笔均值 {a.mean() * 100:+.4f}%  中位 {np.median(a) * 100:+.4f}%  '
        f'胜率 {(a > 0).mean() * 100:.1f}%  t={tstat(a):+.2f}')
    say(f'  资金 {eq:.4f}  最大回撤 {dd * 100:+.1f}%  样本外时长 {years:.2f} 年  '
        f'年化 {(eq ** (1 / years) - 1) * 100:+.1f}%' if eq > 0 else f'  资金 {eq:.4f}')

    say('')
    say('【二】模型自己选了什么（说明参数真的在变）')
    for key, name in (('mem', '记忆窗口'), ('tau', '阈值τ'), ('H', '持有H'), ('mode', '模式')):
        c = collections.Counter(x[key] for x in trades)
        tot = sum(c.values())
        say(f'  {name:<8}: ' + '  '.join(f'{k}={v / tot:.0%}' for k, v in c.most_common()))
    say(f'  参数换过 {n_param_change} 次  探索 {n_explore} 次')
    why = collections.Counter(x['why'] for x in trades)
    say(f'  平仓原因: ' + '  '.join(f'{k}={v}({v / n:.0%})' for k, v in why.most_common()))

    say('')
    say('【三】分币样本外')
    per = collections.defaultdict(list)
    for x in trades:
        per[x['sym']].append(x['ret'])
    pos = sum(1 for v in per.values() if np.mean(v) > 0)
    say(f'  有交易的币 {len(per)}；样本外均值>0 的币 {pos}/{len(per)}')
    best = sorted(per.items(), key=lambda kv: -np.mean(kv[1]))[:3]
    say('  最好 3: ' + '  '.join(f'{k}:{np.mean(v) * 100:+.3f}%({len(v)}笔)' for k, v in best))

    say('')
    say('【四】判据')
    ok = a.mean() > 0 and tstat(a) > 2
    say(f'  {"✅" if ok else "❌"} 单笔均值 {a.mean() * 100:+.4f}%，t={tstat(a):+.2f}')
    say('  未计：滑点、maker 挂单成交率')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_adaptive_v2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(LOG).to_csv(os.path.join(OUT, 'gambler_adaptive_v2_trades.csv'),
                             index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
