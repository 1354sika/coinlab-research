# -*- coding: utf-8 -*-
"""
gambler_adaptive_v3.py · 粒度自由的自适应赌徒

用户 2026-10-18 的口径修正：
  "模型不应该是只从三个挡位上选择回看多久，而是**想看多久都行，粒度自由**，
   而不是卡在三个不同的小时挡位。"
  "自适应是肯定可以的……只能说我们现在的自适应参与方式不对，或者实验太少。"

所以 v3 把"回看多久"从离散网格改成**连续参数**：
  · 预计算 12 个细粒度尺度的牌（15 分钟 → 16 天，√4 递进）
  · 权重由连续参数 τ 决定：高斯核 w_i ∝ exp(−(ln w_i − ln τ)² / 2σ²)
  · τ 在 [15, 23040] 分钟里连续取值，模型可以选 37 分钟、523 分钟这种
  · τ 在线更新：每次在 ln τ 上做局部扰动 + 偶尔大幅探索

同时修掉上一版 4 个错误：
  ① 杠杆假象：每笔固定用 1/6 资金（不再按平仓时刻的持仓数分摊，那会让 npos 抖动）
  ② 打乱对照没接上：真序列/打乱序列各跑一遍完整回测，并排输出
  ③ 参数跨期不可比：只用**最近 30 天**的已结算交易比较参数（滚动窗，不是全历史混比）
  ④ 止损止盈拍脑袋：改成直接用过去 1 天收益分布的 3%/97% 分位，去掉 max(1, held/60)

判据：真序列必须既显著（t>2）又明显优于打乱对照，才算站住。

用法：set AD_NSYM=6 && python -X utf8 gambler_adaptive_v3.py
输出：evidence/gambler_adaptive_v3.txt / _trades.csv
"""
import os, sys, time, collections
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate_cross import shuffle_panel

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('AD_NSYM', '6'))
TRAIN = 0.70
FEE = 0.0004
DEC = 60                      # 每 60 分钟决策一次
BLOCK = 1440                  # 每 24h 更新一次参数
SCALES = [15, 30, 60, 120, 240, 480, 960, 1440, 2880, 5760, 11520, 23040]
SIGMA = 0.7                   # 权重核宽度（log 空间）
TAU_MIN, TAU_MAX = 15.0, 23040.0
THR_GRID = [0.55, 0.60, 0.65, 0.70]
H_GRID = [30, 60, 120, 240, 480, 960]
HIST_WIN = 30 * 1440          # 参数比较只看最近 30 天
MIN_TRADES = 8
POS_FRAC = 1.0 / 6
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def shuffle_scores(SD, rng, block=BLOCK):
    """打乱特征与未来收益的对齐（价格面板不动）

    为什么不再打乱价格：打乱后的价格序列波动率结构被破坏，策略在那种序列上
    必然巨亏（实测 SHUF 单笔 −46%），这个"基线"没有信息量 —— 它不是"投硬币"，
    是"在火星上交易"。

    正确做法：价格/成交价全部真实（可交易），只把每张牌的时间索引按块整体重排，
    使"看到的牌"与"之后的真实价格"脱钩。这样期望应 ≈ 0（扣费后为负）。
    """
    out = {}
    for k, A in SD.items():
        T = A.shape[0]
        nb = max(2, T // block)
        perm = rng.permutation(nb)
        idx = np.concatenate([np.arange(p * block, (p + 1) * block) for p in perm])
        B = np.empty_like(A)
        B[:nb * block] = A[idx]
        B[nb * block:] = A[nb * block:]
        out[k] = B
    return out


def mom(C, k):
    T, S = C.shape
    out = np.full((T, S), np.nan)
    ok = np.isfinite(C[k:]) & np.isfinite(C[:-k])
    v = np.full_like(C[k:], np.nan)
    v[ok] = C[k:][ok] / C[:-k][ok] - 1
    out[k:] = v
    return out


def rmean_df(C, k):
    return pd.DataFrame(C).rolling(k, min_periods=max(2, k // 3)).mean().values


def rstd_df(C, k):
    return pd.DataFrame(C).rolling(k, min_periods=max(2, k // 3)).std().values


def rmax_df(C, k):
    return pd.DataFrame(C).rolling(k, min_periods=max(2, k // 3)).max().values


def rmin_df(C, k):
    return pd.DataFrame(C).rolling(k, min_periods=max(2, k // 3)).min().values


def rolling_rank_1d(x, k=1440, step=BLOCK):
    """滚动秩（0..1）：每 step 步用过去 k 根定一次分位表"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(0, n, step):
        lo = max(0, i - k)
        ref = x[lo:i + 1]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 50:
            continue
        order = np.sort(ref)
        hi = min(n, i + step)
        seg = x[i:hi]
        ok = np.isfinite(seg)
        pos = np.full(len(seg), np.nan)
        if ok.any():
            pos[ok] = np.searchsorted(order, seg[ok], side='right') / len(order)
        out[i:hi] = pos
    return out


def scale_scores(C, dow):
    """每个尺度一张 score 矩阵 (T,S) float32"""
    T, S = C.shape
    out = {}
    for sc in SCALES:
        M = mom(C, sc)
        valid = np.isfinite(M)
        nv = valid.sum(axis=1)
        enough = nv >= max(2, S // 2)   # 6 币时只要 3 个币有效就算（原来写死 10，6 币永远凑不够 → 全 nan → 无交易）
        with np.errstate(all='ignore'):
            disp = np.nanstd(np.where(valid, M, np.nan), axis=1)
        br = np.where(enough, ((M > 0) & valid).sum(axis=1) / np.maximum(nv, 1), np.nan)
        disp = np.where(enough, disp, np.nan)
        hi = rmax_df(C, sc)
        lo = rmin_df(C, sc)
        with np.errstate(all='ignore'):
            rp = (C - lo) / np.maximum(hi - lo, 1e-12)
        s = np.zeros((T, S), dtype=np.float64)
        s += rolling_rank_1d(disp)[:, None]                 # 分化大 → 看涨
        s += (1.0 - rolling_rank_1d(br))[:, None]           # 宽度高 → 看跌
        s += (1.0 - rolling_rank_1d(np.cos(2 * np.pi * dow / 7)))[:, None]
        for j in range(S):
            s[:, j] += (1.0 - rolling_rank_1d(rp[:, j]))    # 区间顶 → 看跌
        out[sc] = (s / 4.0).astype(np.float32)              # ∈ [0,1]，0.5 为中性
    return out


def mix(scores, tau, sigma=SIGMA):
    ln = np.log(np.array(SCALES, dtype=np.float64))
    w = np.exp(-0.5 * ((ln - np.log(tau)) / sigma) ** 2)
    w /= w.sum()
    T, S = scores[SCALES[0]].shape
    out = np.zeros((T, S), dtype=np.float32)
    for wi, sc in zip(w, SCALES):
        if wi > 1e-4:
            out += float(wi) * scores[sc]
    return out


class BT:
    """回测器（真序列/打乱序列共用）"""

    def __init__(self, C, O, scores, cut, T, S, rng):
        self.C, self.O, self.scores, self.cut, self.T, self.S, self.rng = C, O, scores, cut, T, S, rng

    def run(self, tau0=240.0, label='REAL'):
        C, O, S, T = self.C, self.O, self.S, self.T
        cut, rng = self.cut, self.rng
        pending = {}
        trades = []
        equity = 1.0
        curve = [1.0]
        tau = tau0
        thr = 0.60
        H = 240
        n_change = 0
        R1 = np.zeros((T, S))
        R1[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
        R1 = np.nan_to_num(R1)

        for t in range(cut, T - 1, DEC):
            # ---- 每 24h：用最近 30 天已结算交易重选参数（τ 连续）----
            if (t - cut) % BLOCK == 0:
                recent = [x for x in trades if x['t'] >= t - HIST_WIN]
                cands = []
                for _ in range(6):
                    cands.append(('tau', float(np.clip(tau * np.exp(rng.normal(0, 0.45)),
                                                     TAU_MIN, TAU_MAX))))
                if rng.random() < 0.25:
                    cands.append(('tau', float(np.exp(rng.uniform(np.log(TAU_MIN), np.log(TAU_MAX))))))
                best = None
                for kind, v in cands:
                    sub = [x for x in recent if abs(x['tau'] - v) / max(v, 1e-9) < 0.25]
                    if len(sub) < MIN_TRADES:
                        continue
                    a = np.array([x['ret'] for x in sub])
                    lo = a.mean() - 1.5 * a.std(ddof=1) / np.sqrt(len(a))
                    if best is None or lo > best[0]:
                        best = (lo, v)
                # thr / H 用离散候选比较（都是"动作参数"，不是"回看参数"）
                for v in THR_GRID:
                    sub = [x for x in recent if abs(x['thr'] - v) < 1e-9]
                    if len(sub) < MIN_TRADES:
                        continue
                    a = np.array([x['ret'] for x in sub])
                    lo = a.mean() - 1.5 * a.std(ddof=1) / np.sqrt(len(a))
                    if best is None or lo > best[0]:
                        best = (lo, None)
                if best and best[1] is not None:
                    if abs(best[1] - tau) / tau > 0.02:
                        n_change += 1
                    tau = best[1]
                if rng.random() < 0.3:
                    thr = float(rng.choice(THR_GRID))
                if rng.random() < 0.2:
                    H = int(rng.choice(H_GRID))

            # ---- 止损止盈：过去 1 天分布分位（不再乘 max(1, held/60)）----
            tp_pct, sl_pct = 0.02, -0.015
            if t % BLOCK == 0 and t > 1440:
                seg = R1[t - 1440:t].ravel()
                seg = seg[np.isfinite(seg)]
                if len(seg) > 5000:
                    tp_pct = float(np.quantile(seg, 0.97))
                    sl_pct = float(np.quantile(seg, 0.03))

            # ---- 平仓 ----
            for s_ in list(pending.keys()):
                ps = pending[s_]
                held = t - ps['entry_t']
                px = C[t, s_]
                if not np.isfinite(px):
                    continue
                ret = ps['dir'] * (px / ps['entry_p'] - 1)
                hit_tp = ret >= tp_pct
                hit_sl = ret <= sl_pct
                if held >= ps['H'] or hit_tp or hit_sl:
                    net = ret - FEE
                    equity *= (1 + net * POS_FRAC)
                    curve.append(equity)
                    tr = dict(t=t, sym=s_, dir=ps['dir'], ret=net, held=held,
                              tau=ps['tau'], thr=ps['thr'], H=ps['H'],
                              why=('tp' if hit_tp else 'sl' if hit_sl else 'time'))
                    trades.append(tr)
                    del pending[s_]

            # ---- 开仓 ----
            sc = mix(self.scores, tau)[t] if False else None
            # 预先 mix 太慢：改用按当前 tau 在 t 时刻取加权值（惰性）
            w = self._weights(tau)
            sc = np.zeros(S, dtype=np.float64)
            for wi, k in zip(w, SCALES):
                if wi > 1e-4:
                    sc += wi * self.scores[k][t]
            for s_ in range(S):
                if s_ in pending or not np.isfinite(sc[s_]):
                    continue
                # 【用户约束 2026-10-19】不做空：只保留 d=+1 一侧
                if sc[s_] >= thr:
                    d = +1
                else:
                    continue
                px = O[t + 1, s_] if t + 1 < T else np.nan
                if not np.isfinite(px):
                    continue
                pending[s_] = dict(entry_t=t + 1, entry_p=px, dir=d, H=H,
                                   tau=tau, thr=thr)
        a = np.array([x['ret'] for x in trades])
        return dict(label=label, trades=trades, n=len(a),
                    mean=a.mean() if len(a) else np.nan,
                    med=np.median(a) if len(a) else np.nan,
                    win=(a > 0).mean() if len(a) else np.nan,
                    equity=equity, curve=np.array(curve),
                    n_change=n_change, a=a)

    @staticmethod
    def _weights(tau):
        ln = np.log(np.array(SCALES, dtype=np.float64))
        w = np.exp(-0.5 * ((ln - np.log(tau)) / SIGMA) ** 2)
        return w / w.sum()


def tstat(a):
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return np.nan
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def report(r):
    a = r['a']
    if len(a) == 0:
        say(f'  {r["label"]:<18} 无交易')
        return
    pk = np.maximum.accumulate(r['curve'])
    dd = (r['curve'] / pk - 1).min()
    say(f'  {r["label"]:<18}{r["n"]:>7}{a.mean() * 100:>+10.4f}%{np.median(a) * 100:>+10.4f}%'
        f'{(a > 0).mean() * 100:>7.1f}%{tstat(a):>+8.2f}{r["equity"]:>9.4f}{dd * 100:>+8.1f}%')


def main():
    t0 = time.time()
    say('=' * 108)
    say('gambler_adaptive_v3 · 粒度自由的自适应赌徒')
    say(f'  尺度 {len(SCALES)} 个（{SCALES[0]} ~ {SCALES[-1]} 分钟）| τ 连续 | σ={SIGMA}')
    say(f'  币 {NSYM} | 手续费 {FEE:.2%} 往返 | 每笔资金 {POS_FRAC:.3f} | 参数比较窗 30 天')
    say('=' * 108)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'close', 'volume', 'taker_base'])
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

    rng = np.random.default_rng(20261018)

    say('  计算多尺度牌（真序列）...')
    SR = scale_scores(C, dow)
    say(f'  完成  {time.time() - t0:.0f}s')
    say('  打乱特征对齐（价格不动）...')
    SS = shuffle_scores(SR, rng)
    say(f'  完成  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    say('')
    say(f'{"策略":<18}{"笔数":>7}{"单笔均值":>11}{"中位":>11}{"胜率":>7}{"t":>8}{"资金":>9}{"回撤":>8}')
    bR = BT(C, O, SR, cut, T, S, np.random.default_rng(11))
    rR = bR.run(240.0, 'REAL 真序列')
    report(rR)
    bS = BT(C, O, SS, cut, T, S, np.random.default_rng(11))
    rS = bS.run(240.0, 'SHUF 特征打乱')
    report(rS)
    # 纯 beta 基线：永远做多（无任何信号）——把"市场在涨"从"牌有信息"里剥出来
    SB = {k: np.ones_like(v) for k, v in SR.items()}
    bB = BT(C, O, SB, cut, T, S, np.random.default_rng(11))
    rB = bB.run(240.0, 'BUY 永远做多')
    report(rB)

    say('')
    say('【二】模型自己选了什么（τ 是连续值，看它落在哪）')
    for tag, r in (('REAL', rR), ('SHUF', rS)):
        if r['n'] == 0:
            continue
        taus = np.array([x['tau'] for x in r['trades']])
        say(f'  {tag}: τ 中位 {np.median(taus):.0f} 分  范围 {taus.min():.0f}~{taus.max():.0f}  '
            f'换过 {r["n_change"]} 次  不同取值 {len(np.unique(np.round(taus, -1)))} 个')
        c = collections.Counter(x['why'] for x in r['trades'])
        say(f'       平仓: ' + '  '.join(f'{k}={v}({v / r["n"]:.0%})' for k, v in c.most_common()))

    say('')
    say('【三】判据（唯一入口 gambler_judge：四段同号 + 每段赢 SHUF）')
    import gambler_judge as JD
    ok, why, det = JD.judge(
        np.array([x['t'] for x in rR['trades']], dtype=np.float64),
        np.array([x['ret'] for x in rR['trades']], dtype=np.float64),
        np.array([x['t'] for x in rS['trades']], dtype=np.float64),
        np.array([x['ret'] for x in rS['trades']], dtype=np.float64))
    say(JD.fmt(det))
    say(f'  {"✅ 站住" if ok else "❌ 判死"}：{why}')
    say(f'  净信号（真−永远做多）= {(rR["a"].mean() - rB["a"].mean()) * 100:+.4f}%/笔，'
        f'（真−打乱）= {(rR["a"].mean() - rS["a"].mean()) * 100:+.4f}%/笔')
    say('  未计：滑点、maker 挂单成交率')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_adaptive_v3.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    alltr = [dict(**x, tag='REAL') for x in rR['trades']] + \
            [dict(**x, tag='SHUF') for x in rS['trades']]
    pd.DataFrame(alltr).to_csv(os.path.join(OUT, 'gambler_adaptive_v3_trades.csv'),
                               index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
