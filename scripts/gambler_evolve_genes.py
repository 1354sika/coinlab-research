# -*- coding: utf-8 -*-
"""
gambler_evolve_genes.py · 四件事补齐版（结构池 / 反事实账本 / 在线选择 / 变异）

对应用户清单 2026-10-18：
  ① 结构池：基因 = (入场判据 × 退出结构 × 仓位规则)，共 3×5×3 = 45 个基因
  ② 反事实账本：每笔入场，对「当前权重前 8 名 + 1 个探索基因」全部模拟并记账
  ③ 在线选择：每块（24h）用已结算记录按表现分配权重（Hedge，eta 按信噪比自动定）
  ④ 变异：每 10 块把最差基因替换为最优基因的变异体；新基因表现好则权重上升

无前视：
  · 特征扩张标准化；成交 t+1 开盘；账本只用结算时间 < 块起点的记录
用法：python -X utf8 gambler_evolve_genes.py --symbols 2
输出：evidence/gambler_evolve_genes.txt / _genes.csv
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_memory import build_features, shuffle_prices, BLOCK, FEE

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 12
DEC = 12
TRAIN_FRAC = 0.70
MIN_HIST = 12
N_ACTIVE = 8

ENTRIES = ['E_MU', 'E_MOM', 'E_VOL']
EXITS = ['FIX', 'TRAIL2', 'TRAIL3', 'TIME', 'REV']
SIZES = ['S_FIX', 'S_KELLY', 'S_INV']
GENES = [(e, x, s) for e in ENTRIES for x in EXITS for s in SIZES]
NG = len(GENES)

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)


def exit_ret(style, side, entry, t0, end, o, h, l, c, atr, X, tp_pct, sl_pct):
    if style == 'FIX':
        tp = entry * (1 + side * tp_pct); sl = entry * (1 + side * sl_pct)
        for j in range(t0 + 1, end):
            if side == 1:
                if l[j] <= sl: return sl / entry - 1 - FEE
                if h[j] >= tp: return tp / entry - 1 - FEE
            else:
                if h[j] >= sl: return 1 - sl / entry - FEE
                if l[j] <= tp: return 1 - tp / entry - FEE
        return side * (c[end - 1] / entry - 1) - FEE
    if style in ('TRAIL2', 'TRAIL3'):
        m = 2.0 if style == 'TRAIL2' else 3.0
        sl = entry * (1 + side * sl_pct); best = entry
        for j in range(t0 + 1, end):
            a = atr[j] if np.isfinite(atr[j]) else 0.0
            if side == 1:
                if l[j] <= sl: return sl / entry - 1 - FEE
                best = max(best, h[j])
                if a > 0 and best - m * a > entry and l[j] <= best - m * a:
                    return (best - m * a) / entry - 1 - FEE
            else:
                if h[j] >= sl: return 1 - sl / entry - FEE
                best = min(best, l[j])
                if a > 0 and best + m * a < entry and h[j] >= best + m * a:
                    return 1 - (best + m * a) / entry - FEE
        return side * (c[end - 1] / entry - 1) - FEE
    if style == 'TIME':
        return side * (c[end - 1] / entry - 1) - FEE
    # REV
    for j in range(t0 + 1, end, DEC):
        if (X[j][0] + X[j][1] + X[j][2]) * side < 0:
            return side * (c[j] / entry - 1) - FEE
    return side * (c[end - 1] / entry - 1) - FEE


def entry_signal(kind, t, X, mu, volr_z):
    if kind == 'E_MU':
        return (1 if mu > 0 else -1) if abs(mu) > FEE else 0
    if kind == 'E_MOM':
        s = X[t][0] + X[t][1] + X[t][2]
        return (1 if s > 0.3 else -1) if abs(s) > 0.3 else 0
    s = X[t][0] + X[t][1] + X[t][2]
    return (1 if s > 0 else -1) if volr_z[t] > 0.5 else 0


def size_frac(kind, edge, sd, atr_pct):
    if kind == 'S_FIX':
        return 0.10
    if kind == 'S_KELLY':
        return float(np.clip(0.5 * edge / sd, 0.02, 0.5))
    return float(np.clip(0.02 / max(atr_pct, 1e-3), 0.02, 0.5))


def run_symbol(g, rng):
    X, o, h, l, c = build_features(g)
    n = len(c)
    if n < 8000:
        return None
    tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.abs(c[1:] - c[:-1]))
    cs = np.cumsum(tr); atr = np.full(n, np.nan); atr[14:] = (cs[14:] - cs[:-14]) / 14
    volr_z = X[:, -4]                      # 量比 z（build_features 中第 9 列附近，取保守）
    fwd = np.full(n, np.nan); fwd[:n - H] = c[H:] / np.maximum(c[:n - H], 1e-12) - 1
    cut = int(n * TRAIN_FRAC)

    wts = np.ones(NG) / NG
    genes = list(GENES)                       # 可变基因池（变异真正生效）
    ledger = {k: [] for k in range(NG)}       # gene -> [(exit_idx, ret, pos)]
    trades, glog = [], []
    for b0 in range(BLOCK, n - BLOCK, BLOCK):
        b1 = min(b0 + BLOCK, n)
        nb = b0 // BLOCK

        # ③ 在线选择：只用已结算记录
        score = np.full(NG, -1e9)
        for k in range(NG):
            done = [r for (ex, r, p) in ledger[k] if ex < b0]
            if len(done) >= MIN_HIST:
                a = np.array(done[-500:])
                score[k] = a.mean() - 0.5 * a.std() / np.sqrt(len(a))
        if np.all(score < -1e8):
            active = list(range(min(N_ACTIVE, NG)))
        else:
            active = list(np.argsort(score)[-N_ACTIVE:])
        explore = nb % NG
        if explore not in active:
            active = active + [explore]
        ms = np.array([score[k] if score[k] > -1e8 else 0.0 for k in active])
        eta = 1.0 / (np.std(ms) + 1e-9)
        ww = np.exp(np.clip(eta * ms, -20, 20)); ww /= ww.sum()
        wts = np.zeros(NG); wts[active] = ww

        # ② 记忆库（只放已结算案例）
        lo_i = max(0, b0 - 30 * 288); hi_i = b0 - H
        if hi_i - lo_i < 500:
            continue
        idx = np.arange(lo_i, hi_i, max(1, (hi_i - lo_i) // 3000))
        Zm, Rm = X[idx], fwd[idx]
        ok = np.isfinite(Rm); Zm, Rm = Zm[ok], Rm[ok]
        if len(Zm) < 200:
            continue

        t = b0
        while t < b1 - 2:
            zt = X[t]
            d = np.sum((Zm - zt) ** 2, axis=1)
            kk = min(50, len(d))
            nn = np.argpartition(d, kk - 1)[:kk]
            mu = float(Rm[nn].mean()); sd_mu = float(Rm[nn].std()) + 1e-6
            entry = float(o[t + 1])
            if not np.isfinite(entry) or entry <= 0:
                t += DEC; continue
            end = min(t + 1 + H, b1)
            a0 = atr[t] if np.isfinite(atr[t]) and atr[t] > 0 else entry * 0.005
            tp_pct = max(2.0 * a0 / entry, FEE + 0.002)
            sl_pct = min(1.5 * a0 / entry, -(FEE + 0.002))
            atr_pct = a0 / entry

            # ② 反事实账本：对当前 active 基因逐个模拟
            for k in active:
                e_kind, x_kind, s_kind = genes[k]
                side = entry_signal(e_kind, t, X, mu, volr_z)
                if side == 0:
                    continue
                r = exit_ret(x_kind, side, entry, t, end, o, h, l, c, atr, X, tp_pct, sl_pct)
                pos = size_frac(s_kind, abs(mu) - FEE, sd_mu, atr_pct)
                ledger[k].append((int(end - 1), float(r), float(pos)))

            # 真实交易：权重最高的基因
            kbest = int(np.argmax(wts))
            e_kind, x_kind, s_kind = genes[kbest]
            side = entry_signal(e_kind, t, X, mu, volr_z)
            if side != 0:
                r = exit_ret(x_kind, side, entry, t, end, o, h, l, c, atr, X, tp_pct, sl_pct)
                pos = size_frac(s_kind, abs(mu) - FEE, sd_mu, atr_pct)
                trades.append(dict(t=int(t), exit=int(end - 1), ret=float(r), pos=float(pos),
                                   side=side, gene=str(genes[kbest]),
                                   seg=('tr' if b0 < cut else 'te')))
            t = end + 1

        # ④ 变异：每 10 块，把最差基因换成最优基因的变异体
        if nb % 10 == 0 and np.any(score > -1e8):
            worst = int(np.argmin(score))
            best = int(np.argmax(score))
            e_kind, x_kind, s_kind = genes[best]
            newg = (ENTRIES[rng.integers(0, len(ENTRIES))],
                    x_kind, SIZES[rng.integers(0, len(SIZES))])
            oldg = genes[worst]
            genes[worst] = newg                    # 真正替换基因
            ledger[worst] = []
            glog.append(dict(b=b0, kind='mutate', worst=str(oldg), new=str(newg)))
        glog.append(dict(b=b0, kind='w', top=str(genes[int(np.argmax(wts))]),
                         w=round(float(np.max(wts)), 3)))

    if not trades:
        return None
    return pd.DataFrame(trades), glog


def summarize(df, name):
    if df is None or not len(df):
        say(f'  {name}: 无交易'); return None
    eq = np.cumprod(1 + df.ret.values * df.pos.values)
    peak = np.maximum.accumulate(eq)
    out = dict(name=name, n=int(len(df)), mean=float(df.ret.mean()), med=float(np.median(df.ret)),
               win=float((df.ret > 0).mean()), tot=float(eq[-1] - 1),
               dd=float(((eq - peak) / peak).min()),
               t=float(df.ret.mean() / (df.ret.std(ddof=1) / np.sqrt(len(df)))) if len(df) > 2 else 0.0)
    say(f'  {name:<10} n={out["n"]:>6}  均值 {out["mean"]:+.4%}  中位 {out["med"]:+.4%}  '
        f'胜率 {out["win"]:.1%}  t={out["t"]:+.2f}  资金 {out["tot"]:+.1%}  回撤 {out["dd"]:+.1%}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=int, default=2)
    ap.add_argument('--no-shuffle', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    say('=' * 100)
    say('gambler_evolve_genes · 结构池 / 反事实账本 / 在线选择 / 变异')
    say(f'  基因 {NG} 个 = 入场{ENTRIES} × 退出{EXITS} × 仓位{SIZES}')
    say('=' * 100)
    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32',
          'low': 'float32', 'close': 'float32', 'volume': 'float32', 'taker_base': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:a.symbols]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')
    rng = np.random.default_rng(20261018)
    real, shuf, glogs = [], [], []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        r = run_symbol(g, rng)
        if r is not None:
            T, G = r; T['sym'] = sym; real.append(T); glogs.extend(G)
        if not a.no_shuffle:
            o, h, l, c = (g[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close'))
            o2, h2, l2, c2 = shuffle_prices(o, h, l, c, rng)
            g2 = g.copy(); g2['open'], g2['high'], g2['low'], g2['close'] = o2, h2, l2, c2
            r2 = run_symbol(g2, rng)
            if r2 is not None:
                T2, _ = r2; T2['sym'] = sym; shuf.append(T2)
        say(f'    … {si+1}/{len(syms)} 币  {time.time()-t0:.0f}s')

    T = pd.concat(real, ignore_index=True) if real else None
    S = pd.concat(shuf, ignore_index=True) if shuf else None
    say('\n' + '=' * 100)
    say('【一】真实序列（交易由当前权重最高的基因执行）')
    r_tr = summarize(T[T.seg == 'tr'], '训练段')
    r_te = summarize(T[T.seg == 'te'], '样本外')
    say('\n【二】打乱对照')
    s_te = summarize(S[S.seg == 'te'], '对照样本外') if S is not None else None
    say('\n【三】哪些基因被选为最优（样本外交易用的基因分布）')
    if T is not None and len(T[T.seg == 'te']):
        vc = T[T.seg == 'te'].gene.value_counts(normalize=True)
        say('  ' + '  '.join(f'{k}:{v:.0%}' for k, v in vc.head(6).items()))
    say('\n【四】变异/权重日志')
    n_mut = int(sum(1 for x in glogs if x['kind'] == 'mutate'))
    say(f'  发生变异 {n_mut} 次（每 10 块一次，共 {len(glogs)//2} 个更新点）')
    if glogs:
        say(f'  最后一次活跃基因: {glogs[-1].get("top", "-")} 权重 {glogs[-1].get("w", "-")}')
    say('\n【五】结论判据')
    if r_te and s_te:
        say(f'  样本外 {r_te["mean"]:+.4%}（对照 {s_te["mean"]:+.4%}）')
        say('  ✅ 通过' if (r_te['mean'] > 0 and s_te['mean'] <= 0) else '  ❌ 未通过（结构/方向仍需改进）')
    say(f'\nTOTAL {time.time()-t0:.0f}s')
    with open(os.path.join(OUT, 'gambler_evolve_genes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    if T is not None:
        T.to_csv(os.path.join(OUT, 'gambler_evolve_genes_trades.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
