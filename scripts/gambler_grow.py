# -*- coding: utf-8 -*-
"""
gambler_grow.py · 让退出结构「自己长出来」（策略池 + 反事实账本 + 在线加权）

用户口径（2026-10-18）：
  "补完跑是基础，后面如何调整优化、如何让模型自己长出来才是重要的"
  → 不再由人预设一个退出结构，而是给模型一个结构池，让它按「已结算的表现」自己分配权重

结构池（不是参数池，是形状不同的四种退出）：
  FIX   固定止盈止损（q90 / q10）
  TRAIL 跟踪止盈（回撤 m×ATR 才走）+ q10 止损
  TIME  时间退出（持有 H 根，不设价）
  REV   反向信号退出（期望转负就走）
另外留一个「无」对照（只持 H 根），用于看其它结构是否真比裸持有强。

机制：
  ① 每个决策点，对全部结构同时模拟（反事实账本）——这是"自己长出来"的前提：
     模型必须知道"没走的那条路"结果如何，才有依据选。
  ② 每块（24h）用已结算的结构表现更新权重：w ∝ exp(eta * 均值)，eta 按信噪比自动定。
  ③ 组合收益 = Σ w_k * ret_k（资金按权重分给各结构，等价于"当下同时采用几种打法"）。

无前视：
  特征用扩张窗口；权重只用「结算时间 < 块起点」的结构记录；成交在 t+1 开盘。

用法：python -X utf8 gambler_grow.py --symbols 4
输出：evidence/gambler_grow.txt / .json / _trades.csv
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_memory import build_features, shuffle_prices, BLOCK, DEC, FEE

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 12                      # 结构统一持有上限（1h），由 TIME 结构检验
TRAIL_MULTS = [2.0, 3.0]
STYLES = ['FIX', 'TRAIL2', 'TRAIL3', 'TIME', 'REV']
TRAIN_FRAC = 0.70
MIN_HIST = 30
CAP = 0.5

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)


def sim_exit(style, side, entry, t0, end, o, h, l, c, atr, tp_pct, sl_pct, X):
    """在 [t0+1, end) 内按结构模拟一笔，返回净收益（已扣费）"""
    if style == 'FIX':
        tp = entry * (1 + side * tp_pct)
        sl = entry * (1 + side * sl_pct)
        for j in range(t0 + 1, end):
            if side == 1:
                if l[j] <= sl: return sl / entry - 1 - FEE
                if h[j] >= tp: return tp / entry - 1 - FEE
            else:
                if h[j] >= sl: return 1 - sl / entry - FEE
                if l[j] <= tp: return 1 - tp / entry - FEE
        return side * (c[end - 1] / entry - 1) - FEE
    if style.startswith('TRAIL'):
        m = 2.0 if style == 'TRAIL2' else 3.0
        sl = entry * (1 + side * sl_pct)
        best = entry
        for j in range(t0 + 1, end):
            a = atr[j] if np.isfinite(atr[j]) else 0.0
            if side == 1:
                if l[j] <= sl: return sl / entry - 1 - FEE
                best = max(best, h[j])
                if a > 0 and best - m * a > entry and l[j] <= best - m * a:
                    px = best - m * a
                    return px / entry - 1 - FEE
            else:
                if h[j] >= sl: return 1 - sl / entry - FEE
                best = min(best, l[j])
                if a > 0 and best + m * a < entry and h[j] >= best + m * a:
                    px = best + m * a
                    return 1 - px / entry - FEE
        return side * (c[end - 1] / entry - 1) - FEE
    if style == 'TIME':
        return side * (c[end - 1] / entry - 1) - FEE
    if style == 'REV':
        for j in range(t0 + 1, end, DEC):
            if (X[j][0] + X[j][1] + X[j][2]) * side < 0:      # 反向信号出现就走
                return side * (c[j] / entry - 1) - FEE
        return side * (c[end - 1] / entry - 1) - FEE
    return side * (c[end - 1] / entry - 1) - FEE


def run_symbol(g, rng, tag='REAL'):
    X, o, h, l, c = build_features(g)
    n = len(c)
    if n < 8000:
        return None
    a_tr = np.zeros(n); a_tr[1:] = np.maximum(h[1:] - l[1:], np.abs(c[1:] - c[:-1]))
    cs = np.cumsum(a_tr); atr = np.full(n, np.nan); atr[14:] = (cs[14:] - cs[:-14]) / 14
    cut = int(n * TRAIN_FRAC)
    fwd = np.full(n, np.nan)
    fwd[:n - H] = c[H:] / np.maximum(c[:n - H], 1e-12) - 1
    K = len(STYLES)
    wts = np.ones(K) / K
    ledger = {s: [] for s in STYLES}       # 每结构：[(exit_idx, ret)]
    rows = []
    for b0 in range(BLOCK, n - BLOCK, BLOCK):
        b1 = min(b0 + BLOCK, n)
        # ① 只用已结算记录更新权重（结算时间 < b0）
        for k, s in enumerate(STYLES):
            done = [r for (ex, r) in ledger[s] if ex < b0]
            if len(done) >= MIN_HIST:
                m = np.array(done[-2000:]).mean()
                sd = np.array(done[-2000:]).std() + 1e-9
                rows.append(dict(kind='w', s=s, m=float(m)))
        ms, ns = [], []
        for s in STYLES:
            done = [r for (ex, r) in ledger[s] if ex < b0]
            if len(done) >= MIN_HIST:
                a = np.array(done[-2000:])
                ms.append(a.mean()); ns.append(a.std() + 1e-9)
            else:
                ms.append(0.0); ns.append(1.0)
        ms = np.array(ms); ns = np.array(ns)
        eta = 1.0 / (np.std(ms) + 1e-9)          # 信噪比定步长，无固定 eta
        wts = np.exp(np.clip(eta * ms, -20, 20)); wts /= wts.sum()

        # ② 建记忆库（只放结算时间 <= b0 的案例，严格无前视）
        lo_i = max(0, b0 - 30 * 288)
        hi_i = b0 - H
        if hi_i - lo_i < 500:
            continue
        idx = np.arange(lo_i, hi_i, max(1, (hi_i - lo_i) // 3000))
        Zm, Rm = X[idx], fwd[idx]
        ok = np.isfinite(Rm)
        Zm, Rm = Zm[ok], Rm[ok]
        if len(Zm) < 200:
            continue
        # ③ 块内：每个决策点对所有结构做反事实模拟（方向由牌型记忆给，不由人给）
        t = b0
        while t < b1 - 2:
            zt = X[t]
            d = np.sum((Zm - zt) ** 2, axis=1)
            kk = min(50, len(d))
            nn = np.argpartition(d, kk - 1)[:kk]
            mu = float(Rm[nn].mean())
            if abs(mu) <= FEE:
                t += DEC
                continue
            side = 1 if mu > 0 else -1
            entry = float(o[t + 1])
            if not np.isfinite(entry) or entry <= 0:
                t += DEC; continue
            end = min(t + 1 + H, b1)
            # tp/sl 由最近 288 根的波动尺度定（当场算，不用未来）
            a0 = atr[t] if np.isfinite(atr[t]) and atr[t] > 0 else entry * 0.005
            tp_pct = max(2.0 * a0 / entry, FEE + 0.002)
            sl_pct = min(1.5 * a0 / entry, -(FEE + 0.002))
            rets = []
            for s in STYLES:
                r = sim_exit(s, side, entry, t, end, o, h, l, c, atr, tp_pct, sl_pct, X)
                ledger[s].append((int(end - 1), float(r)))
                rets.append(r)
            rets = np.array(rets)
            combo = float(wts @ rets)
            rows.append(dict(kind='t', b=b0, t=int(t), exit=int(end - 1), side=side,
                             combo=combo, rets=[round(float(x), 6) for x in rets],
                             w=[round(float(x), 3) for x in wts],
                             seg=('tr' if b0 < cut else 'te')))
            t = end + 1

    if not rows:
        return None
    T = pd.DataFrame([r for r in rows if r['kind'] == 't'])
    if not len(T):
        return None
    return T, [r for r in rows if r['kind'] == 'w']


def summarize(T, name):
    if T is None or not len(T):
        say(f'  {name}: 无交易'); return None
    r = T.combo.values
    eq = np.cumprod(1 + r * CAP); peak = np.maximum.accumulate(eq)
    out = dict(name=name, n=int(len(r)), mean=float(r.mean()), med=float(np.median(r)),
               win=float((r > 0).mean()), tot=float(eq[-1] - 1), dd=float(((eq - peak) / peak).min()),
               t=float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else 0.0)
    say(f'  {name:<10} n={out["n"]:>6}  均值 {out["mean"]:+.4%}  中位 {out["med"]:+.4%}  '
        f'胜率 {out["win"]:.1%}  t={out["t"]:+.2f}  资金 {out["tot"]:+.1%}  回撤 {out["dd"]:+.1%}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=int, default=4)
    ap.add_argument('--no-shuffle', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    say('=' * 100)
    say('gambler_grow · 让模型自己长出退出结构（策略池 + 反事实账本 + 在线加权）')
    say(f'  结构池 {STYLES}  持有上限 {H} 根  费 {FEE:.4%}')
    say('=' * 100)
    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32',
          'low': 'float32', 'close': 'float32', 'volume': 'float32', 'taker_base': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:a.symbols]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')
    rng = np.random.default_rng(20261018)
    real, shuf, wlogs = [], [], []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        r = run_symbol(g, rng)
        if r is not None:
            T, W = r; T['sym'] = sym; real.append(T); wlogs.extend(W)
        if not a.no_shuffle:
            o, h, l, c = (g[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close'))
            o2, h2, l2, c2 = shuffle_prices(o, h, l, c, rng)
            g2 = g.copy(); g2['open'], g2['high'], g2['low'], g2['close'] = o2, h2, l2, c2
            r2 = run_symbol(g2, rng)
            if r2 is not None:
                T2, _ = r2; T2['sym'] = sym; shuf.append(T2)
        if (si + 1) % 5 == 0:
            say(f'    … {si+1}/{len(syms)} 币  {time.time()-t0:.0f}s')

    T = pd.concat(real, ignore_index=True) if real else None
    S = pd.concat(shuf, ignore_index=True) if shuf else None
    say('\n' + '=' * 100)
    say('【一】组合（权重由模型自己定）')
    r_tr = summarize(T[T.seg == 'tr'], '训练段')
    r_te = summarize(T[T.seg == 'te'], '样本外')
    say('\n【二】各结构自己长成什么样（样本外单独表现）')
    if T is not None and len(T[T.seg == 'te']):
        Te = T[T.seg == 'te']
        for k, s in enumerate(STYLES):
            r = np.array([x[k] for x in Te.rets])
            say(f'  {s:<7} n={len(r):>5}  均值 {r.mean():+.4%}  中位 {np.median(r):+.4%}  胜率 {(r>0).mean():.1%}')
    say('\n【三】打乱对照（判断是不是在记噪声）')
    s_te = summarize(S[S.seg == 'te'], '对照样本外') if S is not None else None
    say('\n【四】结论判据')
    if r_te and s_te:
        if r_te['mean'] > 0 and s_te['mean'] <= 0:
            say('  ✅ 样本外 > 0 且打乱对照 <= 0 ⇒ 结构可迁移')
        else:
            say(f'  ❌ 样本外 {r_te["mean"]:+.4%}（对照 {s_te["mean"]:+.4%}）⇒ 尚未证明可交易')
    say(f'\nTOTAL {time.time()-t0:.0f}s')
    with open(os.path.join(OUT, 'gambler_grow.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    if T is not None:
        T.to_csv(os.path.join(OUT, 'gambler_grow_trades.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
