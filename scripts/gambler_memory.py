# -*- coding: utf-8 -*-
"""
gambler_memory.py · 牌型记忆赌徒 v1（记牌型 / 记忆窗口自选 / 严格无前视 / 三动作 / 仓位自算）

用户口径（2026-10-18）：
  "给模型也记牌型" —— 把历史切成状态向量 + 后续结果，遇到当下找最像的 K 个片段
  "模型自己决定往回看多久" —— 记忆窗口 MEM（天）与近邻数 K 在线选
  "实时决定每个变量当下变成什么样" —— 特征全部是 t 时刻可得的量，标准化用扩张窗口
  "实时决定买/卖/持" —— 由近邻收益分布的期望与成本比较决定
  "所有参数不固定" —— MEM/K/H 由过去已结算的交易在线选（bandit，含探索）
  "杜绝前视" —— ① 记忆库只放结算早于块起点的案例 ② 成交在 t+1 开盘 ③ 参数只用已平仓交易

对照：REAL（真序列） vs SHUFFLE（5m 收益按 288 根块打乱，重构价格）
  SHUFFLE 也赚 ⇒ 记的是噪声；只 REAL 赚 ⇒ 有可迁移的牌型

用法：python -X utf8 gambler_memory.py --symbols 60
输出：evidence/gambler_memory.txt / .json / _trades.csv
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'

SCALES = [1, 3, 6, 12, 24, 48, 96, 288]
H_GRID = [6, 12, 24]                 # 持有上限（5m 根）：30m / 1h / 2h
K_GRID = [20, 50, 100]               # 近邻数（多像才算像）
MEM_GRID = [7, 30, 90]               # 记忆窗口：只看最近多少天
BLOCK = 288                          # 每 24h 在线更新一次参数
DEC = 6                              # 每 30m 做一次决策
MAX_CASE = 4000                      # 记忆库上限（等距采样）
FEE = 0.0004                         # 往返手续费（maker 0.02%×2）
MIN_TRADES = 20
TRAIN_FRAC = 0.70
CAP_KELLY = 0.5                      # 半个凯利

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)


def exp_z(x):
    n = len(x)
    cs = np.nancumsum(x); cs2 = np.nancumsum(x * x)
    cnt = np.arange(1, n + 1, dtype=np.float64)
    mu = cs / cnt
    sd = np.sqrt(np.maximum(cs2 / cnt - mu * mu, 0))
    z = np.zeros(n); ok = sd > 1e-12
    z[ok] = (x[ok] - mu[ok]) / sd[ok]
    return z


def rmean(x, k):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[k - 1:] = (cs[k:] - cs[:-k]) / k
    return out


def build_features(df_g):
    o = df_g.open.values.astype(np.float64)
    h = df_g.high.values.astype(np.float64)
    l = df_g.low.values.astype(np.float64)
    c = df_g.close.values.astype(np.float64)
    v = df_g.volume.values.astype(np.float64)
    tb = df_g.taker_base.values.astype(np.float64)
    n = len(c)
    cols = []
    for k in SCALES:                                   # 多尺度动量（z 化）
        mom = np.zeros(n); mom[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1
        cols.append(exp_z(mom))
    r1 = np.zeros(n); r1[1:] = c[1:] / np.maximum(c[:-1], 1e-12) - 1
    cols.append(exp_z(np.nan_to_num(rmean(r1 * r1, 60) - rmean(r1, 60) ** 2)))   # 波动
    cols.append(exp_z(np.nan_to_num(v / np.maximum(rmean(v, 60), 1e-12) - 1)))   # 量比
    cols.append(exp_z(np.nan_to_num(tb / np.maximum(v, 1e-9) - 0.5)))            # taker 失衡
    hi = pd.Series(h).rolling(288, min_periods=10).max().values
    lo = pd.Series(l).rolling(288, min_periods=10).min().values
    pos = np.nan_to_num((c - lo) / np.maximum(hi - lo, 1e-12) - 0.5)
    cols.append(exp_z(pos))                            # 24h 区间位置
    X = np.stack(cols, axis=1)
    X = np.clip(np.nan_to_num(X), -6, 6)
    return X, o, h, l, c


def shuffle_prices(o, h, l, c, rng, block=BLOCK):
    n = len(c)
    r = np.zeros(n); r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    out = np.empty(n); pos = 0
    for b in rng.permutation(len(blocks)):
        seg = blocks[b]; out[pos:pos + len(seg)] = r[seg]; pos += len(seg)
    newc = np.exp(np.cumsum(out)) * c[0]
    ratio = newc / np.maximum(c, 1e-12)
    return o * ratio, h * ratio, l * ratio, newc


def run_symbol(df_g, rng, tag='REAL'):
    X, o, h, l, c = build_features(df_g)
    n = len(c)
    if n < 8000:
        return None
    fwd = {H: np.concatenate([c[H:] / np.maximum(c[:-H], 1e-12) - 1, np.full(H, np.nan)]) for H in H_GRID}
    cut = int(n * TRAIN_FRAC)
    trades, logrows = [], []
    K_sel, MEM_sel, H_sel = 50, 30, 12

    for b0 in range(BLOCK, n - BLOCK, BLOCK):
        b1 = min(b0 + BLOCK, n)
        nb = b0 // BLOCK
        explore = (nb % 4 == 0)

        # ---- ① 在线选参：只用已平仓交易 ----
        done = [t for t in trades if t['exit'] < b0]
        best = None
        if len(done) >= MIN_TRADES:
            for K in K_GRID:
                for M in MEM_GRID:
                    for H in H_GRID:
                        sub = [t for t in done if t['K'] == K and t['M'] == M and t['H'] == H]
                        if len(sub) < MIN_TRADES:
                            continue
                        a = np.array([t['ret'] for t in sub])
                        lo = a.mean() - 0.5 * a.std(ddof=1) / np.sqrt(len(a))    # 保守下界
                        if best is None or lo > best[0]:
                            best = (lo, K, M, H)
        GRID = [(K, M, H) for K in K_GRID for M in MEM_GRID for H in H_GRID]
        if explore:
            K_sel, MEM_sel, H_sel = GRID[nb % len(GRID)]
        elif best is not None:
            _, K_sel, MEM_sel, H_sel = best

        # ---- ② 建记忆库（只放结算时间 <= b0 的案例，严格无前视）----
        lo_i = max(0, b0 - MEM_sel * 288)
        hi_i = b0 - H_sel
        if hi_i - lo_i < 500:
            continue
        idx = np.arange(lo_i, hi_i, max(1, (hi_i - lo_i) // MAX_CASE))
        Zm = X[idx]
        Rm = fwd[H_sel][idx]
        ok = np.isfinite(Rm)
        Zm, Rm = Zm[ok], Rm[ok]
        if len(Zm) < 200:
            continue

        # ---- ③ 块内决策：买 / 卖 / 持（决策时点由「状态漂移」触发，不是固定时钟）----
        drift_thr = 0.5
        if b0 > 2000:
            dx = np.linalg.norm(np.diff(X[max(0, b0 - 30 * 288):b0], axis=0), axis=1)
            if len(dx) > 100:
                drift_thr = float(np.quantile(dx, 0.60))   # 状态变得比历史 60% 的时刻快，才重新决策
        t = b0
        last_z, last_t = None, b0
        while t < b1 - 2:
            if last_z is not None:
                drift = float(np.linalg.norm(X[t] - last_z))
                if drift < drift_thr and (t - last_t) < 24:
                    t += 1
                    continue
            zt = X[t]
            last_z, last_t = zt, t
            d = np.sum((Zm - zt) ** 2, axis=1)
            kk = min(K_sel, len(d))
            nn = np.argpartition(d, kk - 1)[:kk]
            rn = Rm[nn]
            mu = float(rn.mean())
            sd = float(rn.std(ddof=1)) + 1e-6
            e_long = mu - FEE
            e_short = -mu - FEE
            if e_long > 0 and e_long >= e_short:
                side, edge = 1, e_long
            elif e_short > 0:
                side, edge = -1, e_short
            else:
                t += DEC
                continue                       # 持：这一类牌型扣费后没有正期望
            pos = float(np.clip(CAP_KELLY * edge / sd, 0.02, CAP_KELLY))
            entry = float(o[t + 1])
            if not np.isfinite(entry) or entry <= 0:
                t += DEC; continue
            # 止盈/止损由「这一类牌型」的历史收益分布当场给出，不写死
            q90, q10 = np.quantile(rn, 0.90), np.quantile(rn, 0.10)
            gain_q, loss_q = (q90, q10) if side == 1 else (-q10, -q90)
            gain_q = max(gain_q, FEE + 0.002)
            loss_q = min(loss_q, -(FEE + 0.002))
            tp_px = entry * (1 + side * gain_q)
            sl_px = entry * (1 + side * loss_q)
            end = min(t + 1 + H_sel, b1)
            ret = None
            j = t + 1
            while j < end:
                if side == 1:
                    if l[j] <= sl_px:
                        ret = sl_px / entry - 1 - FEE; break
                    if h[j] >= tp_px:
                        ret = tp_px / entry - 1 - FEE; break
                else:
                    if h[j] >= sl_px:
                        ret = 1 - sl_px / entry - FEE; break
                    if l[j] <= tp_px:
                        ret = 1 - tp_px / entry - FEE; break
                j += 1
            if ret is None:
                ret = side * (c[end - 1] / entry - 1) - FEE
                j = end - 1
            trades.append(dict(t=int(t), exit=int(j), ret=float(ret), pos=pos, edge=float(edge),
                               K=K_sel, M=MEM_sel, H=H_sel, side=side,
                               seg=('tr' if b0 < cut else 'te')))
            t = j + 1

        logrows.append(dict(b=b0, K=K_sel, M=MEM_sel, H=H_sel))

    if not trades:
        return None
    return pd.DataFrame(trades), logrows


def summarize(df, name):
    if df is None or len(df) == 0:
        say(f'  {name:<12} 无交易'); return None
    r = df.ret.values
    eq = np.cumprod(1 + r * df.pos.values)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    out = dict(name=name, n=int(len(r)), mean=float(r.mean()), med=float(np.median(r)),
               win=float((r > 0).mean()), total=float(eq[-1] - 1), dd=dd,
               t=float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else 0.0)
    say(f'  {name:<12} n={out["n"]:>6}  均值 {out["mean"]:+.4%}  中位 {out["med"]:+.4%}  '
        f'胜率 {out["win"]:.1%}  t={out["t"]:+.2f}  资金 {out["total"]:+.1%}  回撤 {dd:+.1%}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=int, default=60)
    ap.add_argument('--no-shuffle', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    say('=' * 104)
    say('gambler_memory · 牌型记忆赌徒（记牌型 / 记忆窗口在线自选 / 三动作 / 仓位按优势）')
    say(f'  牌面 {len(SCALES)+5} 维 | K {K_GRID} | 记忆窗口 {MEM_GRID}天 | 持有 {H_GRID}根 | 费 {FEE:.4%}')
    say('=' * 104)

    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32', 'low': 'float32',
          'close': 'float32', 'volume': 'float32', 'taker_base': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:a.symbols]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

    rng = np.random.default_rng(20261018)
    real, shuf, logs = [], [], []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        r = run_symbol(g, rng)
        if r is not None:
            tr, lg = r; tr['sym'] = sym; real.append(tr); logs.extend(lg)
        if not a.no_shuffle:
            o, h, l, c = (g[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close'))
            o2, h2, l2, c2 = shuffle_prices(o, h, l, c, rng)
            g2 = g.copy()
            g2['open'], g2['high'], g2['low'], g2['close'] = o2, h2, l2, c2
            r2 = run_symbol(g2, rng)
            if r2 is not None:
                tr2, _ = r2; tr2['sym'] = sym; shuf.append(tr2)
        if (si + 1) % 10 == 0:
            say(f'    … {si+1}/{len(syms)} 币  {time.time()-t0:.0f}s')

    T = pd.concat(real, ignore_index=True) if real else None
    S = pd.concat(shuf, ignore_index=True) if shuf else None

    say('\n' + '=' * 104)
    say('【一】真实序列')
    r_tr = summarize(T[T.seg == 'tr'], '训练段')
    r_te = summarize(T[T.seg == 'te'], '样本外')
    say('\n【二】打乱对照（结构破坏，只用来判断"是不是在记噪声"）')
    s_te = summarize(S[S.seg == 'te'], '对照样本外') if S is not None else None

    say('\n【三】模型自己选了什么（说明参数确实在变，不是扫出来的）')
    L = pd.DataFrame(logs)
    if len(L):
        say('  记忆窗口分布： ' + '  '.join(f'{k}天={v:.0%}' for k, v in L.M.value_counts(normalize=True).items()))
        say('  近邻数分布  ： ' + '  '.join(f'K={k}:{v:.0%}' for k, v in L.K.value_counts(normalize=True).items()))
        say('  持有期分布  ： ' + '  '.join(f'H={k}根:{v:.0%}' for k, v in L.H.value_counts(normalize=True).items()))
        say(f'  记忆窗口换过 {int((L.M.diff().fillna(0) != 0).sum())} 次 / 共 {len(L)} 个更新点')
        say(f'  弃牌率（无正期望不下注）： {1 - len(T) / max(1, len(L) * (BLOCK // 6)):.1%}')

    say('\n【四】分币样本外')
    if T is not None and len(T[T.seg == 'te']):
        g = T[T.seg == 'te'].groupby('sym').ret.agg(['count', 'mean'])
        g = g[g['count'] >= 10].sort_values('mean', ascending=False)
        if len(g):
            say(f'  有交易的币 {len(g)}；样本外均值>0 的币 {int((g["mean"]>0).sum())}/{len(g)}')
            say('  最好 3: ' + '  '.join(f'{i}:{v["mean"]:+.3%}' for i, v in g.head(3).iterrows()))
            say('  最差 3: ' + '  '.join(f'{i}:{v["mean"]:+.3%}' for i, v in g.tail(3).iterrows()))

    say('\n【五】结论判据')
    if r_te and s_te:
        if r_te['mean'] > 0 and s_te['mean'] <= 0:
            say('  ✅ 真序列样本外 > 0 且打乱对照 <= 0 ⇒ 存在可迁移结构（不是记噪声）')
        elif r_te['mean'] > 0 and s_te['mean'] > 0:
            say('  ⚠ 两者都 > 0 ⇒ 可能只是吃到了波动/成本结构，不足以证明有牌型记忆')
        else:
            say('  ❌ 真序列样本外 <= 0 ⇒ 这套牌型记忆在扣费后没有可交易优势（未证明不存在，只证明这一版不行）')
    say(f'\nTOTAL {time.time()-t0:.0f}s')

    res = dict(params=dict(scales=SCALES, K=K_GRID, MEM=MEM_GRID, H=H_GRID, fee=FEE,
                           block=BLOCK, dec=DEC, train_frac=TRAIN_FRAC, kelly=CAP_KELLY),
               symbols=[str(s) for s in syms], real_tr=r_tr, real_te=r_te, shuffle_te=s_te)
    with open(os.path.join(OUT, 'gambler_memory.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'gambler_memory.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    if T is not None:
        T.to_csv(os.path.join(OUT, 'gambler_memory_trades.csv'), index=False, encoding='utf-8-sig')
    if S is not None:
        S.to_csv(os.path.join(OUT, 'gambler_memory_shuffle_trades.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
