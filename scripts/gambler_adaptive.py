# -*- coding: utf-8 -*-
"""
gambler_adaptive.py · 自适应赌徒 v1（严格无前视 / 回看窗口自选 / 参数在线自选）

设计目标（对应用户口径 2026-10-18）：
  1. 杜绝前视：t 时刻决策只用 t 及以前的数据；成交在 t+1 开盘；参数只用「已平仓」的交易更新
  2. 回看窗口不写死：8 个尺度专家（5m~24h）在线竞价，权重由各自已实现净收益决定
  3. 每个变量当下变：S（组合分）→ 阈值分位 q 由历史选；持有期 H、止损倍数 sl 由历史选
  4. 实时决定买/卖/持：S>+thr 买、S<-thr 卖、否则持（平）
  5. 无固定业务参数：q/H/sl 从网格中按过去表现在线选；eta 按信噪比自动定

对照实验（回答"记住答案是否正确"）：
  REAL    : 真实序列
  SHUFFLE : 把 5m 收益按 288 根块随机打乱重建价格（结构破坏，特征-收益错配）
  若 SHUFFLE 也赚 ⇒ 模型在记噪声；若只 REAL 赚 ⇒ 有可迁移结构；若仅训练段赚 ⇒ 记住答案

用法：python -X utf8 gambler_adaptive.py [--symbols 60] [--quick]
输出：evidence/gambler_adaptive.txt / .json
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'

SCALES = [1, 3, 6, 12, 24, 48, 96, 288]      # 5m 根数 → 5m / 15m / 30m / 1h / 2h / 4h / 8h / 24h
H_GRID = [6, 24, 96]                          # 持有上限：30m / 2h / 8h
SL_GRID = [0.0, 2.5]                          # 止损：无 / 2.5×ATR
Q_GRID = [0.70, 0.80, 0.90, 0.95]             # 门槛分位
BLOCK = 288                                   # 每 24h 更新一次参数（在线，不用未来）
LOOKBACK_BLOCKS = 30                          # 专家权重看的过去块数（30 天）
FEE = 0.0004                                  # 往返手续费（maker 0.02%×2）
MIN_TRADES = 20                               # 参数选择的最小已平仓样本
TRAIN_FRAC = 0.70


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)


def load_symbol(sym, df_all):
    g = df_all[df_all.symbol == sym]
    o = g.open.values.astype(np.float64)
    h = g.high.values.astype(np.float64)
    l = g.low.values.astype(np.float64)
    c = g.close.values.astype(np.float64)
    return o, h, l, c


def exp_z(x):
    """扩张窗口 z 分（严格：第 i 点只用 [0,i] 的统计量；i 之前无数据则 z=0）"""
    n = len(x)
    cs = np.nancumsum(x)
    cs2 = np.nancumsum(x * x)
    cnt = np.arange(1, n + 1, dtype=np.float64)
    mu = cs / cnt
    var = np.maximum(cs2 / cnt - mu * mu, 0)
    sd = np.sqrt(var)
    z = np.zeros(n)
    ok = sd > 1e-12
    z[ok] = (x[ok] - mu[ok]) / sd[ok]
    return z


def build_features(o, h, l, c):
    n = len(c)
    K = len(SCALES)
    S = np.zeros((n, K))
    for j, k in enumerate(SCALES):
        mom = np.zeros(n)
        if n > k:
            mom[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1
        S[:, j] = np.tanh(exp_z(mom) / 2.0)          # 每尺度专家的方向信号 ∈ (-1,1)
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:] - l[1:], np.abs(c[1:] - c[:-1]))
    cs = np.cumsum(tr)
    k = 14
    atr[k:] = (cs[k:] - cs[:-k]) / k
    for i in range(1, min(k, n)):
        atr[i] = cs[i] / (i + 1)
    return S, atr


def expert_realized(S, c, H):
    """每个专家若按自己的方向持有 H 根，到 t+H 才知道的净收益（只用于 t+H 之后的参数更新）"""
    n = len(c)
    fwd = np.full(n, np.nan)
    if n > H:
        fwd[:n - H] = c[H:] / np.maximum(c[:n - H], 1e-12) - 1
    return np.sign(S) * fwd[:, None] - FEE


def shuffle_price(o, h, l, c, rng, block=BLOCK):
    """块间打乱收益后重建价格（结构破坏对照）：块内保留，块序随机"""
    n = len(c)
    r = np.zeros(n)
    r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    out = np.empty(n)
    pos = 0
    for b in rng.permutation(len(blocks)):      # 块序随机，块内保留
        seg = blocks[b]
        out[pos:pos + len(seg)] = r[seg]
        pos += len(seg)
    newc = np.exp(np.cumsum(out)) * c[0]
    ratio = newc / np.maximum(c, 1e-12)
    return o * ratio, h * ratio, l * ratio, newc


def run_symbol(o, h, l, c, rng, tag='REAL'):
    n = len(c)
    if n < 5000:
        return None
    S, atr = build_features(o, h, l, c)
    E = {H: expert_realized(S, c, H) for H in H_GRID}
    K = len(SCALES)
    w = np.ones(K) / K
    q_sel, H_sel, sl_sel = 0.90, 24, 2.5
    trades = []
    log_rows = []
    cut = int(n * TRAIN_FRAC)

    for b0 in range(0, n - BLOCK, BLOCK):
        b1 = min(b0 + BLOCK, n)
        # ---- ① 只用「b0 时刻已平仓」的历史更新参数 ----
        done = [t for t in trades if t['exit'] < b0] if trades else []
        nblocks = b0 // BLOCK
        explore = (nblocks % 4 == 0)      # 每 4 块探索一次其余参数组（bandit 轮换）
        # 专家权重：用 t <= b0-H 的实现收益（保证结算时间 < b0）
        for H in H_GRID:
            hi = max(0, b0 - H)
            if hi > 200:
                past = E[H][max(0, b0 - LOOKBACK_BLOCKS * BLOCK):hi]
                m = np.nanmean(past, axis=0)
                m = np.nan_to_num(m, nan=0.0)
                break
        # 用当前选定的 H 做权重（H 在下面随之更新）
        # 先按已平仓交易选 (q, H, sl)
        best = None
        if len(done) >= MIN_TRADES:
            for q in Q_GRID:
                for H in H_GRID:
                    for sl in SL_GRID:
                        sub = [t for t in done if t['q'] == q and t['H'] == H and t['sl'] == sl]
                        if len(sub) < MIN_TRADES:
                            continue
                        arr = np.array([t['ret'] for t in sub])
                        score = arr.mean() - 0.5 * arr.std(ddof=1) / np.sqrt(len(arr))  # 保守下界
                        if best is None or score > best[0]:
                            best = (score, q, H, sl)
        GRID = [(q, H, sl) for q in Q_GRID for H in H_GRID for sl in SL_GRID]
        if explore:
            q_sel, H_sel, sl_sel = GRID[nblocks % len(GRID)]
        elif best is not None:
            _, q_sel, H_sel, sl_sel = best

        hi = max(0, b0 - H_sel)
        if hi > 200:
            past = E[H_sel][max(0, b0 - LOOKBACK_BLOCKS * BLOCK):hi]
            m = np.nan_to_num(np.nanmean(past, axis=0), nan=0.0)
            sd = np.nanstd(m) + 1e-9
            eta = 1.0 / sd                              # 信噪比自动定步长（无固定 eta）
            w = np.exp(np.clip(eta * m, -20, 20))
            w = w / w.sum()

        # 阈值：过去 S 分布的 q 分位（只用 b0 之前）
        histS = S[max(0, b0 - LOOKBACK_BLOCKS * BLOCK):b0].sum(axis=1)  # 占位，真实阈值用加权分
        Scomb_hist = S[max(0, b0 - LOOKBACK_BLOCKS * BLOCK):b0] @ w
        thr = np.quantile(Scomb_hist, q_sel) if len(Scomb_hist) > 50 else 0.0

        # ---- ② 块内交易（成交在 t+1 开盘；持仓可在块内结束）----
        i = b0
        while i < b1 - 2:
            sc = float(S[i] @ w)
            if sc > thr or sc < -thr:
                side = 1 if sc > thr else -1
                entry = float(o[i + 1])
                if not np.isfinite(entry) or entry <= 0:
                    i += 1
                    continue
                a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
                stop = a * sl_sel
                end = min(i + 1 + H_sel, b1)
                ret = None
                j = i + 1
                while j < end:
                    if side == 1:
                        if stop > 0 and l[j] <= entry - stop:
                            ret = (entry - stop) / entry - 1 - FEE
                            break
                    else:
                        if stop > 0 and h[j] >= entry + stop:
                            ret = 1 - (entry + stop) / entry - FEE
                            break
                    j += 1
                if ret is None:
                    ret = side * (c[end - 1] / entry - 1) - FEE
                    j = end - 1
                trades.append(dict(t=b0, exit=int(j), ret=float(ret), q=q_sel, H=H_sel, sl=sl_sel,
                                   seg=('tr' if b0 < cut else 'te'), scale=int(np.argmax(np.abs(w)))))
                i = j + 1
            else:
                i += 1

        log_rows.append(dict(b=b0, q=q_sel, H=H_sel, sl=sl_sel,
                             w=[round(float(x), 3) for x in w],
                             best_scale=int(np.argmax(w))))

    if not trades:
        return None
    tr = pd.DataFrame(trades)
    return dict(sym=None, tag=tag, trades=tr, log=log_rows, n=n)


def summarize(df, name):
    if df is None or len(df) == 0:
        say(f'  {name}: 无交易')
        return None
    r = df.ret.values
    eq = np.cumprod(1 + r * 0.10)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    out = dict(name=name, n=int(len(r)), mean=float(r.mean()), med=float(np.median(r)),
               win=float((r > 0).mean()), total=float(eq[-1] - 1), dd=dd,
               t=float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else 0.0)
    say(f'  {name:<10} n={out["n"]:>6}  均值 {out["mean"]:+.4%}  中位 {out["med"]:+.4%}  '
        f'胜率 {out["win"]:.1%}  t={out["t"]:+.2f}  资金 {out["total"]:+.1%}  回撤 {dd:+.1%}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=int, default=60)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-shuffle', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    say('=' * 100)
    say('gambler_adaptive · 无前视 / 回看窗口自选 / 参数在线自选')
    say(f'  尺度 {SCALES} 根(5m) | 持有 {H_GRID} | 止损 {SL_GRID}×ATR | 门槛分位 {Q_GRID} | 费 {FEE:.4%}')
    say('=' * 100)

    dt = {'symbol': 'category', 'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:args.symbols]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

    rng = np.random.default_rng(20261018)
    all_tr, all_te, logs, shuf_te = [], [], [], []
    for si, sym in enumerate(syms):
        o, h, l, c = load_symbol(sym, df)
        r = run_symbol(o, h, l, c, rng, 'REAL')
        if r is None:
            continue
        r['trades']['sym'] = sym
        all_tr.append(r['trades'])
        logs.append((sym, r['log']))
        if not args.no_shuffle:
            o2, h2, l2, c2 = shuffle_price(o, h, l, c, rng)
            r2 = run_symbol(o2, h2, l2, c2, rng, 'SHUF')
            if r2 is not None:
                r2['trades']['sym'] = sym
                shuf_te.append(r2['trades'])
        if (si + 1) % 10 == 0:
            say(f'    … {si+1}/{len(syms)} 币  {time.time()-t0:.0f}s')

    T = pd.concat(all_tr, ignore_index=True)
    S_all = pd.concat(shuf_te, ignore_index=True) if shuf_te else None

    say('\n' + '=' * 100)
    say('【一】真实序列')
    a = summarize(T[T.seg == 'tr'], '训练段')
    b = summarize(T[T.seg == 'te'], '样本外')
    say('\n【二】打乱对照（结构破坏）')
    c2 = summarize(S_all[S_all.seg == 'te'], '对照样本外') if S_all is not None else None

    say('\n【三】模型自己选了什么')
    rows = []
    for sym, lg in logs:
        for r in lg:
            rows.append(r)
    L = pd.DataFrame(rows)
    if len(L):
        cnt = L.best_scale.value_counts(normalize=True).sort_index()
        say('  权重最大的尺度分布： ' + '  '.join(
            f'{SCALES[int(k)]}根={v:.0%}' for k, v in cnt.items()))
        say('  门槛分位选择分布： ' + '  '.join(f'q={k}:{v:.0%}' for k, v in L.q.value_counts(normalize=True).items()))
        say('  持有期选择分布： ' + '  '.join(f'H={k}根:{v:.0%}' for k, v in L.H.value_counts(normalize=True).items()))
        say('  止损选择分布： ' + '  '.join(f'sl={k}×ATR:{v:.0%}' for k, v in L.sl.value_counts(normalize=True).items()))
        # 权重漂移强度：说明"窗口不是固定的"
        wmat = np.array(L['w'].tolist(), dtype=float)
        say(f'  专家权重的时间均值： ' + '  '.join(f'{SCALES[i]}根={wmat[:,i].mean():.2f}' for i in range(len(SCALES))))
        say(f'  权重在尺度间的标准差（越大=越挑剔）： {wmat.std(axis=1).mean():.3f}')

    say('\n【四】分币样本外')
    if len(T[T.seg == 'te']):
        g = T[T.seg == 'te'].groupby('sym').ret.agg(['count', 'mean'])
        g = g[g['count'] >= 10].sort_values('mean', ascending=False)
        say(f'  有交易的币 {len(g)}；样本外均值>0 的币 {int((g["mean"]>0).sum())}/{len(g)}')
        say('  最好 5 币: ' + '  '.join(f'{i}:{v.mean:+.3%}(n{int(v["count"])})' for i, v in g.head(5).iterrows()))
        say('  最差 5 币: ' + '  '.join(f'{i}:{v.mean:+.3%}(n{int(v["count"])})' for i, v in g.tail(5).iterrows()))

    say('\n【五】信息泄露自检')
    say('  · 参数更新只用 exit<b0 的已平仓交易（脚本内 done 过滤）')
    say('  · 标准化用扩张窗口（第 i 点只用 [0,i]）')
    say('  · 成交在 t+1 开盘价，信号在 t 收盘')
    say('  · 专家实现收益切片上限 b0-H，保证结算在块起点之前')
    say(f'\nTOTAL {time.time()-t0:.0f}s')

    res = dict(param=dict(scales=SCALES, H=H_GRID, sl=SL_GRID, q=Q_GRID, fee=FEE,
                          train_frac=TRAIN_FRAC, block=BLOCK, symbols=[str(s) for s in syms]),
               real_tr=a, real_te=b, shuffle_te=c2)
    with open(os.path.join(OUT, 'gambler_adaptive.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    T.to_csv(os.path.join(OUT, 'gambler_adaptive_trades.csv'), index=False, encoding='utf-8-sig')
    if S_all is not None:
        S_all.to_csv(os.path.join(OUT, 'gambler_adaptive_shuffle_trades.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
