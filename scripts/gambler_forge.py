# -*- coding: utf-8 -*-
"""
gambler_forge.py · 造牌铺（全枚举组合 + 测谎闸门 + 多重检验校正）

在 gambler_gate.py 的基础上再进一步：
  ① 不再只测 12 个手工基元，而是把基元两两组合（加/减/乘/比/取大/取小）全枚举，
     外加一元变形（平方/取负/符号/绝对值）——共 912 个候选特征。
  ② 每个候选同时跑"真序列"和"24h 块打乱序列"。
  ③ 关键的多重检验校正：打乱序列上 912 个候选的最大 |t| 就是"纯噪声的天花板"。
     真序列候选必须超过这个天花板，才算长出了一张真牌；否则搜得越多，越容易
     撞出一个假牌（这正是手工加特征的死因）。

判据（刻意取严）：
  过闸 = |t(真)| >= 2.5  且  |t(真)| > 噪声天花板 + 0.5  且  t(真) 与 t(真-打乱) 同号
  显著性按"币"聚合（n=币数），不用分钟级样本，避免自相关虚高 t 值。

用法：set FORGE_NSYM=6 && python -X utf8 gambler_forge.py
输出：evidence/gambler_forge.txt / .csv
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_memory import build_features, BLOCK, SCALES
from gambler_gate import shuffle_all, tstat

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 12
NSYM = int(os.environ.get('FORGE_NSYM', '6'))
TRAIN = 0.70
CLIP = 10.0
MIN_BARS = 20000

BASE = [f'mom{k}' for k in SCALES] + ['vol', 'volr', 'taker', 'pos24']
NB = len(BASE)

BIN = [
    ('add', lambda a, b: a + b),
    ('sub', lambda a, b: a - b),
    ('mul', lambda a, b: a * b),
    ('div', lambda a, b: a / np.where(np.abs(b) > 0.3, b, np.nan)),
    ('mn',  np.minimum),
    ('mx',  np.maximum),
]
UN = [
    ('sq',  lambda a: a * a),
    ('neg', lambda a: -a),
    ('sgn', np.sign),
    ('abs', np.abs),
]

TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def candidates():
    """返回 [(名字, 一元? , op名, i, j)]，i/j 为基元下标"""
    out = []
    for i, j in itertools.product(range(NB), range(NB)):
        for op, _ in BIN:
            out.append((f'{op}({BASE[i]},{BASE[j]})', 'bin', op, i, j))
    for i in range(NB):
        for op, _ in UN:
            out.append((f'{op}({BASE[i]})', 'un', op, i, i))
    return out


def build(cands, X):
    """为一批候选构造取值矩阵 (n, m)，返回 float32"""
    n = X.shape[0]
    V = np.empty((n, len(cands)), dtype=np.float32)
    binmap = dict(BIN)
    unmap = dict(UN)
    for k, (_, kind, op, i, j) in enumerate(cands):
        if kind == 'bin':
            v = binmap[op](X[:, i], X[:, j])
        else:
            v = unmap[op](X[:, i])
        V[:, k] = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -CLIP, CLIP)
    return V


def delta_col(V, fwd, cut, m=None):
    """每列：评估段里 高20%组未来收益 − 低20%组未来收益（分位用训练段定）"""
    m = V.shape[1]
    out = np.full(m, np.nan)
    y = fwd[cut:]
    ok_y = np.isfinite(y)
    for k in range(m):
        v = V[:, k]
        ql, qh = np.quantile(v[:cut], [0.2, 0.8])
        if not np.isfinite(ql) or not np.isfinite(qh) or qh - ql < 1e-9:
            continue
        vs = v[cut:]
        hi = ok_y & (vs >= qh)
        lo = ok_y & (vs <= ql)
        if hi.sum() > 200 and lo.sum() > 200:
            out[k] = y[hi].mean() - y[lo].mean()
    return out


def main():
    t0 = time.time()
    cands = candidates()
    say('=' * 110)
    say('gambler_forge · 造牌铺（全枚举组合 + 测谎闸门 + 多重检验校正）')
    say(f'  基元 {NB} 个 × 二元 {len(BIN)} 种 × 一元 {len(UN)} 种 ⇒ 候选 {len(cands)} 个')
    say(f'  持有 H={H} 根 | 训练定分位 {TRAIN:.0%} | 币 {NSYM} | 显著性按币聚合')
    say('=' * 110)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'high', 'low',
                                    'close', 'volume', 'taker_base'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time() - t0:.0f}s')

    rng = np.random.default_rng(20261018)
    store = []
    for i, s in enumerate(syms):
        g = df[df.symbol == s].sort_values('open_time')
        if len(g) < MIN_BARS:
            continue
        o = g.open.values.astype(np.float64)
        h = g.high.values.astype(np.float64)
        l = g.low.values.astype(np.float64)
        c = g.close.values.astype(np.float64)
        v = g.volume.values.astype(np.float64)
        tb = g.taker_base.values.astype(np.float64)
        n = len(c)
        cut = int(n * TRAIN)

        X, _, _, _, _ = build_features(g)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        fwd = np.concatenate([c[H:] / np.maximum(c[:-H], 1e-12) - 1, np.full(H, np.nan)])

        o2, h2, l2, c2, v2, tb2 = shuffle_all(o, h, l, c, v, tb, rng)
        g2 = pd.DataFrame({'open': o2, 'high': h2, 'low': l2, 'close': c2,
                           'volume': v2, 'taker_base': tb2})
        X2, _, _, _, _ = build_features(g2)
        X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        fwd2 = np.concatenate([c2[H:] / np.maximum(c2[:-H], 1e-12) - 1, np.full(H, np.nan)])

        store.append((s, X, fwd, X2, fwd2, cut))
        say(f'  {i + 1}/{len(syms)} {s}  {n:,} 根  {time.time() - t0:.0f}s')

    n_sym = len(store)
    say('')
    say(f'  开始枚举 {len(cands)} 个候选 × {n_sym} 币 × 2（真/打乱）...')
    R = np.full((n_sym, len(cands)), np.nan)
    S = np.full((n_sym, len(cands)), np.nan)
    for si, (s, X, fwd, X2, fwd2, cut) in enumerate(store):
        VR = build(cands, X)
        R[si] = delta_col(VR, fwd, cut)
        del VR
        VS = build(cands, X2)
        S[si] = delta_col(VS, fwd2, cut)
        del VS
        say(f'  枚举 {si + 1}/{n_sym} {s}  {time.time() - t0:.0f}s')

    # 逐候选算 t
    tR = np.full(len(cands), np.nan)
    tS = np.full(len(cands), np.nan)
    gap = np.full(len(cands), np.nan)
    tG = np.full(len(cands), np.nan)
    for k in range(len(cands)):
        r = R[:, k]
        sg = S[:, k]
        tR[k] = tstat(r)
        tS[k] = tstat(sg)
        gap[k] = np.nanmean(r - sg)
        tG[k] = tstat(r - sg)

    ceiling = np.nanmax(np.abs(tS))                     # 纯噪声天花板
    say('')
    say('=' * 110)
    say(f'【一】噪声天花板（打乱序列上 {len(cands)} 个候选的最大 |t|）= {ceiling:.2f}')
    say(f'     直白说：随便乱搜 {len(cands)} 个假特征，最好那个也能到 |t|={ceiling:.2f}，')
    say(f'     所以真牌必须明显超过它，否则就是搜出来的假货。')

    ok = (np.abs(tR) >= 2.5) & (np.abs(tR) > ceiling + 0.5) & \
         (np.abs(tG) >= 2.5) & (np.sign(tR) == np.sign(tG))
    idx = np.where(ok)[0]
    say('')
    say(f'【二】过闸特征 {len(idx)} 个 / {len(cands)}')
    if len(idx) == 0:
        say('  ❌ 没有任何组合能同时满足"真序列显著 + 超过噪声天花板 + 打乱后消失"。')
    else:
        order = idx[np.argsort(-np.abs(tG[idx]))]
        say(f'  {"特征":<34}{"真差值":>10}{"打乱差值":>11}{"差值":>10}{"t(真)":>8}{"t(差)":>8}')
        for k in order:
            say(f'  {cands[k][0]:<34}{np.nanmean(R[:, k]) * 100:>+9.3f}%'
                f'{np.nanmean(S[:, k]) * 100:>+10.3f}%'
                f'{gap[k] * 100:>+9.3f}%{tR[k]:>+8.2f}{tG[k]:>+8.2f}')

    say('')
    say(f'【三】未过闸但真序列本身显著的（可能是统计性质，供参考）：')
    weak = np.where((np.abs(tR) >= 2.5) & ~ok)[0]
    if len(weak) == 0:
        say('  无')
    else:
        order = weak[np.argsort(-np.abs(tR[weak]))][:15]
        for k in order:
            say(f'  {cands[k][0]:<34}{np.nanmean(R[:, k]) * 100:>+9.3f}%'
                f'{np.nanmean(S[:, k]) * 100:>+10.3f}%{tR[k]:>+8.2f}{tG[k]:>+8.2f}')

    say('')
    say(f'【四】结论')
    say(f'  真牌 {len(idx)} 个。若为 0，说明"从现有基元里组合出可交易特征"这条路在 5m 上不成立 ——')
    say(f'  下一步必须换原料（订单流/盘口/跨币/资金费），而不是继续组合现有的价格量。')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_forge.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    out = pd.DataFrame(dict(name=[c[0] for c in cands], t_real=tR, t_shuf=tS,
                            d_real=np.nanmean(R, axis=0), d_shuf=np.nanmean(S, axis=0),
                            d_gap=gap, t_gap=tG, passed=ok))
    out.sort_values('t_gap', key=lambda x: -np.abs(x)).to_csv(
        os.path.join(OUT, 'gambler_forge.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
