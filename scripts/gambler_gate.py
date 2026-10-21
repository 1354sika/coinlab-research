# -*- coding: utf-8 -*-
"""
gambler_gate.py · 牌面测谎仪

要回答的问题：现在这 12 个特征里，哪些是"真牌"（抓的是时间上的因果结构），
哪些只是"统计巧合"（在打乱顺序的历史上照样有预测力）。

判法：
  真序列  —— 用前 70% 定分位，在后 30% 上量：特征高分位组的未来收益 − 低分位组
  打乱序列 —— 把 5m 收益按 24h 块随机重排（价格/量/taker 同块一起搬），
              再算同一个差值。结构被打碎，但分布保留。
  真牌  ⇒ 真序列有预测力、打乱后消失
  巧合  ⇒ 两者都有（来自波动聚集这类统计性质，不是可交易的时间结构）

显著性按"币"聚合算 t 值（n=币数），不用分钟级样本算 —— 分钟级严重自相关，
会把 t 值虚高（旧版 t=-24 就是这么来的），这里刻意取严。

用法：set GATE_NSYM=6 && python -X utf8 gambler_gate.py
输出：evidence/gambler_gate.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_memory import build_features, BLOCK, SCALES

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 12                 # 未来 12 根 5m = 1 小时
NSYM = int(os.environ.get('GATE_NSYM', '6'))
TRAIN = 0.70
MIN_BARS = 20000

FEAT_NAMES = [f'mom{k}' for k in SCALES] + ['vol', 'volr', 'taker', 'pos24']
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def shuffle_all(o, h, l, c, v, tb, rng, block=BLOCK):
    """价格 + 量 + taker 按同一批 24h 块一起重排。"""
    n = len(c)
    r = np.zeros(n)
    r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))

    def cat(a):
        out = np.empty(n)
        pos = 0
        for b in order:
            seg = blocks[b]
            out[pos:pos + len(seg)] = a[seg]
            pos += len(seg)
        return out

    rr = cat(r)
    newc = np.exp(np.cumsum(rr)) * c[0]
    ratio = newc / np.maximum(c, 1e-12)
    return o * ratio, h * ratio, l * ratio, newc, cat(v), cat(tb)


def feat_delta(X, y, q_lo, q_hi):
    """每个特征：高分位组未来收益 − 低分位组未来收益"""
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        col = X[:, j]
        v_hi = y[col >= q_hi[j]]
        v_lo = y[col <= q_lo[j]]
        if len(v_hi) > 100 and len(v_lo) > 100:
            out[j] = np.nanmean(v_hi) - np.nanmean(v_lo)
    return out


def tstat(a):
    a = a[~np.isnan(a)]
    if len(a) < 3:
        return np.nan
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_gate · 牌面测谎仪（真序列 vs 24h 块打乱）')
    say(f'  特征 {len(FEAT_NAMES)} 个 | 持有 H={H} 根 | 训练定分位 {TRAIN:.0%} | 币 {NSYM}')
    say('  显著性按币聚合（n=币数），刻意避开分钟级自相关虚高')
    say('=' * 100)
    say('  载入 ...')
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'high', 'low',
                                    'close', 'volume', 'taker_base'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time() - t0:.0f}s')

    rng = np.random.default_rng(20261018)
    R, S = [], []
    used = []
    for i, s in enumerate(syms):
        g = df[df.symbol == s].sort_values('open_time')
        if len(g) < MIN_BARS:
            say(f'  - {s} 跳过（{len(g)} 根）')
            continue
        o = g.open.values.astype(np.float64)
        h = g.high.values.astype(np.float64)
        l = g.low.values.astype(np.float64)
        c = g.close.values.astype(np.float64)
        v = g.volume.values.astype(np.float64)
        tb = g.taker_base.values.astype(np.float64)
        n = len(c)
        cut = int(n * TRAIN)

        # ---- 真序列 ----
        X, _, _, _, _ = build_features(g)
        fwd = np.concatenate([c[H:] / np.maximum(c[:-H], 1e-12) - 1, np.full(H, np.nan)])
        q_lo = np.nanquantile(X[:cut], 0.2, axis=0)
        q_hi = np.nanquantile(X[:cut], 0.8, axis=0)
        d_r = feat_delta(X[cut:], fwd[cut:], q_lo, q_hi)

        # ---- 打乱序列 ----
        o2, h2, l2, c2, v2, tb2 = shuffle_all(o, h, l, c, v, tb, rng)
        g2 = pd.DataFrame({'open': o2, 'high': h2, 'low': l2, 'close': c2,
                           'volume': v2, 'taker_base': tb2})
        X2, _, _, _, _ = build_features(g2)
        fwd2 = np.concatenate([c2[H:] / np.maximum(c2[:-H], 1e-12) - 1, np.full(H, np.nan)])
        q2_lo = np.nanquantile(X2[:cut], 0.2, axis=0)
        q2_hi = np.nanquantile(X2[:cut], 0.8, axis=0)
        d_s = feat_delta(X2[cut:], fwd2[cut:], q2_lo, q2_hi)

        R.append(d_r)
        S.append(d_s)
        used.append(s)
        say(f'  {i + 1}/{len(syms)} {s}  {n:,} 根  {time.time() - t0:.0f}s')

    R = np.array(R)
    S = np.array(S)
    n_sym = len(used)

    say('')
    say('=' * 100)
    say(f'【一】逐特征：真序列 vs 打乱（单位 %/笔，按币聚合 n={n_sym}）')
    say(f'{"特征":<9}{"真差值":>10}{"打乱差值":>11}{"真-打乱":>11}{"t(真)":>8}{"t(差)":>8}   判定')
    res = []
    for j, nm in enumerate(FEAT_NAMES):
        mr = np.nanmean(R[:, j]) * 100
        ms = np.nanmean(S[:, j]) * 100
        md = np.nanmean(R[:, j] - S[:, j]) * 100
        tr = tstat(R[:, j])
        td = tstat(R[:, j] - S[:, j])
        if abs(tr) >= 2 and abs(td) >= 2 and np.sign(tr) == np.sign(td):
            verdict = '✅ 真牌'
        elif abs(tr) >= 2:
            verdict = '⚠ 只在真序列显著（可能是统计性质）'
        else:
            verdict = '❌ 无预测力'
        res.append(dict(feat=nm, d_real=mr, d_shuf=ms, d_gap=md, t_real=tr, t_gap=td, verdict=verdict))
        say(f'{nm:<9}{mr:>+9.3f}%{ms:>+10.3f}%{md:>+10.3f}%{tr:>+8.2f}{td:>+8.2f}   {verdict}')

    say('')
    say('=' * 100)
    real = [r for r in res if r['verdict'].startswith('✅')]
    weak = [r for r in res if r['verdict'].startswith('⚠')]
    say(f'【二】汇总：真牌 {len(real)}/{len(res)}，只在真序列显著 {len(weak)}/{len(res)}，无预测力 {len(res) - len(real) - len(weak)}/{len(res)}')
    if real:
        say('  真牌：' + '  '.join(f"{r['feat']}({r['d_gap']:+.3f}%)" for r in real))
    if weak:
        say('  待查：' + '  '.join(f"{r['feat']}(t={r['t_real']:+.1f})" for r in weak))
    say('')
    say('  判据说明：')
    say('  · 真牌 = 真序列显著 且 打乱后消失（t(真-打乱) 也显著）')
    say('  · 若几乎全是"无预测力"，说明手工牌面确实不含可交易信息 —— 模型该自己去找牌')
    say('  · 若真牌很少（1~2 个），后续"长出来的特征"必须过同一道闸才允许进场')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_gate.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(res).to_csv(os.path.join(OUT, 'gambler_gate.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
