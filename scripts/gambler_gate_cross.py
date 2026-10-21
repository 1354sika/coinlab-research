# -*- coding: utf-8 -*-
"""
gambler_gate_cross.py · 换原料：跨币 / 市场宽度 / 时间结构 过闸门

为什么做这个：
  gambler_forge.py 把 12 个价格量基元全枚举（912 个组合），过闸 0 个 ——
  说明"在价格/量/taker 内部怎么组合"这条路在 5m 上不成立。
  所以换**不同源**的原料：币与币之间的关系、全市场的状态、一天中的时刻。

跨币数据：klines_1m_1y.csv（60 币 × 1 年 1m）
新原料：
  ① 市场宽度 breadth_k  = 60 币里过去 k 根上涨的比例（人群行为，不是单币价格变形）
  ② BTC 领先信号 btc_mom_k
  ③ 超额动量 rel_mom_k  = 本币动量 − BTC 动量（剥掉大盘，剩下的才是自己的牌）
  ④ 截面离散 disp_k     = 60 币动量的截面标准差（分化程度）
  ⑤ 全市场量能 mkt_volr = 全市场成交量 / 过去 1 天均值
  ⑥ 时间结构 hour/dow 的正弦编码（规则决定的、可重复的时段效应）

对照（关键）：
  所有币的收益按**同一组 1 天块顺序**打乱 —— 跨币相关性保留，时间因果链打断。
  这样"打乱后仍有效"的才是人群作息（时间结构），"打乱后消失"的才是真结构。

用法：set CROSS_N=60 && python -X utf8 gambler_gate_cross.py
输出：evidence/gambler_gate_cross.txt / .csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate import tstat

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 60                     # 未来 60 根 1m = 1 小时
BAR_PER_DAY = 1440
BLOCK = BAR_PER_DAY        # 打乱块 = 1 天
TRAIN = 0.70
NSYM = int(os.environ.get('CROSS_N', '60'))

TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def mom(C, k):
    """C: (T,S) 收盘价矩阵 → 过去 k 根收益率（nan 安全：缺价不参与）"""
    out = np.full_like(C, np.nan)
    a = C[k:]
    b = C[:-k]
    ok = np.isfinite(a) & np.isfinite(b)
    v = np.full_like(a, np.nan)
    v[ok] = a[ok] / b[ok] - 1
    out[k:] = v
    return out


def zs(A, win=1440):
    """按币做扩张 z 化（只用到当前及过去）"""
    T, S = A.shape
    out = np.empty_like(A)
    for s in range(S):
        x = A[:, s]
        cs = np.nancumsum(np.nan_to_num(x))
        cs2 = np.nancumsum(np.nan_to_num(x) ** 2)
        cnt = np.arange(1, T + 1, dtype=np.float64)
        mu = cs / cnt
        sd = np.sqrt(np.maximum(cs2 / cnt - mu * mu, 0))
        z = np.zeros(T)
        ok = sd > 1e-12
        z[ok] = (x[ok] - mu[ok]) / sd[ok]
        out[:, s] = z
    return out


def shuffle_panel(C, rng, block=BLOCK):
    """块级重排：保留块内结构 + 跨币同步 + 总收益，只打断时间因果

    v3 的 SHUF 崩到 −12%/笔，根因在**块接缝**：
      原来用 argsort 按块重排整个 R，而每块的第一个元素 R[k*block] 是
      "跨块收益"（C[k*block]/C[k*block−1]−1）。重排后它接在了**另一块的末尾**，
      于是每个块接缝都插入一个幅度等于真实 24h 变化的假跳变（1 分钟内暴涨/暴跌
      几十个百分点），止损被这些假跳变反复打穿 —— 这才是 −12%/笔 的来源，
      不是"记噪声"，是价格面板本身被打坏了。

    修法：把跨块收益单独抽出，**只重排、不重算**（总和不变 ⇒ 总收益严格不变），
    块内收益保持连续，接缝处的跳变仍取自真实跨块收益分布，幅度正常。
    """
    T, S = C.shape
    R = np.zeros_like(C)
    R[1:] = np.log(np.maximum(C[1:], 1e-12) / np.maximum(C[:-1], 1e-12))
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)   # 缺价处收益记 0

    nb = max(2, T // block)
    head = R[:nb * block].reshape(nb, block, S)
    cross = head[:, 0, :].copy()          # 每块首 = 跨块收益
    inner = head.copy()
    inner[:, 0, :] = 0.0                  # 块内从块首起连续，不夹假跳变
    inner2 = inner[rng.permutation(nb)].reshape(nb * block, S)
    inner2[::block] = cross[rng.permutation(nb)]   # 接缝放真实跨块收益（只换顺序）
    R2 = np.concatenate([inner2, R[nb * block:]], axis=0)

    base = np.where(np.isfinite(C[0]), C[0], np.nanmedian(C, axis=0))
    return np.exp(np.cumsum(R2, axis=0)) * base


def build_panel_feats(C, V, TB, idx_hour, idx_dow):
    """返回 {名字: (T,S) 特征矩阵}"""
    T, S = C.shape
    F = {}
    for k in (15, 60, 240):
        M = mom(C, k)
        F[f'mom{k}'] = zs(M)
        valid = np.isfinite(M)
        nv = valid.sum(axis=1, keepdims=True)
        enough = nv >= 30
        F[f'relmom{k}'] = zs(np.where(enough, M - np.nanmean(M, axis=1, keepdims=True), np.nan))
        F[f'disp{k}'] = np.repeat(np.where(enough,
                                           np.nanstd(M, axis=1, keepdims=True), np.nan), S, axis=1)
        br = np.where(enough, ((M > 0) & valid).sum(axis=1, keepdims=True) / np.maximum(nv, 1), np.nan)
        F[f'breadth{k}'] = np.repeat(br, S, axis=1)
    M15 = mom(C, 15)
    F['btc_mom15'] = np.repeat(M15[:, [0]], S, axis=1)
    F['btc_lead'] = zs(F['btc_mom15'] - F['mom15'])
    Vm = V / np.maximum(pd.DataFrame(V).rolling(BAR_PER_DAY, min_periods=60).mean().values, 1e-12)
    F['volr'] = zs(Vm - 1)
    mktv = np.repeat(np.nansum(Vm, axis=1, keepdims=True) / S, S, axis=1)
    F['mkt_volr'] = zs(mktv - 1)
    F['hour_sin'] = np.repeat(np.sin(2 * np.pi * idx_hour / 24)[:, None], S, axis=1)
    F['hour_cos'] = np.repeat(np.cos(2 * np.pi * idx_hour / 24)[:, None], S, axis=1)
    F['dow_sin'] = np.repeat(np.sin(2 * np.pi * idx_dow / 7)[:, None], S, axis=1)
    F['dow_cos'] = np.repeat(np.cos(2 * np.pi * idx_dow / 7)[:, None], S, axis=1)
    return F


def delta_for(F, fwd, cut):
    """每币：高20%组 − 低20%组（分位用训练段），返回 (S,)"""
    T, S = fwd.shape
    out = np.full(S, np.nan)
    for s in range(S):
        v = F[:, s]
        y = fwd[:, s]
        tr = v[:cut]
        ql, qh = np.nanquantile(tr, [0.2, 0.8])
        if not np.isfinite(ql) or not np.isfinite(qh) or abs(qh - ql) < 1e-9:
            continue
        vs = v[cut:]
        ys = y[cut:]
        ok = np.isfinite(vs) & np.isfinite(ys)
        hi = ok & (vs >= qh)
        lo = ok & (vs <= ql)
        if hi.sum() > 500 and lo.sum() > 500:
            out[s] = np.nanmean(ys[hi]) - np.nanmean(ys[lo])
    return out


def main():
    t0 = time.time()
    say('=' * 108)
    say('gambler_gate_cross · 换原料：跨币 / 市场宽度 / 分化 / 时间结构')
    say(f'  持有 H={H} 根 1m | 打乱块=1天 | 训练定分位 {TRAIN:.0%} | 币 {NSYM}')
    say('  对照：所有币用同一块顺序打乱（跨币相关保留，时间因果打断）')
    say('=' * 108)

    say('  载入 1m ...')
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume', 'taker_base'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    say(f'  {len(df):,} 行 / {len(syms)} 币  {time.time() - t0:.0f}s')

    # 对齐到共同时间轴
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv_c = df.pivot_table(index='t', columns='symbol', values='close').sort_index()
    piv_v = df.pivot_table(index='t', columns='symbol', values='volume').sort_index()
    piv_tb = df.pivot_table(index='t', columns='symbol', values='taker_base').sort_index()
    piv_c = piv_c[syms]
    piv_v = piv_v[syms]
    piv_tb = piv_tb[syms]
    C = piv_c.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    say(f'  缺失率 {np.isnan(C).mean():.3%}（不填 0，宽度分母只用有效币）')
    V = np.nan_to_num(piv_v.values.astype(np.float64), nan=0.0)
    TB = np.nan_to_num(piv_tb.values.astype(np.float64), nan=0.0)
    T, S = C.shape
    say(f'  对齐矩阵 {T:,} 时间点 × {S} 币  {time.time() - t0:.0f}s')

    idx = piv_c.index
    hour = idx.hour.values
    dow = idx.dayofweek.values

    cut = int(T * TRAIN)
    fwd = np.full((T, S), np.nan)
    fwd[:-H] = C[H:] / np.maximum(C[:-H], 1e-12) - 1

    say('  构造真序列特征 ...')
    FR = build_panel_feats(C, V, TB, hour, dow)
    names = list(FR.keys())
    say(f'  特征 {len(names)} 个：{", ".join(names)}')

    rng = np.random.default_rng(20261018)
    say('  打乱面板（同块顺序）...')
    C2 = shuffle_panel(C, rng)
    fwd2 = np.full((T, S), np.nan)
    fwd2[:-H] = C2[H:] / np.maximum(C2[:-H], 1e-12) - 1
    # 成交量/时间戳跟同一块顺序搬
    def shuffle_rows(A):
        idx0 = np.arange(T)
        blocks = [idx0[i:i + BLOCK] for i in range(0, T, BLOCK)]
        order = rng.permutation(len(blocks))
        out = np.empty_like(A)
        pos = 0
        for b in order:
            seg = blocks[b]
            out[pos:pos + len(seg)] = A[seg]
            pos += len(seg)
        return out
    V2 = shuffle_rows(V)
    TB2 = shuffle_rows(TB)
    FS = build_panel_feats(C2, V2, TB2, hour, dow)

    say('  逐个特征过闸 ...')
    rows = []
    for nm in names:
        dR = delta_for(FR[nm], fwd, cut)
        dS = delta_for(FS[nm], fwd2, cut)
        tr = tstat(dR)
        ts = tstat(dS)
        tg = tstat(dR - dS)
        rows.append(dict(feat=nm, d_real=np.nanmean(dR) * 100, d_shuf=np.nanmean(dS) * 100,
                         d_gap=np.nanmean(dR - dS) * 100, t_real=tr, t_shuf=ts, t_gap=tg))
        say(f'    {nm:<14} 真 {np.nanmean(dR) * 100:+.3f}%  打乱 {np.nanmean(dS) * 100:+.3f}%  '
            f't(真) {tr:+.2f}  t(差) {tg:+.2f}  {time.time() - t0:.0f}s')

    res = pd.DataFrame(rows)
    ceiling = float(np.nanmax(np.abs(res.t_shuf)))
    ok = (res.t_real.abs() >= 2.5) & (res.t_real.abs() > ceiling + 0.5) & \
         (res.t_gap.abs() >= 2.5) & (np.sign(res.t_real) == np.sign(res.t_gap))

    say('')
    say('=' * 108)
    say(f'【一】噪声天花板 = {ceiling:.2f}（打乱面板上 {len(names)} 个特征的最大 |t|）')
    say('')
    say(f'【二】过闸路径（口径同 forge：|t(真)|≥2.5 且 超过天花板+0.5 且 打乱后消失）')
    if ok.sum() == 0:
        say('  ❌ 0 个')
    else:
        for _, r in res[ok].sort_values('t_gap', key=lambda x: -np.abs(x)).iterrows():
            say(f'  ✅ {r.feat:<14} 真 {r.d_real:+.3f}%  打乱 {r.d_shuf:+.3f}%  '
                f'差 {r.d_gap:+.3f}%  t(真) {r.t_real:+.2f}  t(差) {r.t_gap:+.2f}')
    say('')
    say('【三】按 |t(真-打乱)| 排名前 10（不代表过闸）')
    for _, r in res.reindex(res.t_gap.abs().sort_values(ascending=False).index).head(10).iterrows():
        say(f'  {r.feat:<14} 真 {r.d_real:+.3f}%  打乱 {r.d_shuf:+.3f}%  '
            f'差 {r.d_gap:+.3f}%  t(真) {r.t_real:+.2f}  t(差) {r.t_gap:+.2f}')
    say('')
    say('【四】读法')
    say('  · 打乱后仍显著（t(差)≈0 但 t(真) 大）⇒ 人群作息/时段效应，不是因果结构，但可作为状态输入')
    say('  · 真序列与打乱都接近 0 ⇒ 这张牌没用')
    say('  · 只有 t(差) 显著 ⇒ 才是"时间上有因果结构的牌"')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_gate_cross.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    res.to_csv(os.path.join(OUT, 'gambler_gate_cross.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
