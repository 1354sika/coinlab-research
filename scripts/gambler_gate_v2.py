# -*- coding: utf-8 -*-
"""
gambler_gate_v2.py · 第二批牌面过闸（跨币结构 / 形态 / 流动性）

第一批（gambler_gate_cross.py）20 个特征里 3 个过闸：
  disp240 / breadth60 / dow_cos —— 但都是"全市场统一取值"，只能做大盘择时。

这一批换方向，要的是**币与币之间、以及单币形态**的牌，
这样模型才有可能做"选币"而不是"押大盘方向"。

候选（12 个，全部用现有 1m 数据可算）：
  ① lead5        BTC 5 分钟动量 − 本币 5 分钟动量（领先滞后）
  ② lead15       BTC 15 分钟动量 − 本币 15 分钟动量
  ③ beta240      本币与全市场收益的 240 根滚动相关（联动强度）
  ④ streak       连续同向根数（正=连涨，负=连跌）
  ⑤ brkhi240     现价 / 前 240 根最高价 − 1（突破度）
  ⑥ brklo240     现价 / 前 240 根最低价 − 1
  ⑦ volratio     20 根波动 / 240 根波动（波动结构陡峭度）
  ⑧ amtshare     本币成交额 / 全市场成交额（z 化，流动性地位）
  ⑨ skewmkt      截面 60 根动量的偏度（市场分化方向）
 ⑩ updnvol      过去 60 根上涨时的量 / 下跌时的量
 ⑪ takim240     240 根 taker 失衡均值（谁在主动成交）
 ⑫ rangepos     现价在 240 根高低区间中的位置

闸门口径与 gambler_gate_cross.py 完全一致（含 1 天块打乱、按币聚合 t、噪声天花板）。

用法：set CROSS_N=60 && python -X utf8 gambler_gate_v2.py
输出：evidence/gambler_gate_v2.txt / .csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate import tstat
from gambler_gate_cross import mom, shuffle_panel

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H = 60
TRAIN = 0.70
NSYM = int(os.environ.get('CROSS_N', '60'))
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def exp_z_rows(A):
    """按列（币）做扩张 z 化，只用过去"""
    T, S = A.shape
    out = np.empty_like(A, dtype=np.float32)
    for s in range(S):
        x = np.nan_to_num(A[:, s])
        cs = np.cumsum(x)
        cs2 = np.cumsum(x * x)
        cnt = np.arange(1, T + 1, dtype=np.float64)
        mu = cs / cnt
        sd = np.sqrt(np.maximum(cs2 / cnt - mu * mu, 0))
        z = np.zeros(T)
        ok = sd > 1e-12
        z[ok] = (x[ok] - mu[ok]) / sd[ok]
        out[:, s] = np.clip(z, -8, 8)
    return out


def rolling_std(A, k):
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


def rolling_mean(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    cs = np.cumsum(np.nan_to_num(A), axis=0)
    for i in range(k - 1, T):
        s1 = cs[i] - (cs[i - k] if i >= k else 0)
        out[i] = s1 / k
    return out


def rolling_max(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    for i in range(k - 1, T):
        out[i] = np.nanmax(A[i - k + 1:i + 1], axis=0)
    return out


def rolling_min(A, k):
    T, S = A.shape
    out = np.full((T, S), np.nan)
    for i in range(k - 1, T):
        out[i] = np.nanmin(A[i - k + 1:i + 1], axis=0)
    return out


def build_one(name, C, V, TB, hour, dow):
    """单个候选特征 → (T,S)"""
    T, S = C.shape
    if name == 'lead5' or name == 'lead15':
        k = int(name[4:])
        M = mom(C, k)
        btc = np.repeat(M[:, [0]], S, axis=1)
        return exp_z_rows(btc - M)
    if name == 'beta240':
        R = np.zeros((T, S))
        R[1:] = np.log(np.maximum(C[1:], 1e-12) / np.maximum(C[:-1], 1e-12))
        R = np.nan_to_num(R)
        mkt = np.mean(R, axis=1, keepdims=True)
        out = np.full((T, S), np.nan)
        k = 240
        for i in range(k - 1, T):
            a = R[i - k + 1:i + 1]
            m = mkt[i - k + 1:i + 1]
            am = a - a.mean(axis=0)
            mm = m - m.mean()
            cov = (am * mm).sum(axis=0) / k
            sd_a = a.std(axis=0)
            sd_m = m.std()
            den = np.maximum(sd_a * sd_m, 1e-12)
            out[i] = np.clip(cov / den, -1, 1)
        return exp_z_rows(out)
    if name == 'streak':
        R = np.zeros((T, S))
        R[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
        sgn = np.sign(np.nan_to_num(R))
        out = np.zeros((T, S))
        for i in range(1, T):
            out[i] = np.where(sgn[i] == sgn[i - 1], out[i - 1] + sgn[i], sgn[i])
        return exp_z_rows(np.clip(out, -30, 30))
    if name == 'brkhi240':
        hi = rolling_max(C, 240)
        return exp_z_rows(C / np.maximum(hi, 1e-12) - 1)
    if name == 'brklo240':
        lo = rolling_min(C, 240)
        return exp_z_rows(C / np.maximum(lo, 1e-12) - 1)
    if name == 'volratio':
        R = np.zeros((T, S))
        R[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
        v20 = rolling_std(R, 20)
        v240 = rolling_std(R, 240)
        return exp_z_rows(np.log(np.maximum(v20, 1e-12) / np.maximum(v240, 1e-12)))
    if name == 'amtshare':
        q = C * V
        share = q / np.maximum(np.nansum(q, axis=1, keepdims=True), 1e-12)
        return exp_z_rows(share)
    if name == 'skewmkt':
        M = mom(C, 60)
        T_, S_ = M.shape
        sk = np.full(T_, np.nan)
        for i in range(480, T_):
            row = M[i]
            row = row[np.isfinite(row)]
            if len(row) > 20:
                m2 = row.var()
                if m2 > 1e-12:
                    sk[i] = ((row - row.mean()) ** 3).mean() / m2 ** 1.5
        return np.repeat(exp_z_rows(sk[:, None]), S, axis=1)
    if name == 'updnvol':
        R = np.zeros((T, S))
        R[1:] = C[1:] / np.maximum(C[:-1], 1e-12) - 1
        up = np.where(R > 0, V, 0.0)
        dn = np.where(R < 0, V, 0.0)
        k = 60
        u = rolling_mean(up, k)
        d = rolling_mean(dn, k)
        return exp_z_rows(np.log(np.maximum(u, 1e-12) / np.maximum(d, 1e-12)))
    if name == 'takim240':
        imb = TB / np.maximum(V, 1e-9) - 0.5
        return exp_z_rows(rolling_mean(imb, 240))
    if name == 'rangepos':
        hi = rolling_max(C, 240)
        lo = rolling_min(C, 240)
        return exp_z_rows((C - lo) / np.maximum(hi - lo, 1e-12) - 0.5)
    raise KeyError(name)


NAMES = ['lead5', 'lead15', 'beta240', 'streak', 'brkhi240', 'brklo240',
         'volratio', 'amtshare', 'skewmkt', 'updnvol', 'takim240', 'rangepos']


def delta_col(F, fwd, cut):
    """(T,S) 特征 → 每币一个差值（高20% − 低20%，分位用训练段）"""
    T, S = F.shape
    out = np.full(S, np.nan)
    for s in range(S):
        v = F[:, s]
        tr = v[:cut]
        ql, qh = np.nanquantile(tr, [0.2, 0.8])
        if not np.isfinite(ql) or not np.isfinite(qh) or abs(qh - ql) < 1e-9:
            continue
        vs = v[cut:]
        ys = fwd[cut:, s]
        ok = np.isfinite(vs) & np.isfinite(ys)
        hi = ok & (vs >= qh)
        lo = ok & (vs <= ql)
        if hi.sum() > 500 and lo.sum() > 500:
            out[s] = np.nanmean(ys[hi]) - np.nanmean(ys[lo])
    return out


def main():
    t0 = time.time()
    say('=' * 104)
    say('gambler_gate_v2 · 第二批牌面过闸（跨币结构 / 形态 / 流动性）')
    say(f'  {len(NAMES)} 个候选 | H={H} 分 | 打乱块=1天 | 币 {NSYM}')
    say('=' * 104)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume', 'taker_base'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv_c = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    piv_v = df.pivot_table(index='t', columns='symbol', values='volume').sort_index()[syms]
    piv_tb = df.pivot_table(index='t', columns='symbol', values='taker_base').sort_index()[syms]
    C = piv_c.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    V = np.nan_to_num(piv_v.values.astype(np.float64), nan=0.0)
    TB = np.nan_to_num(piv_tb.values.astype(np.float64), nan=0.0)
    T, S = C.shape
    hour = piv_c.index.hour.values
    dow = piv_c.index.dayofweek.values
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    V2 = V.copy()
    TB2 = TB.copy()
    say(f'  打乱面板完成  {time.time() - t0:.0f}s')

    cut = int(T * TRAIN)
    fwd = np.full((T, S), np.nan)
    ok = np.isfinite(C[H:]) & np.isfinite(C[:-H])
    v = np.full_like(C[H:], np.nan)
    v[ok] = C[H:][ok] / C[:-H][ok] - 1
    fwd[:-H] = v
    fwd2 = np.full((T, S), np.nan)
    ok2 = np.isfinite(C2[H:]) & np.isfinite(C2[:-H])
    v2 = np.full_like(C2[H:], np.nan)
    v2[ok2] = C2[H:][ok2] / C2[:-H][ok2] - 1
    fwd2[:-H] = v2

    rows = []
    say('')
    for nm in NAMES:
        try:
            FR = build_one(nm, C, V, TB, hour, dow)
            dR = delta_col(FR, fwd, cut)
            del FR
            FS = build_one(nm, C2, V2, TB2, hour, dow)
            dS = delta_col(FS, fwd2, cut)
            del FS
        except Exception as e:
            say(f'  {nm:<12} 计算失败: {e}')
            continue
        tr, ts, tg = tstat(dR), tstat(dS), tstat(dR - dS)
        rows.append(dict(feat=nm, d_real=np.nanmean(dR) * 100, d_shuf=np.nanmean(dS) * 100,
                         d_gap=np.nanmean(dR - dS) * 100, t_real=tr, t_shuf=ts, t_gap=tg))
        say(f'  {nm:<12} 真 {np.nanmean(dR) * 100:+.3f}%  打乱 {np.nanmean(dS) * 100:+.3f}%  '
            f't(真) {tr:+.2f}  t(差) {tg:+.2f}  {time.time() - t0:.0f}s')

    res = pd.DataFrame(rows)
    if len(res) == 0:
        say('  全部计算失败')
        return
    ceiling = float(np.nanmax(np.abs(res.t_shuf)))
    m = (res.t_real.abs() >= 2.5) & (res.t_real.abs() > ceiling + 0.5) & \
        (res.t_gap.abs() >= 2.5) & (np.sign(res.t_real) == np.sign(res.t_gap))

    say('')
    say('=' * 104)
    say(f'【一】噪声天花板 = {ceiling:.2f}')
    say(f'【二】过闸 {int(m.sum())} / {len(res)}')
    for _, r in res[m].iterrows():
        say(f'  ✅ {r.feat:<12} 真 {r.d_real:+.3f}%  打乱 {r.d_shuf:+.3f}%  '
            f't(真) {r.t_real:+.2f}  t(差) {r.t_gap:+.2f}')
    say('')
    say('【三】全部候选（按 |t(差)| 排序）')
    for _, r in res.reindex(res.t_gap.abs().sort_values(ascending=False).index).iterrows():
        tag = '✅' if m[res.index[res.feat == r.feat][0]] else '  '
        say(f'  {tag} {r.feat:<12} 真 {r.d_real:+.3f}%  打乱 {r.d_shuf:+.3f}%  '
            f't(真) {r.t_real:+.2f}  t(差) {r.t_gap:+.2f}')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_gate_v2.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    res.to_csv(os.path.join(OUT, 'gambler_gate_v2.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
