# -*- coding: utf-8 -*-
r"""
gambler_k_adaptive.py — 模型决定「什么算极端」（不是决定「下不下注」）

八种入场都失败，共同点：它们都在让模型决定「现在下不下注」
  → 这是不可学的（AUC≈0.5，市场每根都差不多）

但 v4 的「跌 5%」有效，说明：**可学的是「什么算极端」**
  · 极端 ≠ 预测；极端是「跌得比平时多几倍波动」
  · k（跌几倍波动）由模型自己选：每 30 天用「最近这个 k 的表现」重新挑
  · 触发后：凯利定仓 + 连亏停牌 + 模型退出（v4 骨架）

这才是「入场由模型决定」的正确形态：
  模型不预测未来，模型调「我出手的门槛」——
  市场安静就把门槛降低（跌 2σ 就出手），市场狂躁就把门槛提高（跌 4σ 才出手）。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\k_adaptive.txt'
NSYM = int(os.environ.get('KA_NSYM', '60'))

COST      = 0.0009
H         = 176
STEP_TE   = 50
OBSERVE   = 0.50
BAR_MS    = 300_000
DAY_MS    = 86_400_000

K_SET     = [1.5, 2.0, 2.5, 3.0, 4.0]
REBUILD_MS= 30*DAY_MS
POS_MAX   = 0.02
KELLY     = 0.25
STOP_N    = 8
STOP_LOSS = 0.004
STOP_BARS = 50

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def build(g):
    c = g['close'].values.astype(np.float64)
    r5 = np.concatenate([np.full(5, np.nan), c[5:]/c[:-5]-1.0])
    ret = pd.Series(c).pct_change()
    sig5 = (ret.rolling(20).std()*np.sqrt(5)).values     # 5 根窗口的典型跌幅尺度
    return np.column_stack([r5, sig5])

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def main():
    say('='*100)
    say('gambler_k_adaptive · 模型自己决定「跌几倍波动才出手」')
    say('='*100)
    say(f'k ∈ {K_SET} | 每 30 天重挑一次 | 凯利×{KELLY} 上限{POS_MAX*100:.0f}% | 停牌 {STOP_N}笔/{STOP_LOSS*100:.1f}%')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    Xs, ys, ts = [], [], []
    for s, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        F = build(g)
        c = g['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
        Xs.append(F[ok]); ys.append(fwd[ok].astype(np.float64)); ts.append(g['open_time'].values[ok])
    X = np.concatenate(Xs); y = np.concatenate(ys); T = np.concatenate(ts)
    o = np.argsort(T, kind='stable'); X = X[o]; y = y[o]; T = T[o]
    say(f'  样本 {len(X):,}  {time.time()-t0:.0f}s')

    n = len(X)
    cut0 = int(n*OBSERVE)
    say(f'  观察段结束 {pd.to_datetime(T[cut0], unit="ms")}')

    te = np.arange(cut0, n, STEP_TE)
    r5   = X[te, 0]
    s5   = X[te, 1]
    tt   = T[te]
    rets = y[te]

    # 每块挑 k：用「上一个块里该 k 的已结算表现」
    k_now = 3.0
    prev = {k: [] for k in K_SET}
    next_rebuild = tt[0] + REBUILD_MS

    pos = np.zeros(len(te))
    pause_left = 0
    g_hist = deque(maxlen=40)
    settle = deque()
    n_bet = 0; n_pause = 0
    k_track = []

    for i in range(len(te)):
        t = tt[i]
        # 结算（延迟 H 根，绝不前视）
        while settle and settle[0][0] <= t:
            _, k_, r_ = settle.popleft()
            prev[k_].append(r_)

        # 到点重挑 k
        if t >= next_rebuild:
            best_k, best_v = k_now, -1e9
            for k in K_SET:
                a = prev[k]
                if len(a) >= 5:
                    v = float(np.mean(a))
                    if v > best_v: best_v, best_k = v, k
            k_now = best_k
            next_rebuild = t + REBUILD_MS
        k_track.append(k_now)

        if pause_left > 0:
            pause_left -= 1; n_pause += 1
            pos[i] = 0.0
            continue

        trig = r5[i] < -k_now * s5[i]
        if not trig:
            pos[i] = 0.0
            continue

        # 观察账：无论下不下注，都记录「这次触发后的真实结果」（记账，不是下注）
        settle.append((t + H*BAR_MS, k_now, rets[i]))

        # 仓位：凯利（用该 k 的历史观察估）
        a = np.array(prev[k_now]) if prev[k_now] else np.array([0.0])
        e = a.mean() if len(a) >= 3 else 0.0
        v = a.var()+1e-10 if len(a) >= 3 else 1e-10
        if e <= 0:
            pos[i] = 0.0
            continue
        p = min(e/v*KELLY, POS_MAX)
        pos[i] = p
        n_bet += 1
        g_hist.append(p*rets[i])
        # 停牌检查
        if len(g_hist) >= STOP_N and sum(list(g_hist)[-STOP_N:]) < -STOP_LOSS:
            pause_left = STOP_BARS
            n_pause += 1

    pnl = pos*rets
    cap = np.cumprod(1.0+pnl)
    dd = (cap/np.maximum.accumulate(cap)-1.0).min()

    say()
    say('='*100)
    say('【一】资金曲线（后 50% 时间）')
    say('='*100)
    say(f'  期末资金 {cap[-1]:.4f}   最大回撤 {dd*100:.2f}%')
    say(f'  下注 {n_bet} 次  停牌 {n_pause} 次  下注率 {n_bet/len(te)*100:.2f}%')
    say(f'  仓位加权单笔 {(pos*rets).sum()/max(pos.sum(),1e-9)*100:+.3f}%')

    say()
    say('='*100)
    say('【二】模型自己选了哪个门槛（k 分布）')
    say('='*100)
    kk = np.array(k_track)
    for k in K_SET:
        say(f'  k={k:<4} {(kk==k).mean()*100:5.1f}%', end='') if False else None
    say('  ' + '   '.join(f'k={k}:{(kk==k).mean()*100:.1f}%' for k in K_SET))
    say(f'  k 换过 {int((np.diff(kk)!=0).sum())} 次')

    say()
    say('='*100)
    say('【三】四段资金')
    say('='*100)
    n4 = len(pos); ok4 = True
    for i in range(4):
        aa = i*n4//4; bb = (i+1)*n4//4
        seg = pnl[aa:bb]
        c_seg = np.cumprod(1.0+seg)[-1] if len(seg) else 1.0
        nk = np.median(kk[aa:bb])
        say(f'  段{i+1}  下注率 {(pos[aa:bb]>0).mean()*100:5.2f}%  中位k {nk:.1f}  段内资金 {c_seg:.4f}')
        if c_seg <= 1.0: ok4 = False
    say(f'  {"✅ 四段都赚" if ok4 else "❌ 有段亏损"}')

    say()
    say('='*100)
    say('【四】判据（用户口径：存活 + 持续盈利）')
    say('='*100)
    ok = cap[-1] > 1.0 and dd > -0.30
    say(f'  期末 {cap[-1]:.4f} > 1 ? {"✅" if cap[-1]>1 else "❌"}    回撤 {dd*100:.2f}% > -30% ? {"✅" if dd>-0.30 else "❌"}')
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}；四段 {"✅" if ok4 else "❌"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'期末 {cap[-1]:.4f} 回撤 {dd:.4f}\n')

if __name__ == '__main__':
    main()
