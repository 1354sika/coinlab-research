# -*- coding: utf-8 -*-
r"""
gambler_ledger_stop.py — 账本定仓位 + 全局停牌 + 资金曲线

上一步（ledger_entry）的发现：
  账本确实在学（下注率 30%→52%，加权单笔 -0.599%→-0.102%）
  但四段仍全负 —— 单笔均值救不回来

而 v4 的实证是：**单笔均值可以负，但资金曲线可以涨**
  靠的是「连亏 5 次停 50 笔」—— 躲过亏损期

所以本版补上最后一块：**全局停牌**
  1. 账本定每格的仓位（已有）
  2. **全局账本**：最近 30 笔的「实际盈亏」（仓位加权）
  3. **停牌**：最近 10 笔累计亏 > 0.5% ⇒ 暂停 50 笔
  4. **资金曲线**：capital *= (1 + 仓位 × 单笔收益)

判据换成用户口径：期末资金 > 1、最大回撤 < 30%、四段资金都为正。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\ledger_stop.txt'
NSYM = int(os.environ.get('LS_NSYM', '60'))

COST     = 0.0009
H        = 176
STEP_TE  = 50
OBSERVE  = 0.50
NB       = 5
NCELL    = NB**3
K_HIST   = 50
MIN_HIST = 10
POS_MAX  = 0.02
KELLY    = 0.25
BAR_MS   = 300_000

G_HIST   = 30
STOP_N   = 10
STOP_LOSS= 0.005
STOP_BARS= 50

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def build(g):
    c = g['close'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    r60 = np.concatenate([np.full(60, np.nan), c[60:]/c[:-60]-1.0])
    vma = rmean(v, 60)
    volr = v/np.where(vma > 0, vma, np.nan)
    sig = pd.Series(c).pct_change().rolling(240).std().values
    return np.column_stack([r60, volr, sig])

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def main():
    say('='*100)
    say('gambler_ledger_stop · 账本定仓位 + 全局停牌 + 资金曲线')
    say('='*100)
    say(f'状态 {NB}³ | 每格 {K_HIST} 笔 | 停牌: 最近{STOP_N}笔累计亏>{STOP_LOSS*100:.1f}% → 停{STOP_BARS}笔')
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

    qs = []
    for j in range(3):
        q = np.percentile(X[:cut0, j], np.linspace(0, 100, NB+1))
        q[0], q[-1] = -np.inf, np.inf
        qs.append(q)

    te_all = np.arange(cut0, n, STEP_TE)
    cell_all = np.zeros(len(te_all), dtype=np.int32)
    for j in range(3):
        b = np.clip(np.searchsorted(qs[j], X[te_all, j], side='right')-1, 0, NB-1)
        cell_all = cell_all*NB + b

    ledger = [deque(maxlen=K_HIST) for _ in range(NCELL)]
    g_hist = deque(maxlen=G_HIST)
    settle = deque()
    pos_out = np.zeros(len(te_all))
    pause_left = 0
    n_pause = 0
    n_bet = 0

    for k in range(len(te_all)):
        i = te_all[k]; t = T[i]
        # 结算（延迟 H 根，无前视）
        while settle and settle[0][0] <= t:
            _, c_, r_, ps_ = settle.popleft()
            ledger[c_].append(r_)
            g_hist.append(ps_ * r_)
        # 决策
        c = cell_all[k]
        pos = 0.0
        if pause_left > 0:
            pause_left -= 1; n_pause += 1
        else:
            hist = ledger[c]
            if len(hist) >= MIN_HIST:
                a = np.fromiter(hist, dtype=np.float64, count=len(hist))
                e = a.mean(); v = a.var() + 1e-10
                if e > 0:
                    pos = min(e/v*KELLY, POS_MAX)
            # 全局停牌检查
            if len(g_hist) >= STOP_N:
                recent = list(g_hist)[-STOP_N:]
                if sum(recent) < -STOP_LOSS:
                    pause_left = STOP_BARS
                    pos = 0.0
                    n_pause += 1
        if pos > 0: n_bet += 1
        pos_out[k] = pos
        settle.append((t + H*BAR_MS, c, y[i], pos))

    R = y[te_all]; P = pos_out
    pnl = P * R
    cap = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(cap)
    dd = (cap/peak - 1.0).min()

    say()
    say('='*100)
    say('【一】资金曲线（后 50% 时间）')
    say('='*100)
    say(f'  期末资金 {cap[-1]:.4f}   最大回撤 {dd*100:.2f}%')
    say(f'  下注 {n_bet} 次 / 停牌 {n_pause} 次 / 总点 {len(P):,}')
    say(f'  仓位均值 {P.mean()*100:.3f}%   仓位加权单笔 {(P*R).sum()/max(P.sum(),1e-9)*100:+.3f}%')

    # 满仓对照
    cap1 = np.cumprod(1.0 + R)
    dd1 = (cap1/np.maximum.accumulate(cap1) - 1.0).min()
    say(f'  A 每根满仓对照: 期末 {cap1[-1]:.6f}  回撤 {dd1*100:.2f}%')

    say()
    say('='*100)
    say('【二】四段资金')
    say('='*100)
    n4 = len(P); ok4 = True
    for k in range(4):
        aa = k*n4//4; bb = (k+1)*n4//4
        seg = pnl[aa:bb]
        c_seg = np.cumprod(1.0 + seg)[-1] if len(seg) else 1.0
        say(f'  段{k+1}  下注率 {(P[aa:bb]>0).mean()*100:5.1f}%  段内资金 {c_seg:.4f}')
        if c_seg <= 1.0: ok4 = False
    say(f'  {"✅ 四段都赚" if ok4 else "❌ 有段亏损"}')

    say()
    say('='*100)
    say('【三】账本学到了什么')
    say('='*100)
    nz = [(c_, np.array(ledger[c_])) for c_ in range(NCELL) if len(ledger[c_]) >= MIN_HIST]
    say(f'  有经验格子 {len(nz)}/{NCELL}')
    if nz:
        exps = np.array([v.mean() for _, v in nz])
        say(f'  各格期望 P50 {np.percentile(exps,50)*100:+.3f}%   >0 的 {int((exps>0).sum())}/{len(exps)}   >成本线的 {int((exps>COST).sum())}')

    say()
    say('='*100)
    say('【四】判据（用户口径）')
    say('='*100)
    say('  · 期末资金 > 1 且最大回撤 < 30% ⇒ 活着且在赚')
    say('  · 四段资金都 > 1 ⇒ 是"持续"，不是靠某一段')
    ok = cap[-1] > 1.0 and dd > -0.30
    say(f'  {"✅ 活着且在赚" if ok else "❌ 未达标"}；四段 {"✅" if ok4 else "❌"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(f'期末 {cap[-1]:.4f} 回撤 {dd:.4f}\n')

if __name__ == '__main__':
    main()
