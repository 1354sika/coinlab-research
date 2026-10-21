# -*- coding: utf-8 -*-
r"""
gambler_sizing.py — 仓位网格：能不能既保回撤，又吃到暴涨

full2 的 0.9975 只有 0.56% 平均仓位（凯利 0.25 × 上限 2%）。
诊断出的病根：子弹锁得太死 —— 熊市不亏（✅），暴涨也吃不到（❌）。

这次测 6 个配置：
  凯利系数 0.25 / 0.5 / 1.0  ×  仓位上限 2% / 5% / 10%

真实约束不变：每币独立子账户（1/N）、同币不重叠、连亏停牌。
KNN 只建一次，6 个配置复用（省时间）。

判据（用户口径）：
  · 总资产 > 1
  · 最大回撤 < 30%
  · 四段都赚
  · 相对等权买入持有基准（1.2845）的超额
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\sizing.txt'
NSYM = int(os.environ.get('SZ_NSYM', '60'))

COST      = 0.0009
H         = 176
STEP_TR   = 100
STEP_TE   = 50
OBSERVE   = 0.50
NBLOCK    = 30
K         = 5
STOP_N    = 8
STOP_LOSS = 0.004
STOP_BARS = 50
H_STEPS   = int(np.ceil(H/STEP_TE))
BENCH     = 1.2845

CONFIGS = [
    ('K0.25 / 2%  (基线)', 0.25, 0.02),
    ('K0.50 / 2%',         0.50, 0.02),
    ('K0.50 / 5%',         0.50, 0.05),
    ('K1.00 / 5%',         1.00, 0.05),
    ('K1.00 / 10%',        1.00, 0.10),
    ('K0.25 / 5%',         0.25, 0.05),
]

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
    F = []
    for k in (5, 15, 60):
        F.append(np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1.0]))
    vma = rmean(v, 60)
    F.append(v/np.where(vma > 0, vma, np.nan))
    F.append(pd.Series(c).pct_change().rolling(240).std().values)
    return np.column_stack(F)

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def collect(g):
    """一次跑完 KNN，返回测试段每个点的 (exp, var, R)"""
    F = build(g); c = g['close'].values
    fwd = np.full(len(c), np.nan)
    fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
    ok = np.isfinite(F).all(axis=1) & np.isfinite(fwd) & (np.arange(len(c)) < len(c)-H)
    X, y = F[ok], fwd[ok]
    n = len(X)
    if n < 2000: return None
    cut0 = int(n*OBSERVE)
    if cut0 < 300: return None
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0)+1e-9
    Xn = (X-mu)/sd
    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    e_l, v_l, r_l = [], [], []
    for i in range(NBLOCK):
        tr_end, te_s, te_e = edges[i], edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        e_l.append(r.mean(1)); v_l.append(r.var(1)+1e-10); r_l.append(y[te_idx])
    if not e_l: return None
    return np.concatenate(e_l), np.concatenate(v_l), np.concatenate(r_l)

def simulate(exp, var, R, kelly, posmax):
    """不重叠 + 停牌 + 复利"""
    raw = np.where(exp > 0, np.clip(exp/np.maximum(var, 1e-10)*kelly, 0, posmax), 0.0)
    pause = 0; gh = deque(maxlen=40)
    P = np.zeros_like(raw)
    nxt = 0; n_trades = 0
    for k in range(len(raw)):
        if k < nxt: continue
        if pause > 0:
            pause -= 1; nxt = k+1; continue
        if raw[k] <= 0: continue
        P[k] = raw[k]; nxt = k + H_STEPS; n_trades += 1
        gh.append(raw[k]*R[k])
        if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
            pause = STOP_BARS
    pnl = P*R
    cap = np.cumprod(1.0+pnl)
    dd = (cap/np.maximum.accumulate(cap)-1.0).min()
    n4 = len(pnl); segs = []
    for i in range(4):
        aa = i*n4//4; bb = (i+1)*n4//4
        segs.append(np.cumprod(1.0+pnl[aa:bb])[-1] if bb > aa else 1.0)
    return cap[-1], dd, segs, n_trades, P.mean()

def main():
    say('='*100)
    say('gambler_sizing · 仓位网格（真实约束不变）')
    say('='*100)
    say(f'K={K} | 不重叠 {H_STEPS} 步 | 停牌 {STOP_N}笔/{STOP_LOSS*100:.1f}% | 基准(等权买入持有) {BENCH:.4f}')
    say('='*100)

    df = load(DATA, NSYM)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    data = []
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        r = collect(g)
        if r is None: continue
        data.append(r)
    say(f'  子账户 {len(data)} 个  {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【一】6 个配置（等权组合，真实约束）')
    say('='*100)
    say(f'  {"配置":<18} {"期末":>9} {"收益%":>9} {"回撤%":>8} {"均仓%":>7} {"笔数":>7}  四段')
    rows = []
    for name, kelly, posmax in CONFIGS:
        finals, dds, segs_sum, trades, pm = [], [], np.zeros(4), 0, []
        for exp, var, R in data:
            c, d, s, nt, p = simulate(exp, var, R, kelly, posmax)
            finals.append(c); dds.append(d); segs_sum += np.array(s); trades += nt; pm.append(p)
        total = float(np.mean(finals))
        dd = float(np.mean(dds))
        seg_avg = segs_sum/len(data)
        ok4 = bool(all(v > 1.0 for v in seg_avg))
        tag = "✅" if ok4 else "❌"
        say(f'  {name:<18} {total:>9.4f} {(total-1)*100:>+8.2f}% {dd*100:>7.2f}% {np.mean(pm)*100:>6.2f}% {trades:>7}  {tag} [{seg_avg[0]:.3f} {seg_avg[1]:.3f} {seg_avg[2]:.3f} {seg_avg[3]:.3f}]')
        rows.append((name, total, dd, ok4, trades, np.mean(pm)))

    say()
    say('='*100)
    say('【二】判据汇总')
    say('='*100)
    best = None
    for name, total, dd, ok4, trades, pm in rows:
        good = total > 1.0 and dd > -0.30 and ok4
        if good and (best is None or total > best[1]):
            best = (name, total, dd)
        say(f'  {name:<18} 总资产{total:.4f}{"✅" if total>1 else "❌"}  回撤{dd*100:.1f}%{"✅" if dd>-0.30 else "❌"}  四段{"✅" if ok4 else "❌"}')
    say()
    if best:
        say(f'  ✅ 同时满足三条的最好配置：{best[0]}  期末 {best[1]:.4f}  回撤 {best[2]*100:.2f}%')
        say(f'     相对等权基准({BENCH:.4f}) {"跑赢" if best[1]>BENCH else "跑输"} {abs(best[1]-BENCH)*100:.2f}pp')
    else:
        say('  ❌ 没有配置同时满足三条')
        # 给最接近的
        cand = sorted(rows, key=lambda r: (r[1] - 1.0) + r[2], reverse=True)[0]
        say(f'     最接近：{cand[0]}  期末 {cand[1]:.4f}  回撤 {cand[2]*100:.2f}%')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for name, total, dd, ok4, trades, pm in rows:
            f.write(f'{name}\t{total:.4f}\t{dd:.4f}\t{ok4}\n')

if __name__ == '__main__':
    main()
