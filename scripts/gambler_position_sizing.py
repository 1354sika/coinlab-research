# -*- coding: utf-8 -*-
r"""
gambler_position_sizing.py — 按赔率定仓位（不是"选/不选"）

前几版的错：入场是二值（下注 / 不下注）
  · KNN 选太多（29.7%，平均状态无 alpha）→ -0.199%
  · 跌5% 选太少（0.02%，靠稀有）→ +0.669%

赌徒的做法：**每个时刻都评估，仓位按赔率定**
  · 期望负 → 仓位 0（自然弃牌）
  · 期望正 → 仓位 ∝ 期望/方差（凯利）
  · 期望高 → 仓位大
  ⇒ 下注率可以高，但总风险由仓位控制

状态：r60（跌幅）分位 × sig（波动）分位 = 20×5 = 100 桶
  这是「行为状态」的二维表示（恐慌程度 × 市场紧张度）
  每桶算出「期望 / 方差」，仓位 = 期望/方差 × 0.25，上限 2%

边测边学：30 块滚动，每块用「本块之前最近 W 天」重算分位表和赔率表。
"""
import sys, io, os, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\position_sizing.txt'
NSYM = int(os.environ.get('PS_NSYM', '60'))

COST    = 0.0009
H       = 176
STEP_TE = 50
DAY_MS  = 86_400_000
W_BASE  = 30.0
W_MIN, W_MAX = 15.0, 365.0
NBLOCK  = 30
OBSERVE = 0.50
NB_R, NB_S = 20, 5          # 100 桶
POS_MAX = 0.02
KELLY   = 0.25
MIN_CELL = 20

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def build(g):
    """只有 3 个量：r60（跌幅）、sig（波动）、volr（量比）"""
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
    say('gambler_position_sizing · 按赔率定仓位（不是选/不选）')
    say('='*100)
    say(f'状态 = r60分位({NB_R}) × sig分位({NB_S}) = {NB_R*NB_S}桶 | 凯利×{KELLY} 上限{POS_MAX*100:.0f}% | 成本{COST*100:.3f}% | 持{H}根')
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
    sig_med = float(np.median(X[:cut0][:, 2]))
    say(f'  观察段结束 {pd.to_datetime(T[cut0], unit="ms")}  波动中位 {sig_med*100:.3f}%')

    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    all_pos, all_ret, all_idx = [], [], []

    for i in range(NBLOCK):
        tr_end = edges[i]; te_s, te_e = edges[i], edges[i+1]
        if te_e <= te_s: continue
        t_now = T[te_s]
        back = X[max(0,tr_end-200):tr_end, 2]
        sig_now = float(np.median(back)) if len(back) else sig_med
        w_days = float(np.clip(W_BASE * sig_med / max(sig_now,1e-10), W_MIN, W_MAX))

        age = t_now - T[:tr_end]
        m = age <= w_days*DAY_MS
        sub = np.where(m)[0]
        if len(sub) < 2000:
            say(f'    块 {i+1}/{NBLOCK} 记忆太少 ({len(sub)}) 跳过'); continue

        # 分位点（只用这块记忆）
        q_r = np.percentile(X[sub, 0], np.linspace(0, 100, NB_R+1))
        q_s = np.percentile(X[sub, 2], np.linspace(0, 100, NB_S+1))
        q_r[0], q_r[-1] = -np.inf, np.inf
        q_s[0], q_s[-1] = -np.inf, np.inf

        def bidx(arr):
            br = np.clip(np.searchsorted(q_r, arr[:,0], side='right')-1, 0, NB_R-1)
            bs = np.clip(np.searchsorted(q_s, arr[:,2], side='right')-1, 0, NB_S-1)
            return br*NB_S + bs

        b_sub = bidx(X[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        b_te = bidx(X[te_idx])

        exp_t = np.full(NB_R*NB_S, np.nan)
        var_t = np.full(NB_R*NB_S, np.nan)
        cnt_t = np.zeros(NB_R*NB_S, dtype=int)
        for b in range(NB_R*NB_S):
            mm = b_sub == b
            c_ = int(mm.sum())
            cnt_t[b] = c_
            if c_ < MIN_CELL: continue
            exp_t[b] = y[sub][mm].mean()
            var_t[b] = y[sub][mm].var() + 1e-12

        e = exp_t[b_te]; v = var_t[b_te]
        pos = np.where(np.isfinite(e) & (e > 0), np.clip(e/np.maximum(v,1e-10)*KELLY, 0, POS_MAX), 0.0)

        all_pos.append(pos)
        all_ret.append(y[te_idx])
        all_idx.append(te_idx)

        ok_pos = pos > 0
        say(f'    块 {i+1:>2}/{NBLOCK} 窗{w_days:5.1f}天 记忆{len(sub):>6} 有效桶{int((cnt_t>=MIN_CELL).sum()):>3}/{NB_R*NB_S} 下注率{ok_pos.mean()*100:5.1f}% 均仓{pos.mean()*100:5.2f}% {time.time()-t0:.0f}s')

    P = np.concatenate(all_pos); R = np.concatenate(all_ret); I = np.concatenate(all_idx)

    say()
    say('='*100)
    say('【一】结果（后 50% 时间，边测边学，按赔率定仓位）')
    say('='*100)
    m1 = R.mean()
    t1 = m1/(R.std()/np.sqrt(len(R))+1e-12)
    say(f'  A 每根都满仓（基准）        {len(R):>7,} 笔  单笔 {m1*100:+7.3f}%  t={t1:+6.2f}')

    w = P.sum()
    if w > 0:
        wavg = (P*R).sum()/w
        # 每单位仓位的收益（加权）
        tt = wavg/((R**2*P).sum()/w - wavg**2 + 1e-12)**0.5 * np.sqrt(len(R))
        say(f'  B 按赔率定仓位             仓位均值 {P.mean()*100:.2f}%  下注率 {(P>0).mean()*100:.1f}%')
        say(f'     仓位加权单笔 {wavg*100:+7.3f}%   等效总收益（Σ仓位×收益） {(P*R).sum()*100:+.2f}%')
        say(f'     对比满仓 {(m1*100):+.3f}%  ⇒  按仓位 {(wavg*100):+.3f}%  （差 {(wavg-m1)*100:+.3f}%/单位仓位）')
    else:
        say('  B 全程仓位 0 ⇒ 没有任何桶的期望为正')

    say()
    say('='*100)
    say('【二】仓位分布')
    say('='*100)
    pos_only = P[P > 0]
    if len(pos_only) > 0:
        say('  ' + '   '.join(f'P{q}:{np.percentile(pos_only,q)*100:.2f}%' for q in (10,25,50,75,90)))
        say(f'  仓位 0: {(P==0).mean()*100:.1f}%   0~0.5%: {((P>0)&(P<0.005)).mean()*100:.1f}%   0.5~1%: {((P>=0.005)&(P<0.01)).mean()*100:.1f}%   1~2%: {(P>=0.01).mean()*100:.1f}%')

    say()
    say('='*100)
    say('【三】四段（按仓位加权）')
    say('='*100)
    oq = np.argsort(I); Ps = P[oq]; Rs = R[oq]
    n4 = len(Ps); ok4 = True
    for k in range(4):
        aa = k*n4//4; bb = (k+1)*n4//4
        pp = Ps[aa:bb]; rr = Rs[aa:bb]
        ww = pp.sum()
        if ww <= 0: say(f'  段{k+1} 无仓位'); ok4 = False; continue
        av = (pp*rr).sum()/ww
        say(f'  段{k+1}  n={len(pp):>6,}  仓位均值 {pp.mean()*100:5.2f}%  加权单笔 {av*100:+7.3f}%')
        if av <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    say('  · 若「按赔率定仓位」的加权单笔 > 满仓单笔 ⇒ 定仓位这件事本身有价值')
    say('  · 若加权单笔 > 0 且四段同正 ⇒ 赌徒式入场成立（不是靠选少，是靠押对大小）')
    if ok4 and w > 0 and wavg > 0:
        say('  ✅ 成立')
    elif w > 0 and wavg > 0:
        say('  ⚠ 正期望但四段不稳')
    else:
        say('  ❌ 仍未成立')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_position_sizing 完成\n')

if __name__ == '__main__':
    main()
