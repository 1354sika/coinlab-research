# -*- coding: utf-8 -*-
r"""
gambler_walkforward.py — 边测边学（赌徒的方式）

前几版的错：训练 70% / 测试 30% 硬切分
  ⇒ 测试段后半在训练里不存在 ⇒ 短窗口找不到邻居 ⇒ 段3/段4 零成交

本版：时间切成 30 块，逐块前进
  · 前 50% 时间只观察不交易（建立初始记忆）
  · 后 50% 每块：记忆 = 这块之前「最近 W 天」的数据（W 由波动决定 15~365 天）
  · 用这块记忆做 KNN，预测这块
  · 这块结束后，数据自动进入下一块的记忆池
  ⇒ 边测边学：永远只看「最近的牌谱」，但会随时间往前滚
"""
import sys, io, os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\walkforward.txt'
NSYM = int(os.environ.get('WF_NSYM', '60'))

COST    = 0.0009
H       = 176
K       = 50
SAFETY  = 1.5
STEP_TR = 20
STEP_TE = 50
DAY_MS  = 86_400_000
SIG_IDX = 8
W_BASE  = 30.0
W_MIN, W_MAX = 15.0, 365.0
NBLOCK  = 30
OBSERVE = 0.50        # 前 50% 只观察不交易

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
    for k in (5, 15, 60, 240, 1440):
        F.append(np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1.0]))
    vma = rmean(v, 60)
    F.append(v/np.where(vma > 0, vma, np.nan))
    F.append(np.log1p(v))
    hi = pd.Series(g['high'].values).rolling(240).max().values
    lo = pd.Series(g['low'].values).rolling(240).min().values
    rng = np.where(hi >= lo, hi-lo, np.nan)
    F.append(np.where(rng > 0, (c-lo)/rng, np.nan))
    F.append(pd.Series(c).pct_change().rolling(240).std().values)
    return np.column_stack(F)

def load(fn, nsym):
    df = pd.read_csv(fn, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(nsym).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    return df

def main():
    say('='*100)
    say('gambler_walkforward · 边测边学（滚动前进，记忆窗口由节奏决定）')
    say('='*100)
    say(f'BLOCK {NBLOCK} | 观察 {OBSERVE:.0%} | K={K} | 安全边际={SAFETY}× | 成本={COST*100:.3f}% | 窗口 {W_BASE:.0f}天 [{W_MIN:.0f},{W_MAX:.0f}]')
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
    mu = X[:cut0].mean(0); sd = X[:cut0].std(0) + 1e-9
    Xn = (X - mu) / sd
    sig_med = float(np.median(X[:cut0][:, SIG_IDX]))
    thr = COST * SAFETY
    say(f'  观察段结束于 {pd.to_datetime(T[cut0], unit="ms")}  波动中位 {sig_med*100:.3f}%')

    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    bet_r, bet_t, bet_w, bet_n, bet_i, bet_p = [], [], [], [], [], []
    wr = np.zeros(n)

    for i in range(NBLOCK):
        tr_end = edges[i]
        te_s, te_e = edges[i], edges[i+1]
        if te_e <= te_s: continue
        t_now = T[te_s]
        back = X[max(0,tr_end-200):tr_end, SIG_IDX]
        sig_now = float(np.median(back)) if len(back) else sig_med
        w_days = float(np.clip(W_BASE * sig_med / max(sig_now, 1e-10), W_MIN, W_MAX))
        age = t_now - T[:tr_end]
        m = (age <= w_days*DAY_MS) & (np.arange(tr_end) % STEP_TR == 0)
        sub = np.where(m)[0]
        if len(sub) < 100:
            say(f'    块 {i+1}/{NBLOCK}  记忆太少 ({len(sub)})  跳过'); continue

        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Xn[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Xn[te_idx])
        r = y[sub][nb]
        exp = r.mean(1)
        b = exp > thr
        wr[te_s:te_e] = w_days
        if b.sum() > 0:
            bet_r.append(y[te_idx][b])
            bet_w.append(np.full(int(b.sum()), w_days))
            bet_n.append(np.full(int(b.sum()), len(sub)))
            bet_i.append(te_idx[b])
        say(f'    块 {i+1:>2}/{NBLOCK}  窗 {w_days:5.1f}天  记忆 {len(sub):>5}  下注 {int(b.sum()):>4}/{len(te_idx):>4}  {time.time()-t0:.0f}s')

    if not bet_r:
        say('\n  ❌ 全程没有下注 ⇒ 这个口径下没有任何状态的期望超过 1.5×成本')
        return

    rA = np.concatenate(bet_r); wA = np.concatenate(bet_w); nA = np.concatenate(bet_n); iA = np.concatenate(bet_i)
    r_all = y[np.arange(cut0, n, STEP_TE)]

    say()
    say('='*100)
    say('【一】总体（后 50% 时间，边走边学）')
    say('='*100)
    def summ(r, name):
        if len(r) < 10: say(f'  {name:<30} 样本太少'); return None
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  {name:<30} {len(r):>6,} 笔  单笔 {r.mean()*100:+7.3f}%  中位 {np.median(r)*100:+7.3f}%  胜率 {(r>0).mean()*100:4.1f}%  t={t:+6.2f}')
        return r.mean()

    a0 = summ(r_all, 'A 全样（随机下注）')
    a1 = summ(rA,    'C 边测边学（模型选）')
    say(f'  下注率 {len(rA)/len(r_all)*100:.1f}%')
    if a1 is not None and a0 is not None:
        say(f'  净优势 {(a1-a0)*100:+.3f}%/笔')

    say()
    say('='*100)
    say('【二】模型自己看多久（窗口天数）')
    say('='*100)
    for q in (10,25,50,75,90):
        say(f'  P{q}:{np.percentile(wA,q):.1f}天', end='') if False else None
    say('  ' + '   '.join(f'P{q}:{np.percentile(wA,q):.1f}天' for q in (10,25,50,75,90)))
    say(f'  记忆量 中位 {np.median(nA):.0f} 个点')

    say()
    say('='*100)
    say('【三】四段（按交易发生的时间）')
    say('='*100)
    oq = np.argsort(iA); rS = rA[oq]; iS = iA[oq]
    n4 = len(rS); ok4 = True
    for k in range(4):
        aa = k*n4//4; bb = (k+1)*n4//4
        if bb - aa < 10: say(f'  段{k+1} 样本太少'); ok4 = False; continue
        r = rS[aa:bb]
        t = r.mean()/(r.std()/np.sqrt(len(r))+1e-12)
        say(f'  段{k+1}  n={len(r):>5}  单笔 {r.mean()*100:+7.3f}%  t={t:+6.2f}')
        if r.mean() <= 0: ok4 = False
    say(f'  {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

    say()
    say('='*100)
    say('【四】判据')
    say('='*100)
    if a1 is not None and a0 is not None:
        say(f'  边测边学 {a1*100:+.3f}%  vs  全样 {a0*100:+.3f}%')
    say(f'  四段 {"同正 ✅" if ok4 else "非全正 ❌"}')
    if a1 is not None and a0 is not None and a1 > a0 and ok4:
        say('  ✅ 边测边学 + 自适应窗口这条路成立')
    elif a1 is not None and a1 > 0:
        say('  ⚠ 正期望，但四段不稳')
    else:
        say('  ❌ 仍未成立')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('gambler_walkforward 完成\n')

if __name__ == '__main__':
    main()
