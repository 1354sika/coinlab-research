# gambler_unified2.py —— 开仓价值 = 开仓 + 退出模型跑完（不是固定持有 H）
# 上一版退化成"永不开仓"，因为无仓账本用的是"持 H 的期望"（负的）。
# 这一版：无仓账本记"开仓 → 由退出模型决定何时收 → 净收益"，让退出救场算进去。
# 状态 = (特征簇 8 × 市场档 3) = 24 格；只在 E > 0 的格子开仓
import sys, io, os, time, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(ROOT, 'evidence')
NSYM = int(os.environ.get('U2_NSYM', '60'))
BAR = 5; FEE = 0.0002; H = 176; MINHOLD = 12
G_LVL = 5; T_LVL = 5; M_LVL = 3
NCLUST = 8
CAP = 0.10
def say(*a, **kw): print(*a, flush=kw.get('flush', True))
def zs(A, win):
    df = pd.DataFrame(A); m = df.rolling(win, min_periods=win).mean(); s = df.rolling(win, min_periods=win).std()
    return ((df - m) / (s + 1e-10)).fillna(0).values

def main():
    t0 = time.time()
    use = 'symbol'
    with open(os.path.join(DATA, 'klines_1m_1y.csv'), 'r', encoding='utf-8') as f:
        r = pd.read_csv(f, nrows=1)
        use = 'symbol' if 'symbol' in r.columns else 'Symbol'
    df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'),
                     usecols=[use, 'open_time','open','high','low','close','volume'],
                     dtype={use: 'str', 'open_time': 'int64', 'open':'float64','high':'float64','low':'float64','close':'float64','volume':'float64'})
    df.columns = ['sym','t','o','h','l','c','v']
    df = df.sort_values(['sym','t']).reset_index(drop=True)
    df['t'] = df['t'] // (60000 * BAR)
    g = df.groupby('sym', sort=False)
    syms = list(g.groups)[:NSYM]
    gd = {s: g.get_group(s) for s in syms}
    nmin = min((len(gd[s]) // BAR) * BAR for s in syms)
    def agg(s, col, how):
        a = gd[s][col].values[:nmin].reshape(-1, BAR)
        return {'mean': a.mean(axis=1), 'first': a[:, 0], 'max': a.max(axis=1), 'min': a.min(axis=1)}[how]
    C  = np.stack([agg(s, 'c', 'mean')  for s in syms])
    O  = np.stack([agg(s, 'o', 'first') for s in syms])
    Hh = np.stack([agg(s, 'h', 'max')   for s in syms])
    L  = np.stack([agg(s, 'l', 'min')   for s in syms])
    K = C.shape[1]
    del df; gc.collect()
    say(f'  数据 {len(syms)} 币  {K} 根  [{time.time()-t0:.0f}s]', flush=True)
    R1 = np.zeros_like(C); R1[:, 1:] = C[:, 1:] / (C[:, :-1] + 1e-12) - 1.0
    RV = np.abs(R1) * 1e4
    s = pd.DataFrame(R1); F = {}
    for w in [48, 288, 1152]: F[f'r{w}'] = s.rolling(w, min_periods=w).sum().fillna(0).values
    F['vol'] = s.rolling(48, min_periods=48).std().fillna(0).values
    Cr = C[:, None, :]; F['rp'] = ((Cr - L[:, None, :]) / (Hh - L + 1e-12)[:, None, :])[:, 0, :]
    win = 1440
    for k in ['r48','r288','r1152','vol']: F[k] = zs(F[k], win)
    F['sgn'] = zs(pd.DataFrame(np.sign(C - O)).rolling(96, min_periods=96).mean().fillna(0).values, win)
    dev = np.sign(C - O) * np.abs(C - O) / (C + 1e-12) * 1e4
    F['dev'] = zs(pd.DataFrame(dev).rolling(12, min_periods=12).mean().fillna(0).values, win)
    F['vabs'] = zs(pd.DataFrame(pd.DataFrame(RV).rolling(12, min_periods=12).mean().fillna(0).values)
                    .rolling(250, min_periods=250).mean().fillna(0).values, win)
    F['p250'] = pd.DataFrame(C).rolling(250, min_periods=250).max().fillna(0).values
    F['lp'] = zs(np.log(C / (F['p250'] + 1e-12)), win)
    say(f'  基元 ok  [{time.time()-t0:.0f}s]', flush=True)
    cut = int(K * 0.7)
    # ---- 1) 训练退出账本（与 exit_alpha 同口径）----
    G_t = np.zeros((G_LVL, T_LVL)); G_s = np.zeros((G_LVL, T_LVL))
    tr_idx = np.arange(0, cut, 12)
    tr_idx = tr_idx[(tr_idx % K) + H < K]
    for i in tr_idx:
        si = i // K; k0 = i % K
        for hh in range(1, 13):
            p = C[si, k0+hh]/(C[si, k0]+1e-12) - 1.0 - 2.0*FEE
            g1000 = max(0, min(int(p/0.01*1000), G_LVL-1))
            tt = min(int((hh-1)/12)+1, T_LVL-1)
            G_t[tt, g1000] += p; G_s[tt, g1000] += 1
    G = np.where(G_s > 0, G_t/(G_s+1e-10), 0.0)
    say(f'  退出账本  {G_s.sum():.0f} 样本  [{time.time()-t0:.0f}s]', flush=True)
    def run_exit(si, k0):
        p = 0.0
        for hh in range(1, 13):
            p = C[si, k0+hh]/(C[si, k0]+1e-12) - 1.0 - 2.0*FEE
            g1000 = max(0, min(int(p/0.01*1000), G_LVL-1))
            tt = min(int((hh-1)/12)+1, T_LVL-1)
            if G[tt, g1000] < 0.0:
                return p
        return p
    # ---- 2) 训练入场账本：每个候选点的"开仓→退出模型"净收益，按特征簇聚合 ----
    from sklearn.cluster import KMeans
    feat_names = ['r48','r288','vabs','rp']
    Xf = np.column_stack([F[n].ravel() for n in feat_names])
    mkt = np.nanmean(F['vabs'], axis=0)
    mq = np.quantile(mkt[np.isfinite(mkt)], [1/3, 2/3])
    mln = np.zeros(K, dtype=int)
    for k in range(K):
        v = mkt[k]
        mln[k] = 0 if (not np.isfinite(v) or v <= mq[0]) else (1 if v <= mq[1] else 2)
    tr_fit = np.arange(0, cut)
    km = KMeans(n_clusters=NCLUST, n_init=4, random_state=11).fit(Xf[tr_fit])
    cl = km.predict(Xf)
    NB = NCLUST * M_LVL
    TB = np.zeros(NB); SB = np.zeros(NB)
    for i in tr_idx:
        si = i // K; k0 = i % K
        b = cl[i] * M_LVL + mln[k0]
        TB[b] += run_exit(si, k0); SB[b] += 1
    EB = np.where(SB > 0, TB/(SB+1e-10), 0.0)
    say(f'  入场账本 {NB} 格  有效 {int((SB>0).sum())}  [{time.time()-t0:.0f}s]', flush=True)
    # ---- 3) 测试：只在 E > 0 的格子开仓 ----
    rows = []
    for si in range(len(syms)):
        for k in range(cut, K - H - 1, 12):
            b = cl[si*K + k] * M_LVL + mln[k]
            if SB[b] > 0 and EB[b] > 0:
                rows.append((si, run_exit(si, k)))
    if not rows:
        say('  ❌ 没有任何格子 E > 0 ⇒ 模型决定"不赌"（存活但没盈利）')
        return
    a = np.array([r[1] for r in rows]); sids = np.array([r[0] for r in rows])
    n = len(a); m = a.mean(); t = m/(a.std()/np.sqrt(n)+1e-10)
    eq = []
    for si in range(len(syms)):
        mm = sids == si
        if mm.sum() == 0: continue
        eq.append((1 + a[mm].mean()*CAP) ** mm.sum())
    eq = np.array(eq) if eq else np.array([1.0])
    bet = n / max(1, len(syms)*(K-cut)/12)
    say('=' * 100)
    say('【一】开仓价值 = 开仓 + 退出模型（不是固定持有）')
    say(f'  笔数 {n:,}  下注率 {bet*100:.1f}%  单笔 {m*100:+.3f}%  t={t:+.2f}')
    say(f'  子账户 盈利 {int((eq>1).sum())}/{len(eq)}  均值 {eq.mean():.4f}')
    say('【二】四段')
    qs = np.array_split(a, 4); ok4 = True
    for i, q in enumerate(qs):
        mm = q.mean()*100
        say(f'  段{i+1}  n={len(q):>6,}  单笔 {mm:+.3f}%')
        if mm <= 0: ok4 = False
    say('【三】判据（存活 + 持续盈利）')
    ok = eq.mean() > 1 and ok4 and bet > 0.05
    say(f'  总资产 {eq.mean():.4f} > 1 ? {"✅" if eq.mean()>1 else "❌"}   四段同正 ? {"✅" if ok4 else "❌"}   下注率 {bet*100:.1f}% > 5% ? {"✅" if bet>0.05 else "❌"}')
    say(f'  {"✅ 模型的行为在概率下活下来并盈利" if ok else "❌ 未达标"}')
    say('  未计：滑点、maker 挂单成交率')
    say(f'TOTAL {time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(OUT, 'unified2.txt'), 'w', encoding='utf-8') as f:
        f.write(f'开仓价值=开仓+退出模型 笔数={n} 下注率={bet*100:.1f}% 单笔={m*100:+.3f}% t={t:+.2f} 总资产={eq.mean():.4f} 四段同正={ok4}\n')

if __name__ == '__main__':
    import gc
    main()
