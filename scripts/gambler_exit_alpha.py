# gambler_exit_alpha.py —— 随机入场 + 模型退出：退出有没有独立 alpha
# 若"随机入场 + 模型退出" > "随机入场 + 固定持有"，说明退出才是真 alpha，入场不需要择时
import sys, io, os, time, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(ROOT, 'evidence')
NSYM = int(os.environ.get('EA_NSYM', '60'))
BAR = int(os.environ.get('EA_BAR', '5'))
FEE = 0.0002
H = 176 * BAR
BS = 2000
MAXH = 144 * BAR
LVL = 10
def say(*a, **kw): print(*a, flush=kw.get('flush', True))
def zs(A, win):
    df = pd.DataFrame(A); m = df.rolling(win, min_periods=win).mean(); s = df.rolling(win, min_periods=win).std()
    return ((df - m) / (s + 1e-10)).fillna(0).values

def book(hold, gain, fut, lvl):
    n = len(hold); T = np.zeros((lvl, lvl)); S = np.zeros((lvl, lvl))
    g = np.clip((gain / 0.01 * 1000).astype(int), 0, lvl - 1)
    for i in range(n):
        t = g[i] if hold[i] <= 0 else int((hold[i] - 1) / (14 * BAR)) + 1
        if 0 <= t < lvl:
            T[t, g[i]] += fut[i]; S[t, g[i]] += 1
    return np.where(S > 0, T / (S + 1e-10), 0.0), S

def main():
    say('=' * 100)
    say('gambler_exit_alpha · 随机入场 + 模型退出：退出有没有独立 alpha')
    say(f'  币 {NSYM} | K=5 | 停 8笔/0.4%→50步 | 固定 H={H} | 每笔 0.5% | 入场=每 12 步随机')
    say('=' * 100)
    t0 = time.time()
    use = None
    with open(os.path.join(DATA, 'klines_1m_1y.csv'), 'r', encoding='utf-8') as f:
        r = pd.read_csv(f, nrows=1)
        if 'symbol' in r.columns: use = 'symbol'
        elif 'Symbol' in r.columns: use = 'Symbol'
    df = pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=[use, 'open_time','open','high','low','close','volume'],
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
    B = min(1000, C.shape[1] // 2)
    say(f'  数据 {df.shape[0]:,} 行 / {len(syms)} 币  理论 {C.shape[1]} 根 [{time.time()-t0:.0f}s]', flush=True)
    del df; gc.collect()
    R1 = np.zeros_like(C); R1[:, 1:] = C[:, 1:] / (C[:, :-1] + 1e-12) - 1.0
    R1 = np.clip(R1, -0.4, 0.4); R5 = R1
    RV = np.zeros_like(C); RV[:, 1:] = np.abs(C[:, 1:] / (C[:, :-1] + 1e-12) - 1.0) * 1e4
    s = pd.DataFrame(RV); RV = s.rolling(12, min_periods=12).mean().fillna(0).values
    s = pd.DataFrame(R1); F = {}
    for w in [48, 288, 1152]: F[f'r{w}'] = s.rolling(w, min_periods=w).sum().fillna(0).values
    F['vol'] = s.rolling(48, min_periods=48).std().fillna(0).values
    Cr = C[:, None, :]; F['rp'] = ((Cr - L[:, None, :]) / (Hh - L + 1e-12)[:, None, :])[:, 0, :]
    win = 1440
    F['r48'] = zs(F['r48'], win); F['r288'] = zs(F['r288'], win)
    F['r1152'] = zs(F['r1152'], win); F['vol'] = zs(F['vol'], win)
    K = C.shape[1]
    F['sgn'] = np.zeros_like(C); s = pd.DataFrame(np.sign(C - O)); F['sgn'] = s.rolling(96, min_periods=96).mean().fillna(0).values
    F['sgn'] = zs(F['sgn'], win)
    dev = np.sign(C - O) * np.abs(C - O) / (C + 1e-12) * 1e4
    s = pd.DataFrame(dev); F['dev'] = np.zeros_like(C); F['dev'] = s.rolling(12, min_periods=12).mean().fillna(0).values
    F['dev'] = zs(F['dev'], win)
    F['vol250'] = zs(pd.DataFrame(np.log(RV + 1e-12)).rolling(250, min_periods=250).mean().fillna(0).values, win)
    F['vabs'] = zs(pd.DataFrame(RV).rolling(250, min_periods=250).mean().fillna(0).values, win)
    F['p250'] = pd.DataFrame(C).rolling(250, min_periods=250).max().fillna(0).values
    F['lp'] = zs(np.log(C / (F['p250'] + 1e-12)), win)
    names = ['r48','r288','r1152','vol','rp','sgn','dev','vabs','lp','vol250']
    say(f'  {len(names)} 基元   [  {time.time()-t0:.0f}s]', flush=True)
    Xs=[]; ys=[]; ts=[]
    for si, sym in enumerate(syms):
        fwd = np.zeros(K)
        for k in (2, H):
            fwd[:-int(k*1.0)] = np.maximum(fwd[:-int(k*1.0)], C[si, int(k*1.0):]/C[si, :-int(k*1.0)] - 1.0)
        fwd = fwd - 2.0*FEE
        x = np.column_stack([F[n][si] for n in names])
        Xs.append(x.astype(np.float32)); ys.append(fwd.astype(np.float32)); ts.append(np.arange(K, dtype=np.int32))
    X = np.concatenate(Xs); y = np.concatenate(ys); T = np.concatenate(ts)
    nt = len(X)
    cut = int(nt*0.7)
    tr = np.arange(0, cut, 4); te = np.arange(cut, nt, 12)
    say(f'  样本 {nt:,}  训练 {len(tr):,}  测试 {len(te):,}   [  {time.time()-t0:.0f}s]', flush=True)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-10
    Xn = ((X - mu)/sd).astype(np.float32)
    from sklearn.neighbors import KDTree
    tree = KDTree(Xn[tr], leaf_size=40)
    _, nb = tree.query(Xn[te], k=5)
    r = y[tr][nb]
    exp = r.mean(1); var = r.var(1) + 1e-10
    # 模型退出：前 12 根，用"浮盈分档 + 市场状态"账本
    # 训练账本：用随机入场（每 12 步）在前 70% 的路径
    rng = np.random.default_rng(11)
    tr_idx = np.arange(0, cut, 12)
    tr_idx = tr_idx[tr_idx + H < cut]
    tr_idx = tr_idx[(tr_idx % K) + H + MAXH < K]
    say(f'  模型退出训练点 {len(tr_idx):,}   [  {time.time()-t0:.0f}s]', flush=True)
    hold_l = []; gain_l = []; fut_l = []
    for i in tr_idx:
        si = i // K; k0 = i % K
        for hh in range(1, MAXH + 1):
            p = C[si, k0 + hh] / (C[si, k0] + 1e-12) - 1.0 - 2.0*FEE
            hold_l.append(hh); gain_l.append(p)
            fut_l.append(max(p, 0.0) if p > 0 else p)  # 继续持有的收益
    hold = np.array(hold_l); gain = np.array(gain_l); fut = np.array(fut_l)
    G, _ = book(hold, gain, fut, LVL)
    say(f'  账本 {LVL*LVL} 格  [{time.time()-t0:.0f}s]', flush=True)
    say(f'  账本均匀项  [  {time.time()-t0:.0f}s]', flush=True)
    NU = float(fut.mean())
    def fin(a, tag):
        n = len(a)
        m = a.mean(); med = np.median(a); wr = (a>0).mean()
        t = m/(a.std()/np.sqrt(n)+1e-10)
        # 每币独立（按 si 分组）
        sids = np.array([si for si, k0 in te_si])
        eq = []
        for si in range(len(syms)):
            m_ = sids == si
            if m_.sum() == 0: continue
            rr = a[m_]
            eq.append((1 + rr.mean()*0.005) ** len(rr))
        eq = np.array(eq) if eq else np.array([1.0])
        say(f'  {tag:<24} n={n:>7,}  单笔 {m*100:+.3f}%  中位 {med*100:+.3f}%  胜率 {wr*100:.1f}%  t={t:+.2f}  子账户 {eq.mean()-1:+.2%}  {eq.mean():.4f}')
        return m, t
    say('=' * 100)
    say('【一】随机入场 每 12 步（不重叠）')
    base = np.array([C[si, k0+H]/(C[si, k0]+1e-12) - 1.0 - 2.0*FEE for si, k0 in [(i//K, i%K) for i in te if i//K < len(syms) and i%K + H < K]])
    te_si = [(i//K, i%K) for i in te if i//K < len(syms) and i%K + max(H, MAXH) < K]
    base = np.array([C[si, k0+H]/(C[si, k0]+1e-12) - 1.0 - 2.0*FEE for si, k0 in te_si])
    fin(base, '随机入场 + 固定H=176')
    # 模型退出：逐根查表
    gm = []
    for si, k0 in te_si:
        p = 0.0
        for hh in range(1, MAXH + 1):
            p = C[si, k0+hh]/(C[si, k0]+1e-12) - 1.0 - 2.0*FEE
            g1000 = int(p/0.01*1000)
            g1000 = max(0, min(g1000, LVL-1))
            t = g1000 if hh <= 0 else int((hh-1)/(14*BAR))+1
            t = min(t, LVL-1)
            if G[t, g1000] < 0.0:
                break
        gm.append(p)
    gm = np.array(gm)
    fin(gm, '随机入场 + 模型退出')
    say('=' * 100)
    say('【二】判据（存活 + 持续盈利）')
    ok = gm.mean() > 0
    say(f'  随机入场 + 模型退出  单笔 {gm.mean()*100:+.3f}%')
    say(f'  随机入场 + 固定持有  单笔 {base.mean()*100:+.3f}%')
    say(f'  退出 alpha（差）     {(gm.mean()-base.mean())*100:+.3f}%')
    say(f'  {"✅ 退出有独立 alpha ⇒ 入场不需要择时，赌徒只需决定下注与收手" if ok else "❌ 退出也没有 alpha ⇒ 这条路不通"}')
    say('  未计：滑点、maker 挂单成交率')
    say(f'TOTAL {time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(OUT, 'exit_alpha.txt'), 'w', encoding='utf-8') as f:
        f.write('随机入场 + 模型退出：退出有没有独立 alpha\n')
        f.write(f'随机入场 + 固定持有 {base.mean()*100:+.3f}%\n')
        f.write(f'随机入场 + 模型退出 {gm.mean()*100:+.3f}%\n')
        f.write(f'退出 alpha {(gm.mean()-base.mean())*100:+.3f}%\n')

if __name__ == '__main__':
    import gc
    main()
