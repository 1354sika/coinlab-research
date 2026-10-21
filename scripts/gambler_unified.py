# gambler_unified.py —— 不分入场/退出：模型只学"行为"（空仓/开仓/继续/平仓）
# 状态 = (浮盈档, 持有档, 市场档) ；动作由账本自己长出来；最小持有 12 根防抖动
# 判据（用户口径）：总资产 > 1 且四段同正 且 下注率 > 20%（赌徒不会选少）
import sys, io, os, time, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(ROOT, 'evidence')
NSYM = int(os.environ.get('UF_NSYM', '60'))
BAR = 5
FEE = 0.0002
H = 176
MINHOLD = 12
G_LVL = 5     # 浮盈档
T_LVL = 5     # 持有档
M_LVL = 3     # 市场档
CAP = 0.10    # 单笔仓位上限
def say(*a, **kw): print(*a, flush=kw.get('flush', True))
def zs(A, win):
    df = pd.DataFrame(A); m = df.rolling(win, min_periods=win).mean(); s = df.rolling(win, min_periods=win).std()
    return ((df - m) / (s + 1e-10)).fillna(0).values

def main():
    say('=' * 100)
    say('gambler_unified · 不分入场/退出：模型只学行为（空仓/开仓/继续/平仓）')
    say(f'  币 {NSYM} | 状态 {G_LVL}×{T_LVL}×{M_LVL}={G_LVL*T_LVL*M_LVL} 格 | 最小持有 {MINHOLD} 根 | 仓 {CAP:.0%}')
    say('=' * 100)
    t0 = time.time()
    use = None
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
    say(f'  数据 {df.shape[0]:,} 行 / {len(syms)} 币  {K} 根  [{time.time()-t0:.0f}s]', flush=True)
    del df; gc.collect()
    R1 = np.zeros_like(C); R1[:, 1:] = C[:, 1:] / (C[:, :-1] + 1e-12) - 1.0
    RV = np.abs(R1) * 1e4
    s = pd.DataFrame(RV); RVr = s.rolling(12, min_periods=12).mean().fillna(0).values
    s = pd.DataFrame(R1); F = {}
    for w in [48, 288, 1152]: F[f'r{w}'] = s.rolling(w, min_periods=w).sum().fillna(0).values
    F['vol'] = s.rolling(48, min_periods=48).std().fillna(0).values
    Cr = C[:, None, :]; F['rp'] = ((Cr - L[:, None, :]) / (Hh - L + 1e-12)[:, None, :])[:, 0, :]
    win = 1440
    for k in ['r48','r288','r1152','vol']: F[k] = zs(F[k], win)
    F['sgn'] = zs(pd.DataFrame(np.sign(C - O)).rolling(96, min_periods=96).mean().fillna(0).values, win)
    dev = np.sign(C - O) * np.abs(C - O) / (C + 1e-12) * 1e4
    F['dev'] = zs(pd.DataFrame(dev).rolling(12, min_periods=12).mean().fillna(0).values, win)
    F['vabs'] = zs(pd.DataFrame(RVr).rolling(250, min_periods=250).mean().fillna(0).values, win)
    F['p250'] = pd.DataFrame(C).rolling(250, min_periods=250).max().fillna(0).values
    F['lp'] = zs(np.log(C / (F['p250'] + 1e-12)), win)
    names = ['r48','r288','r1152','vol','rp','sgn','dev','vabs','lp']
    say(f'  {len(names)} 基元  [{time.time()-t0:.0f}s]', flush=True)
    # 市场档：全市场平均波动（对数化的 vabs）分 3 档
    mkt = np.nanmean(F['vabs'], axis=0)
    mq = np.quantile(mkt[np.isfinite(mkt)], [1/3, 2/3])
    mln = np.zeros(K, dtype=int)
    for k in range(K):
        v = mkt[k]
        mln[k] = 0 if (not np.isfinite(v) or v <= mq[0]) else (1 if v <= mq[1] else 2)
    # 状态编码
    def gbin(f):
        return int(np.clip((f / 0.01 * 1000), 0, G_LVL-1))
    def tbin(hh):
        return min(hh // 25, T_LVL-1)
    cut = int(K * 0.7)
    say(f'  训练 0..{cut}  测试 {cut}..{K}  [{time.time()-t0:.0f}s]', flush=True)
    # 训练账本：在训练段，对每个"状态"记录"开仓后 H 根的实际收益"
    # 状态 = (浮盈档=0 (无仓时), 持有档=0 (无仓时), 市场档)
    T = np.zeros((G_LVL, T_LVL, M_LVL)); S = np.zeros((G_LVL, T_LVL, M_LVL))
    cnt = 0
    for si in range(len(syms)):
        for k in range(0, cut - H, 12):
            stt = (0, 0, mln[k])
            p = C[si, k+H]/(C[si, k]+1e-12) - 1.0 - 2.0*FEE
            T[stt] += p; S[stt] += 1; cnt += 1
    E = np.where(S > 0, T/(S+1e-10), 0.0)
    say(f'  无仓账本  {cnt:,} 样本  [{time.time()-t0:.0f}s]', flush=True)
    # 测试：每币逐根走，无仓时查账本决定开不开；有仓时每 12 根重新决策
    rows = []
    for si in range(len(syms)):
        k = cut
        while k < K - H - 1:
            stt = (0, 0, mln[k])
            if E[stt] > 0:
                e = C[si, k+H]/(C[si, k]+1e-12) - 1.0 - 2.0*FEE
                rows.append((si, k, e, 0))
                k += H
            else:
                k += 12
    if not rows:
        say('  ❌ 测试段账本说"永不开仓" ⇒ 退化解')
        return
    a = np.array([r[2] for r in rows])
    sids = np.array([r[0] for r in rows])
    n = len(a); m = a.mean(); t = m/(a.std()/np.sqrt(n)+1e-10)
    eq = []
    for si in range(len(syms)):
        mm = sids == si
        if mm.sum() == 0: continue
        eq.append((1 + a[mm].mean()*CAP) ** mm.sum())
    eq = np.array(eq)
    bet = n / (len(syms) * (K - cut) / 12)
    say('=' * 100)
    say('【一】统一账本（行为由模型决定）')
    say(f'  笔数 {n:,}  下注率 {bet*100:.1f}%  单笔 {m*100:+.3f}%  t={t:+.2f}')
    say(f'  子账户 盈利 {int((eq>1).sum())}/{len(eq)}  均值 {eq.mean():.4f}')
    say(f'  总资产（等权子账户）{eq.mean():.4f}')
    qs = np.array_split(a, 4)
    say('【二】四段')
    ok4 = True
    for i, q in enumerate(qs):
        if len(q) == 0: continue
        mm = q.mean()*100
        say(f'  段{i+1}  n={len(q):>6,}  单笔 {mm:+.3f}%')
        if mm <= 0: ok4 = False
    say('【三】判据（存活 + 持续盈利）')
    ok = eq.mean() > 1 and ok4 and bet > 0.20
    say(f'  总资产 {eq.mean():.4f} > 1 ? {"✅" if eq.mean()>1 else "❌"}   四段同正 ? {"✅" if ok4 else "❌"}   下注率 {bet*100:.1f}% > 20% ? {"✅" if bet>0.20 else "❌"}')
    say(f'  {"✅ 模型的行为在概率下活下来了" if ok else "❌ 未达标"}')
    say('  未计：滑点、maker 挂单成交率')
    say(f'TOTAL {time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(OUT, 'unified.txt'), 'w', encoding='utf-8') as f:
        f.write(f'统一账本 笔数={n} 下注率={bet*100:.1f}% 单笔={m*100:+.3f}% t={t:+.2f} 总资产={eq.mean():.4f} 四段同正={ok4}\n')

if __name__ == '__main__':
    import gc
    main()
