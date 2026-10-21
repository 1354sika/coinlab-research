# -*- coding: utf-8 -*-
"""赔率体检 v2（修正前视泄露）

v1 的致命错误：odds = 未来最高涨幅 / 未来最低跌幅 —— 用未来算的赔率去预测未来，
必然单调成立，是循环论证。

v2 的铁律：
  赔率只用**过去** 240 根的区间（高低点），收益只用**未来** H 根。
  两者不共用任何一根 K 线。

  上方空间 up_p = 过去240根最高 − 现在价
  下方空间 dn_p = 现在价 − 过去240根最低
  历史赔率 odds_p = up_p / dn_p
      odds_p 大 ⇒ 贴近区间底部，往上空间大
      odds_p 小 ⇒ 贴近区间顶部，往下空间大

检验：历史赔率能否预测未来 H 根的真实收益（扣费后），且四段同号。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('ODDS_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\odds_gate_v2.txt'
BPD = int(os.environ.get('ODDS_BPD', '288'))
H = int(os.environ.get('ODDS_H', str(BPD)))
LOOK = int(os.environ.get('ODDS_LOOK', '240'))      # 回看窗口（只用过去）
NSYM = int(os.environ.get('ODDS_NSYM', '20'))
FEE = 0.0004
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('赔率体检 v2 · 赔率只用过去，收益只用未来')
say(f'  数据 {os.path.basename(DATA)}  回看 {LOOK} 根  持有 {H} 根  币 {NSYM}  费 {FEE:.2%}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

parts = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    Hi = g['high'].values.astype(float)
    Lo = g['low'].values.astype(float)
    T = len(C)
    if T < LOOK + H + 100:
        continue
    cs = pd.Series(C)
    # ---- 过去（严格 shift 1，绝不包含当前 bar 之后）----
    hi_p = pd.Series(Hi).shift(1).rolling(LOOK, min_periods=LOOK // 2).max().values
    lo_p = pd.Series(Lo).shift(1).rolling(LOOK, min_periods=LOOK // 2).min().values
    up_p = hi_p - C
    dn_p = C - lo_p
    odds = up_p / np.maximum(dn_p, 1e-9)
    # ---- 未来（标签）----
    retH = cs.shift(-H).values / C - 1.0
    # 未来最高/最低（仅用于描述，不进特征）
    fhi = pd.Series(Hi).rolling(H).max().shift(-H).values / C - 1.0
    flo = pd.Series(Lo).rolling(H).min().shift(-H).values / C - 1.0

    ok = np.isfinite(odds) & np.isfinite(retH) & (dn_p > 0)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        continue
    parts.append(pd.DataFrame({
        'symbol': sym, 'i': idx, 'odds': odds[idx],
        'retH': retH[idx], 'fhi': fhi[idx], 'flo': flo[idx],
        'rp': (C[idx] - lo_p[idx]) / np.maximum(hi_p[idx] - lo_p[idx], 1e-12)}))

D = pd.concat(parts, ignore_index=True)
del parts
say(f'  样本 {len(D):,}  {time.time()-t0:.0f}s')

say('')
say('=' * 100)
say('【一】历史赔率分布（只用过去 240 根）')
say('=' * 100)
q = D.odds.quantile([.1, .25, .5, .75, .9])
say(f'  p10 {q[.1]:6.2f}  p25 {q[.25]:6.2f}  p50 {q[.5]:6.2f}  p75 {q[.75]:6.2f}  p90 {q[.9]:6.2f}')
n = len(D)
for thr in (1.0, 1.5, 2.0, 3.0):
    c = int((D.odds >= thr).sum())
    say(f'  odds>= {thr:.1f}: {c:>10,} ({c/n*100:5.1f}%)')

say('')
say('=' * 100)
say('【二】关键检验：历史赔率分 10 档 → 未来 H 根真实收益（扣费后）')
say('=' * 100)
say('  合法的分档：分档依据只用过去，收益只用未来')
say('')
D['q'] = pd.qcut(D.odds, 10, labels=False, duplicates='drop')
say(f'  {"档":>3s} {"odds中位":>9s} {"样本":>10s} {"未来收益":>10s} {"中位":>9s} {"胜率":>7s} {"扣费后":>10s} {"t":>8s}')
for k in range(int(D.q.max()) + 1):
    s = D[D.q == k]
    r = s.retH.values
    m = r.mean(); md = np.median(r); wr = (r > 0).mean()
    tt = m / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 30 and r.std() > 0 else np.nan
    say(f'  {k+1:>3d} {s.odds.median():>9.2f} {len(s):>10,} {m*100:>9.4f}% {md*100:>8.4f}% {wr*100:>6.1f}% {(m-FEE)*100:>9.4f}% {tt:>+8.2f}')

say('')
say('=' * 100)
say('【三】四段稳定性（最高档 vs 最低档，各段单独算）')
say('=' * 100)
D['seg'] = pd.qcut(D.groupby('symbol').cumcount(), 4, labels=False)
lo_k = int(D.q.max())
say(f'  {"段":>3s} {"最低赔率档":>12s} {"最高赔率档":>12s} {"高−低":>10s} {"高n":>9s} {"低n":>9s}')
for sg in range(4):
    a = D[(D.seg == sg) & (D.q == lo_k)].retH.mean()
    b = D[(D.seg == sg) & (D.q == 0)].retH.mean()
    na = int(((D.seg == sg) & (D.q == lo_k)).sum())
    nb = int(((D.seg == sg) & (D.q == 0)).sum())
    say(f'  {sg+1:>3d} {b*100:>11.4f}% {a*100:>11.4f}% {(a-b)*100:>+9.4f}% {na:>9,} {nb:>9,}')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 最高档 扣费后 > 0 且 4 段同号 ⇒ 历史赔率是可用的信号，赌徒该学"挑位置"')
say('  · 各档都在 0 附近或符号乱 ⇒ 赔率（区间位置）在 5m 上不预测方向')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
