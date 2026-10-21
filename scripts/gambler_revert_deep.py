# -*- coding: utf-8 -*-
"""深挖 revert（抄底）：跌多少买、持多久、加急停

上一轮发现：revert（近5根跌3%就买）单笔 +0.368%、t=+4.53，是唯一显著为正的牌。
本脚本回答三个问题：
  ① 跌多少买最优？（网格 1%~10%）
  ② 持多久最优？（网格 24~576 根）
  ③ 加"连续亏损急停"能不能压住回撤？四段是否稳定？

严格无前视：入场只看 i-5 到 i 的价格；急停只看已结算的历史。

输出：evidence/revert_deep.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('RD_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\revert_deep.txt'
FEE = 0.0004
SLIP = float(os.environ.get('RD_SLIP', '0.0005'))
NSYM = int(os.environ.get('RD_NSYM', '60'))
COST = FEE + SLIP

DROPS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
HOLDS = [24, 48, 96, 176, 288, 576]
KELLY = 0.25
MAXLOSS = 0.02
STOP_N = int(os.environ.get('RD_STOPN', '5'))

TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('深挖 revert · 跌多少买 / 持多久 / 急停')
say(f'  {os.path.basename(DATA)}  成本 {COST:.3%}  急停=连亏{STOP_N}次')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

# 预取每个币的 close 与时间
data = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    if len(C) < 3000:
        continue
    data.append((sym, C, g['open_time'].values))
say(f'  可用币 {len(data)}  {time.time()-t0:.0f}s')

def run_grid(drop, hold):
    rows = []
    for sym, C, ts in data:
        T = len(C)
        i = 5
        while i < T - hold:
            if C[i] / C[i - 5] - 1.0 <= -drop:
                rows.append((ts[i], C[i + hold] / C[i] - 1.0 - COST))
                i += hold
            else:
                i += 1
    return rows

say('')
say('=' * 100)
say('【一】网格：跌 N% 买 × 持 H 根')
say('=' * 100)
say(f'  {"跌":>6s} {"持":>6s} {"笔数":>9s} {"单笔均值":>9s} {"胜率":>7s} {"t值":>7s}')
res = {}
for d in DROPS:
    for h in HOLDS:
        rows = run_grid(d, h)
        if len(rows) < 500:
            continue
        r = np.array([x[1] for x in rows])
        mu = r.mean(); sd = r.std(ddof=1)
        tv = mu / (sd / np.sqrt(len(r))) if sd > 0 else 0
        res[(d, h)] = (mu, tv, len(r), (r > 0).mean(), rows)
        say(f'  {d:>5.0%} {h:>6d} {len(r):>9,} {mu*100:>+8.3f}% {(r>0).mean():>6.1%} {tv:>+7.2f}')

if not res:
    say('  ❌ 无有效组合')
    sys.exit(1)

best_key = max(res, key=lambda k: res[k][0])
bd, bh, bmu, btv = best_key[0], best_key[1], res[best_key][0], res[best_key][1]
say('')
say(f'  最好组合：跌 {bd:.0%} 买、持 {bh} 根 → 单笔 {bmu*100:+.3f}%、t={btv:+.2f}')

# ---- 四段稳定 ----
rows = res[best_key][4]
rows.sort(key=lambda x: x[0])
ret = np.array([x[1] for x in rows])
ts = np.array([x[0] for x in rows])
say('')
say('=' * 100)
say('【二】四段稳定性（最好组合）')
say('=' * 100)
seg = pd.qcut(np.arange(len(ret)), 4, labels=False)
for s in range(4):
    m = ret[seg == s]
    say(f'  段{s+1}  n={len(m):>7,}  单笔 {m.mean()*100:>+7.3f}%  t={m.mean()/(m.std(ddof=1)/np.sqrt(len(m))):>+6.2f}')
same = all(ret[seg == s].mean() > 0 for s in range(4))
say(f'  {"✅ 四段同正" if same else "❌ 有段非正"}')

# ---- 加急停 ----
say('')
say('=' * 100)
say('【三】加急停（连亏 N 次就停 M 笔）')
say('=' * 100)
say(f'  {"停多久":>8s} {"期末资金":>12s} {"最大回撤":>10s} {"停牌次数":>8s} {"停掉的笔数":>10s}')
for cool in [0, 50, 200, 500]:
    cap = 1.0; peak = 1.0; dd = 0.0
    lose_streak = 0
    hist = []
    skip = 0
    halts = 0
    for i in range(len(ret)):
        if skip > 0:
            skip -= 1
            continue
        r = ret[i]
        if len(hist) >= 200:
            h = np.array(hist[-200:])
            p = (h > 0).mean()
            win = h[h > 0].mean() if (h > 0).any() else 0.0
            los = -h[h <= 0].mean() if (h <= 0).any() else 1e-9
            b = win / max(los, 1e-9)
            edge = p * b - (1 - p)
            f = min(KELLY * edge / max(b, 1e-9), MAXLOSS / max(los, 1e-9), 0.15) if edge > 0 else 0.0
            f = max(0.0, f)
        else:
            f = 0.0
        cap *= (1.0 + f * r)
        peak = max(peak, cap)
        dd = min(dd, cap / peak - 1.0)
        hist.append(r)
        if r <= 0:
            lose_streak += 1
        else:
            lose_streak = 0
        if cool > 0 and lose_streak >= STOP_N:
            skip = cool
            halts += 1
            lose_streak = 0
    say(f'  {cool:>8d} {cap:>12.4f} {dd*100:>9.2f}% {halts:>8d} {cool*halts:>10,}')

say('')
say('=' * 100)
say('【四】判据')
say('=' * 100)
say('  · 网格里若最优在边界（1% 或 10%）⇒ 说明区间没覆盖到真最优，要扩大')
say('  · 四段同正 ⇒ 这张牌是真的，不是某一段运气')
say('  · 急停能把回撤压下来且期末资金不降 ⇒ 该加')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
