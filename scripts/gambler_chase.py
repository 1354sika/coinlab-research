# -*- coding: utf-8 -*-
"""追涨策略：v2 赔率体检发现的信号做成可执行的东西。

v2 的发现（赔率只用过去 240 根区间，收益只用未来）：
  最低赔率档（odds≈0.07，价格贴着区间上沿）未来 1 天 +0.138%，扣费 +0.098%，t=+15.69
  最高赔率档（odds≈12，贴近区间下沿）未来 1 天 -0.044%
  ⇒ 方向是**追涨**，不是低买高卖。

本脚本把它做成能跑的策略，回答三个问题：
  ① 只在低赔率档买入，资金曲线长什么样
  ② 四段是否同号
  ③ 仓位按优势大小分配（odds 越小仓位越大）是否比固定仓位好

规则（全部无前视）：
  odds = (过去240根最高 − 现在) / (现在 − 过去240根最低)
  odds <= THR 且空仓 → 买入
  持满 H 根 → 卖出（先只测固定持有，不加止损，避免额外参数）
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = os.environ.get('CH_DATA', r'D:\中信证券\coinlab\data\klines_5m_3y.csv')
OUT = r'D:\中信证券\coinlab\evidence\chase_strategy.txt'
BPD = int(os.environ.get('CH_BPD', '288'))
H = int(os.environ.get('CH_H', str(BPD)))
LOOK = int(os.environ.get('CH_LOOK', '240'))
NSYM = int(os.environ.get('CH_NSYM', '60'))
FEE = 0.0004
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)

t0 = time.time()
say('=' * 100)
say('追涨策略 · 只在"贴近区间上沿"时买入')
say(f'  数据 {os.path.basename(DATA)}  回看 {LOOK}  持有 {H}  币 {NSYM}  费 {FEE:.2%}')
say('=' * 100)

df = pd.read_csv(DATA)
syms = sorted(df.symbol.unique())[:NSYM]
df = df[df.symbol.isin(syms)]
say(f'  载入 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

def run(g, thr, size_mode):
    """在单个币上跑时间流，返回交易列表"""
    g = g.sort_values('open_time')
    C = g['close'].values.astype(float)
    Hi = g['high'].values.astype(float)
    Lo = g['low'].values.astype(float)
    T = len(C)
    hi_p = pd.Series(Hi).shift(1).rolling(LOOK, min_periods=LOOK // 2).max().values
    lo_p = pd.Series(Lo).shift(1).rolling(LOOK, min_periods=LOOK // 2).min().values
    up = hi_p - C
    dn = C - lo_p
    with np.errstate(divide='ignore', invalid='ignore'):
        odds = up / np.maximum(dn, 1e-9)
    ok = np.isfinite(odds) & (dn > 0)

    trades = []
    i = LOOK
    while i < T - H:
        if not ok[i] or odds[i] > thr:
            i += 1
            continue
        # 买入
        entry = C[i]
        exit_i = i + H
        r = C[exit_i] / entry - 1.0 - FEE
        if size_mode == 'flat':
            sz = 1.0
        else:  # 按优势：odds 越小（越贴顶）仓位越大，归一到 [0.2, 1.0]
            sz = float(np.clip(1.0 - odds[i] / max(thr, 1e-9) * 0.8, 0.2, 1.0))
        trades.append((i, exit_i, r, sz))
        i = exit_i            # 持满才能再买（不重叠）
    return trades

say('')
say('=' * 100)
say('【一】不同阈值的对比（固定仓位 vs 按优势加仓）')
say('=' * 100)
say(f'  {"阈值":>6s} {"仓位":>6s} {"笔数":>8s} {"单笔均值":>10s} {"中位":>9s} {"胜率":>7s} {"t":>8s} {"资金":>8s}')
BEST = None
for thr in (0.10, 0.20, 0.35, 0.50, 0.80):
    for mode in ('flat', 'edge'):
        allr, allsz, per_sym = [], [], []
        for sym, g in df.groupby('symbol', sort=False):
            tr = run(g, thr, mode)
            if not tr:
                continue
            r = np.array([x[2] for x in tr])
            sz = np.array([x[3] for x in tr])
            allr.append(r); allsz.append(sz)
            per_sym.append((sym, len(r), r.mean(), (r * sz).sum() / max(sz.sum(), 1e-9)))
        if not allr:
            continue
        R = np.concatenate(allr); S = np.concatenate(allsz)
        m = R.mean(); md = np.median(R); wr = (R > 0).mean()
        t = m / (R.std(ddof=1) / np.sqrt(len(R))) if len(R) > 30 else np.nan
        # 资金曲线：每笔按仓位下注，复合
        eq = np.prod(1.0 + R * S) if len(R) < 50000 else np.exp(np.sum(np.log1p(np.clip(R * S, -0.99, None))))
        say(f'  {thr:>6.2f} {mode:>6s} {len(R):>8,} {m*100:>9.4f}% {md*100:>8.4f}% {wr*100:>6.1f}% {t:>+8.2f} {eq:>8.3f}')
        if BEST is None or m > BEST[0]:
            BEST = (m, thr, mode, R, S)

say('')
say('=' * 100)
say('【二】最好那组的四段检查')
say('=' * 100)
if BEST:
    m, thr, mode, R, S = BEST
    say(f'  选 {thr:.2f} / {mode}   单笔 {m*100:+.4f}%')
    q = np.array_split(np.arange(len(R)), 4)
    ok4 = True
    for k, idx in enumerate(q):
        seg = R[idx]
        mm = seg.mean()
        tt = mm / (seg.std(ddof=1) / np.sqrt(len(seg))) if len(seg) > 30 else np.nan
        say(f'   段{k+1}  n={len(seg):>7,}  均值 {mm*100:>+8.4f}%  t={tt:>+7.2f}')
        if mm <= 0:
            ok4 = False
    say(f'  → {"✅ 四段同正" if ok4 else "❌ 有段非正"}')

say('')
say('=' * 100)
say('【三】判据')
say('=' * 100)
say('  · 四段同正 + 单笔扣费后 > 0.05% ⇒ 追涨信号可交易，值得做成赌徒的入场基因')
say('  · 否则 ⇒ v2 的分档单调性只是统计假象，不能落地')
say(f'\nTOTAL {time.time()-t0:.0f}s')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(TXT))
say(f'写入 {OUT}')
