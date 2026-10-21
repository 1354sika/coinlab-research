# -*- coding: utf-8 -*-
"""出场进化实验：随机入场（零信息），模型只学「什么时候平」。
设计：
  - 入场：随机时点、只做多（剥离入场优势，让信号只来自出场）
  - 路径：入场后最多 H=24 根（2 小时）的 high/low/close
  - 事后最优上界：该笔在 H 内的最大浮盈（扣成本）——用来衡量「还差多少」
  - 策略空间（参数化规则，进化搜索）：
       止盈 TP、止损 s、移动止盈启动 act、回撤 trail、超时 T
  - 训练段 = 前 70% 时间；测试段 = 后 30%（时间外验证，防止过拟合）
输出：事后最优上界 / 固定规则基准 / 进化最优参数 / 测试段每笔净收益 / regret
"""
import time
import sys
import numpy as np
import pandas as pd

COND = len(sys.argv) > 1 and sys.argv[1] == 'cond'   # cond=只用「好入场」（急跌+巨量）

DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/exit_evolve_REAL.txt'
COST = 0.0004
H = 24
PER_SYM = 1500
N_TRIAL = 400
SEED = 7

t0 = time.time()
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
df = df.sort_values(['symbol', 'open_time'])
print('rows %d load %.1fs' % (len(df), time.time() - t0), flush=True)

FH, FL, FC, TT = [], [], [], []
for sym, g in df.groupby('symbol'):
    c = g['close'].to_numpy(float)
    hi = g['high'].to_numpy(float)
    lo = g['low'].to_numpy(float)
    tt = g['open_time'].to_numpy()
    n = len(c)
    if n < H + 50:
        continue
    idx = rng.choice(np.arange(0, n - H - 1), size=min(PER_SYM, n - H - 1), replace=False)
    if COND:
        v = g['volume'].to_numpy(float)
        ret6 = np.full(n, np.nan)
        ret6[6:] = c[6:] / c[:-6] - 1
        vma = pd.Series(v).rolling(24).mean().to_numpy()
        volr = v / vma
        ok = (~np.isnan(ret6)) & (~np.isnan(volr)) & (ret6 <= -0.02) & (volr >= 3)
        cand = np.arange(0, n - H - 1)
        cand = cand[ok[cand]]
        if len(cand) < 50:
            continue
        idx = rng.choice(cand, size=min(PER_SYM, len(cand)), replace=False)
    idx.sort()
    p0 = c[idx][:, None]
    FH.append(np.stack([hi[idx + j] for j in range(H + 1)], 1) / p0 - 1)
    FL.append(np.stack([lo[idx + j] for j in range(H + 1)], 1) / p0 - 1)
    FC.append(np.stack([c[idx + j] for j in range(H + 1)], 1) / p0 - 1)
    TT.append(tt[idx])

fh = np.concatenate(FH); fl = np.concatenate(FL); fc = np.concatenate(FC); tt = np.concatenate(TT)
m = len(fc)
order = np.argsort(tt)
fh, fl, fc, tt = fh[order], fl[order], fc[order], tt[order]
split = int(m * 0.7)
print('paths %d  train %d  test %d  prep %.1fs' % (m, split, m - split, time.time() - t0), flush=True)

best = np.max(fh, axis=1) - COST          # 事后最优（每笔最大浮盈，扣成本）

def run(tp, s, act, trail, T_, sl):
    peak = np.maximum.accumulate(fh[sl], axis=1)
    n = sl.stop - sl.start
    out = np.full(n, np.nan)
    alive = np.ones(n, bool)
    for j in range(1, H + 1):
        hit_tp = fh[sl, j] >= tp
        hit_sl = fl[sl, j] <= -s
        hit_tr = (peak[:, j] >= act) & (fl[sl, j] <= peak[:, j] - trail)
        hit_t = np.full(n, j >= T_)
        hit = alive & (hit_tp | hit_sl | hit_tr | hit_t)
        if hit.any():
            e = np.where(hit_sl, -s, np.where(hit_tp, tp, np.where(hit_tr, peak[:, j] - trail, fc[sl, j])))
            out[hit] = e[hit]
            alive &= ~hit
        if not alive.any():
            break
    out[np.isnan(out)] = fc[sl, -1][np.isnan(out)]
    return out - COST

tr = slice(0, split); te = slice(split, m)
grid = dict(tp=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2],
            s=[0.005, 0.01, 0.02, 0.03, 0.05],
            act=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
            trail=[0.003, 0.005, 0.01, 0.02, 0.03, 0.05],
            T=[6, 12, 18, 24])
res = []
for i in range(N_TRIAL):
    tp = rng.choice(grid['tp']); s = rng.choice(grid['s']); act = rng.choice(grid['act'])
    trail = rng.choice(grid['trail']); T_ = int(rng.choice(grid['T']))
    r = run(tp, s, act, trail, T_, tr)
    res.append((r.mean(), tp, s, act, trail, T_))
res.sort(reverse=True)
print('search %d trials %.0fs' % (N_TRIAL, time.time() - t0), flush=True)

# 基准：固定止盈/超时
def bench(tp, T_):
    return run(tp, 1.0, 9.9, 9.9, T_, tr).mean(), run(tp, 1.0, 9.9, 9.9, T_, te).mean()

lines = []
lines.append('出场进化实验：随机入场（零信息）、只做多、H=%d 根(2h)、扣 %.2f%% 往返' % (H, COST * 100))
lines.append('样本 %d 条路径（60 币 × %d 随机时点），按时间 70/30 切训练/测试' % (m, PER_SYM))
lines.append('')
lines.append('【事后最优上界（不考虑能不能抓到）】训练段 %.3f%%/笔  测试段 %.3f%%/笔' % (best[tr].mean() * 100, best[te].mean() * 100))
lines.append('')
lines.append('【固定规则基准（无进化）】训练 / 测试（%%，每笔净）')
for tp, T_ in ((0.03, 12), (0.05, 12), (0.08, 24), (0.12, 24)):
    a, b = bench(tp, T_)
    lines.append('  止盈 %.0f%% + 超时 %d 根        %.3f%% / %.3f%%    regret %.0f%%' % (tp * 100, T_, a * 100, b * 100, (1 - b / best[te].mean()) * 100))
lines.append('')
lines.append('【进化最优（训练段搜出，测试段验证）】')
lines.append('%-30s %14s %14s %10s' % ('参数', '训练/笔', '测试/笔', 'regret'))
for (mean_tr, tp, s, act, trail, T_) in res[:8]:
    r_te = run(tp, s, act, trail, T_, te).mean()
    reg = (1 - r_te / best[te].mean()) * 100
    lines.append('TP%.0f%% s%.1f%% act%.0f%% tr%.1f%% T%d   %13.3f%% %13.3f%% %9.0f%%' %
                 (tp * 100, s * 100, act * 100, trail * 100, T_, mean_tr * 100, r_te * 100, reg))
lines.append('')
lines.append('【读法】regret = 距离「每笔都卖在最高点」还差多少。100% = 一分没赚，0% = 完美。')
lines.append('【提醒】随机入场下每笔期望本就接近 0；此处看的是「同一批随机入场，换出场规则能差多少」。')
lines.append('TOTAL %.0fs' % (time.time() - t0))
txt = '\n'.join(lines)
print(txt, flush=True)
open(OUT, 'w', encoding='utf-8').write(txt)
print('\n已写 ' + OUT, flush=True)
