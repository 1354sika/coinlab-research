# -*- coding: utf-8 -*-
"""gambler_tpsl_grid · 止盈 × 止损 二维穷尽

固定：V05=240min(4h)  V09=2880min  持有=1440min  成本=0.50%/笔  数据=D01(60币/1年/1m)
扫：  止盈 +5%~+50% 步 3%（16 档） ×  止损 -6%~-20% 步 1%（15 档）= 240 格
每格统计两种筛选：
  all = 全部入场点
  h3  = h3∈(0.1, 0.9)（最近3次赢过1~2次；前视已修：只吸收 t_close<=t_enter 的笔）

止盈止损均相对入场价。
"""
import os, time, heapq
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
VW, CONF, HOLD = 240, 2880, 1440
MAXSYM = int(os.environ.get('SYMS', '60'))

TP_GRID = [round(0.05 + 0.03 * i, 4) for i in range(16)]   # 5%..50%
SL_GRID = [round(0.06 + 0.01 * i, 4) for i in range(15)]   # 6%..20%

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_tpsl_grid · 止盈 × 止损 二维穷尽')
log('=' * 100)
log(f'V05={VW}min V09={CONF}min HOLD={HOLD}min FEE={FEE:.2%}')
log(f'止盈 {len(TP_GRID)} 档 +{TP_GRID[0]:.0%}~+{TP_GRID[-1]:.0%}  '
    f'止损 {len(SL_GRID)} 档 -{SL_GRID[0]:.0%}~-{SL_GRID[-1]:.0%}  '
    f'共 {len(TP_GRID)*len(SL_GRID)} 格')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['high'].values.astype(float), g['low'].values.astype(float),
                 g['close'].values.astype(float), g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = pd.Series(l).rolling(121, min_periods=1).min().values
    lo_i = []
    i = 120
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 30
        else:
            i += 1
    pairs = []
    for i in lo_i:
        if i + CONF >= n: continue
        for j in range(i + 1, min(i + CONF + 1, n)):
            if c[j] >= l[i] * 1.03:
                pairs.append((i, j)); break
    return pairs


# pairs 只算一次（V05/V09 固定，与 TP/SL 无关）
SYMS = []
NOBS = 0
for sym, (h, l, c, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    keep = [(li, j) for li, j in pairs if j - VW >= 0 and j + HOLD < n]
    if not keep: continue
    j_a = np.array([p[1] for p in keep], dtype=np.int64)
    li_a = np.array([p[0] for p in keep], dtype=np.int64)
    SYMS.append((h, l, c, t, j_a, l[li_a].astype(float), c[j_a].astype(float)))
    NOBS += len(keep)
log(f'pairs 预处理完成  币 {len(SYMS)}  入场点 {NOBS:,}  {time.time()-t0:.0f}s')

# 测试段界：按入场月份 75/25
all_t = np.concatenate([s[3][s[4]] for s in SYMS])
ym_all = pd.to_datetime(all_t, unit='ms', utc=True).strftime('%Y-%m')
months = sorted(np.unique(ym_all))
ntr = max(1, int(len(months) * 0.75))
test_months = set(months[ntr:])
log(f'月份 {len(months)}  训练 {months[0]}~{months[ntr-1]}  测试 {months[ntr]}~{months[-1]}')


def eval_cell(TP, SL):
    ent_t, ent_ret, ent_tp, ent_close = [], [], [], []
    for (h, l, c, t, j_a, lo_a, e_a) in SYMS:
        m = len(j_a)
        tp_px = e_a * (1.0 + TP)
        sl_px = e_a * (1.0 - SL)
        for k in range(m):
            j = int(j_a[k])
            seg_h = h[j + 1:j + 1 + HOLD]
            seg_l = l[j + 1:j + 1 + HOLD]
            hit_t = seg_h >= tp_px[k]
            hit_s = seg_l <= sl_px[k]
            it = int(np.argmax(hit_t)) if hit_t.any() else -1
            il = int(np.argmax(hit_s)) if hit_s.any() else -1
            if it >= 0 and il >= 0:
                if it < il: ret = tp_px[k] / e_a[k] - 1; ok = 1; ie = it
                else:       ret = sl_px[k] / e_a[k] - 1; ok = 0; ie = il
            elif it >= 0:
                ret = tp_px[k] / e_a[k] - 1; ok = 1; ie = it
            elif il >= 0:
                ret = sl_px[k] / e_a[k] - 1; ok = 0; ie = il
            else:
                ret = c[j + HOLD] / e_a[k] - 1; ok = 0; ie = HOLD - 1
            ent_t.append(t[j]); ent_ret.append(ret); ent_tp.append(ok); ent_close.append(t[j + 1 + ie])

    ent_t = np.array(ent_t, dtype='int64')
    ent_ret = np.array(ent_ret, dtype=float)
    ent_tp = np.array(ent_tp, dtype=np.int8)
    ent_close = np.array(ent_close, dtype='int64')

    # 前视修法：按 t_enter 排序，只有 t_close <= t_enter 的笔才进 h3
    order = np.argsort(ent_t, kind='stable')
    et = ent_t[order]; er = ent_ret[order]; eok = ent_tp[order]; ec = ent_close[order]
    hist, heap = [], []
    h3a = np.empty(len(et), dtype=float)
    for i in range(len(et)):
        while heap and heap[0][0] <= et[i]:
            _, v = heapq.heappop(heap); hist.append(v)
        h3a[i] = float(np.mean(hist[-3:])) if hist else -1.0
        heapq.heappush(heap, (int(ec[i]), int(eok[i])))

    ymi = pd.to_datetime(et, unit='ms', utc=True).strftime('%Y-%m')
    te_mask = np.array([m in test_months for m in ymi])
    ret_net = er - FEE

    def stat(m):
        if m.sum() == 0: return (0, np.nan, np.nan, np.nan)
        r = ret_net[m]
        return (int(m.sum()), float(r.mean() * 100), float((r > 0).mean() * 100),
                float(eok[m].mean() * 100))

    a = stat(te_mask)
    b = stat(te_mask & (h3a > 0.1) & (h3a < 0.9))
    return a, b


rows = []
cell = 0
total = len(TP_GRID) * len(SL_GRID)
for TP in TP_GRID:
    for SL in SL_GRID:
        cell += 1
        a, b = eval_cell(TP, SL)
        rows.append((TP, SL, *a, *b))
        if cell % 20 == 0 or cell == total:
            log(f'  [{cell}/{total}] {time.time()-t0:.0f}s  TP={TP:.0%} SL=-{SL:.0%}  '
                f'all n={a[0]:,} mean={a[1]:+.2f}%  h3 n={b[0]:,} mean={b[1]:+.2f}%')

df = pd.DataFrame(rows, columns=['tp', 'sl', 'n_all', 'mean_all', 'win_all', 'tpr_all',
                                 'n_h3', 'mean_h3', 'win_h3', 'tpr_h3'])
df['p_star'] = (df['sl'] + FEE) / (df['tp'] + df['sl'])
df.to_csv(os.path.join(OUT, 'tpsl_grid.csv'), index=False, encoding='utf-8-sig')

log('')
log('=' * 100)
log('【按 mean_h3 排序 Top20（样本>=100）】')
log(f"{'TP':>6}{'SL':>6}{'n_h3':>8}{'均值%':>9}{'胜率%':>8}{'TP率%':>8}{'p*%':>8}")
for _, r in df[df['n_h3'] >= 100].sort_values('mean_h3', ascending=False).head(20).iterrows():
    log(f"{r['tp']:>6.0%}{-r['sl']:>6.0%}{int(r['n_h3']):>8,}{r['mean_h3']:>9.2f}"
        f"{r['win_h3']:>8.1f}{r['tpr_h3']:>8.1f}{r['p_star']*100:>8.1f}")

log('')
log('【按 mean_all 排序 Top10】')
log(f"{'TP':>6}{'SL':>6}{'n_all':>9}{'均值%':>9}{'胜率%':>8}{'TP率%':>8}")
for _, r in df.sort_values('mean_all', ascending=False).head(10).iterrows():
    log(f"{r['tp']:>6.0%}{-r['sl']:>6.0%}{int(r['n_all']):>9,}{r['mean_all']:>9.2f}"
        f"{r['win_all']:>8.1f}{r['tpr_all']:>8.1f}")

log('')
log('【分布】')
log(f'  mean_all > 0 的格 {int((df["mean_all"] > 0).sum())} / {total}')
log(f'  mean_h3  > 0 的格 {int((df["mean_h3"] > 0).sum())} / {total}')
log(f'  mean_all 中位 {df["mean_all"].median():+.3f}%  最好 {df["mean_all"].max():+.3f}%  最差 {df["mean_all"].min():+.3f}%')
log(f'  mean_h3  中位 {df["mean_h3"].median():+.3f}%  最好 {df["mean_h3"].max():+.3f}%  最差 {df["mean_h3"].min():+.3f}%')

log('')
log('【热力图（mean_all，行=止盈，列=止损）】')
hd = 'TP/SL'.rjust(8) + ''.join(f'{-s:>7.0%}' for s in SL_GRID)
log(hd)
for TP in TP_GRID:
    line = f'{TP:>8.0%}'
    for SL in SL_GRID:
        v = float(df[(df['tp'] == TP) & (df['sl'] == SL)]['mean_all'].iloc[0])
        line += f'{v:>7.2f}'
    log(line)

log('')
log('【判读】')
log('  · 看「正格是否连成一片」；孤立 1~3 格是噪声')
log('  · 最优点若在网格边界 ⇒ 范围不够')
log('  · p* = (SL+费)/(TP+SL)：期望为正需胜率 > p*')

out = os.path.join(OUT, 'tpsl_grid_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'CSV 已写 {os.path.join(OUT, "tpsl_grid.csv")}')
log(f'TOTAL {time.time()-t0:.0f}s')
