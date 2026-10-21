# -*- coding: utf-8 -*-
"""gambler_v05_sweep · 只扫 V05（低点窗口）：4h→24h，步长 5 分钟，共 241 档

固定参数（与 gambler_canonical 完全一致）：
  V09=2880min 确认窗口 | V10=3% 入场触发 | V11=high | V12=创新低作废
  V14=低点×1.30 | V15=低点×0.95 | V16=1440min 持有 | V17=双触算止损 | V18=收盘价
  V19=0.50%/笔 | V20=100 | V21=10% | V22=无筛选 | V23=按入场时间 75/25

校验点：V05=240 那一档的测试段笔数应与 gambler_canonical --data D01 一致（2,926）。

用法：python -X utf8 gambler_v05_sweep.py [--data D01] [--lo 240] [--hi 1440] [--step 5]
"""
import os, time, argparse
import numpy as np
import pandas as pd
from data_registry import resolve

OUT = r'D:\中信证券\coinlab\evidence\cycle'

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='D01', help='D 编号（见数据资产编号表）')
ap.add_argument('--lo', type=int, default=240, help='V05 起点（分钟）')
ap.add_argument('--hi', type=int, default=1440, help='V05 终点（分钟）')
ap.add_argument('--step', type=int, default=5, help='V05 步长（分钟）')
args = ap.parse_args()
DCODE, DATA_PATH, GRAN, DDESC = resolve(args.data)

CONF_MIN, HOLD_MIN = 2880, 1440          # [V09] [V16]
ENTRY_RISE = 0.03                         # [V10]
TP_MULT, SL_MULT = 1.30, 0.95             # [V14] [V15]
FEE = 0.005                               # [V19]
START, POS = 100.0, 0.10                  # [V20] [V21]
MAXSYM = int(os.environ.get('SYMS', '60'))  # [V03]
CONF = CONF_MIN // GRAN
HOLD = HOLD_MIN // GRAN

logbuf = []


def log(s):
    print(s, flush=True); logbuf.append(s)


N = len(range(args.lo, args.hi + 1, args.step))
t0 = time.time()
log('=' * 108)
log(f'gambler_v05_sweep · 只扫 V05：{args.lo}~{args.hi} 分钟，步长 {args.step}（共 {N} 档）')
log(f'数据 {DCODE} = {DDESC} | 粒度 {GRAN}m | 换算 V05根=V05分钟÷{GRAN}')
log(f'固定 V09={CONF_MIN}min({CONF}根) V10={ENTRY_RISE:.0%} V14=低点×{TP_MULT} V15=低点×{SL_MULT} V16={HOLD_MIN}min({HOLD}根)')
log('=' * 108)

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close']
RAW = {}
for ch in pd.read_csv(DATA_PATH, usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        RAW.setdefault(sym, []).append(g)
    if len(RAW) >= MAXSYM:
        break

RAWS = {}
_ot_min = None; _ot_max = None
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000:
        continue
    o = g['open_time'].values.astype('int64')
    a, b = int(o.min()), int(o.max())
    _ot_min = a if _ot_min is None else min(_ot_min, a)
    _ot_max = b if _ot_max is None else max(_ot_max, b)
    RAWS[sym] = (g['high'].values.astype(float), g['low'].values.astype(float),
                 g['close'].values.astype(float), o)
RAW = None
YEARS = max(0.01, (_ot_max - _ot_min) / (365.25 * 86400000.0))
log(f'币 {len(RAWS)}  数据跨度 {YEARS:.2f} 年  预加载 {time.time()-t0:.0f}s')


def build_rows(vw):
    """按 V05=vw（根）重建全部入场点。V05 变 → 低点集合变 → 入场点全变。"""
    step = max(5, vw // 8)
    rows = []
    for sym, (h, l, c, ot) in RAWS.items():
        n = len(c)
        if n < vw + CONF + 2:
            continue
        lo_w = pd.Series(l).rolling(vw, min_periods=vw).min().values
        cand = np.where(l[vw:n - CONF - 1] <= lo_w[vw:n - CONF - 1])[0] + vw
        # 与 canonical 的 i += max(5, VW//8) 去重语义保持一致
        sel = []
        last = -(10 ** 12)
        for i in cand:
            if i - last >= step:
                sel.append(i); last = i
        for lo_i in sel:
            lo = l[lo_i]
            if lo <= 0:
                continue
            trig = lo * (1 + ENTRY_RISE)
            j = lo_i + 1
            end = min(n - 1, lo_i + CONF)
            hit = -1
            while j <= end:
                if h[j] >= trig:
                    hit = j; break
                if l[j] < lo:
                    break
                j += 1
            if hit > 0:
                rows.append((ot[hit], sym, lo, trig, hit, n))
    rows.sort(key=lambda x: x[0])
    return rows


def backtest(rows):
    out = []
    cache = {}
    for (t_entry, sym, lo, entry, hit, n) in rows:
        if sym not in cache:
            cache[sym] = RAWS[sym]
        h, l, c, ot = cache[sym]
        tp = lo * TP_MULT; sl = lo * SL_MULT
        r = None; kind = 'to'
        kmax = min(hit + HOLD + 1, n)
        for k in range(hit + 1, kmax):
            ht = h[k] >= tp; hs = l[k] <= sl
            if ht and hs:
                r = sl / entry - 1; kind = 'sl'; break
            if ht:
                r = tp / entry - 1; kind = 'tp'; break
            if hs:
                r = sl / entry - 1; kind = 'sl'; break
        if r is None:
            kk = min(hit + HOLD, n - 1)
            r = c[kk] / entry - 1; kind = 'to'
        out.append((t_entry, r - FEE, kind))
    return out


recs = []
for vw_min in range(args.lo, args.hi + 1, args.step):
    vw = max(1, vw_min // GRAN)          # 【单位换算】分钟 → 根（GRAN=1 时等价）
    t1 = time.time()
    rows = build_rows(vw)
    if not rows:
        recs.append(dict(vw=vw_min, vw_roots=vw, n_entry=0))
        log(f'V05={vw_min:5d}min({vw}根)  无入场点')
        continue
    ts = np.array([r[0] for r in rows])
    cut = ts[int(len(ts) * 0.75)]          # [V23] 按入场时间 75/25
    tr = ts < cut
    trades = backtest(rows)
    rets = np.array([t[1] for t in trades])
    tr_r = rets[tr]; te_r = rets[~tr]
    kinds = {}
    for i, t in enumerate(trades):
        if not tr[i]:
            kinds[t[2]] = kinds.get(t[2], 0) + 1
    cap = START; peak = START; mdd = 0.0
    for r in te_r:
        cap *= (1 + POS * r); peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak)
    tot = len(te_r) or 1
    per_year = len(rows) / YEARS
    rec = dict(vw=vw_min, vw_roots=vw, n_entry=len(rows), per_year=per_year, n_train=int(tr.sum()), n_test=int((~tr).sum()),
               mean_tr=float(tr_r.mean()) if len(tr_r) else np.nan,
               mean_te=float(te_r.mean()) if len(te_r) else np.nan,
               med_te=float(np.median(te_r)) if len(te_r) else np.nan,
               win_te=float((te_r > 0).mean()) if len(te_r) else np.nan,
               final=float(cap), mdd=float(mdd),
               p_tp=kinds.get('tp', 0) / tot, p_sl=kinds.get('sl', 0) / tot, p_to=kinds.get('to', 0) / tot)
    recs.append(rec)
    log(f'V05={vw_min:5d}min({vw:>4d}根)  入场 {len(rows):>7,}（每年 {per_year:>6,.1f} 次）  测试 {int((~tr).sum()):>6,}  '
        f'均值 {rec["mean_te"]:+.3%}  胜率 {rec["win_te"]:5.1%}  {START:.0f}→{cap:>10,.2f}  '
        f'回撤 {mdd:5.1%}  TP{rec["p_tp"]:5.1%} SL{rec["p_sl"]:5.1%} TO{rec["p_to"]:5.1%}  ({time.time()-t1:.1f}s)')

df = pd.DataFrame(recs)
csv = os.path.join(OUT, f'v05_sweep_{DCODE}_{args.lo}_{args.hi}.csv')
df.to_csv(csv, index=False, encoding='utf-8-sig')

log('\n' + '=' * 108)
log('【摘要】按测试段单笔均值排序 Top15')
log('=' * 108)
d = df.dropna(subset=['mean_te']).sort_values('mean_te', ascending=False)
for _, r in d.head(15).iterrows():
    log(f'  V05={int(r.vw):5d}min({int(r.vw)//60:>2}h{int(r.vw)%60:02d}m)  入场 {int(r.n_entry):>6,}（每年 {r.per_year:>6,.1f} 次）  '
        f'测试 {int(r.n_test):>6,} 笔  均值 {r.mean_te:+.3%}  胜率 {r.win_te:.1%}  {START:.0f}→{r.final:,.2f}  回撤 {r.mdd:.1%}')

log('\n【分布】')
if len(d):
    log(f'  均值>0 的档数 {int((d.mean_te > 0).sum())} / {len(d)}')
    log(f'  均值中位 {d.mean_te.median():+.3%}  最好 {d.mean_te.max():+.3%}  最差 {d.mean_te.min():+.3%}')
    log(f'  样本<100笔的档数 {int((d.n_test < 100).sum())}（这些档结论不可信）')

log(f'\nCSV 已写 {csv}')
out = os.path.join(OUT, f'v05_sweep_{DCODE}_{args.lo}_{args.hi}_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'摘要已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
