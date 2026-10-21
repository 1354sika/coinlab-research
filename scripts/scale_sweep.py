# -*- coding: utf-8 -*-
"""时间尺度扫描：止损和止盈各自用不同的确认尺度（只看该尺度收盘价）。
尺度 = 分钟数（1/5/15/60/240），在 1m 数据上等价于每 N 根确认一次。
"""
import os, sys, itertools
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'

fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
TP    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
SL    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
COST  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
POOL  = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DROP  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
VOLR  = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
SCALES = [int(x) for x in (sys.argv[8].split(',') if len(sys.argv) > 8 else ['1','5','15','60','240'])]

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

p('=' * 108)
p('SCALE SWEEP  file=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d'
  % (fname, TP*100, SL*100, COST*100, POOL))
p('scales(分钟) = %s' % SCALES)
p('含义：止损只看「每 N 分钟那根的收盘」是否跌破；止盈同理。盘中影线不算。')
p('=' * 108)

f = os.path.join(DATA, fname)
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = [c.strip().replace('\ufeff', '') for c in fh.readline().split(',')]
need = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(f, usecols=[c for c in need if c in hdr])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
df = df.sort_values(['symbol', 'open_time'])
tf = '5m' if '5m' in fname else ('1d' if '1d' in fname else ('1h' if '1h' in fname else '1m'))
BARS_PER_MIN = {'1m': 1, '5m': 0.2, '1h': 1/60.0, '1d': 1/1440.0}[tf]
p('timeframe=%s  rows=%d  symbols=%d' % (tf, len(df), df.symbol.nunique()))

# ---- 先收集所有信号点（入场索引 + 该币路径） ----
paths = {}
sigs = []
for sym, b in df.groupby('symbol', sort=False):
    b = b.sort_values('open_time')
    n = len(b)
    if n < 30:
        continue
    c = b.close.to_numpy(np.float64)
    v = b.volume.to_numpy(np.float64)
    ts = b.open_time.to_numpy(np.int64)
    paths[sym] = (c, ts)
    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR)
    for i in np.where(sig)[0]:
        if i + 2 < n:
            sigs.append((int(ts[i]), sym, i))
p('signals = %d' % len(sigs))

# ---- 尺度网格 ----
rows = []
for s_sl, s_tp in itertools.product(SCALES, repeat=2):
    step_sl = max(1, int(round(s_sl * BARS_PER_MIN)))
    step_tp = max(1, int(round(s_tp * BARS_PER_MIN)))
    step = min(step_sl, step_tp)
    out = []
    for t_ms, sym, i in sigs:
        c, ts_arr = paths[sym]
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = c[i+1:]
        n = len(path)
        if n < 2:
            continue
        idx = np.arange(n)
        h_sl = np.where((idx % step_sl == 0) & (path <= entry * (1 - SL)))[0]
        h_tp = np.where((idx % step_tp == 0) & (path >= entry * (1 + TP)))[0]
        i_sl = int(h_sl[0]) if len(h_sl) else 10**9
        i_tp = int(h_tp[0]) if len(h_tp) else 10**9
        if i_sl == 10**9 and i_tp == 10**9:
            r = path[-1] / entry - 1.0; reason = 'open_end'; j = n - 1
        elif i_tp <= i_sl:
            r = TP; reason = 'tp'; j = i_tp
        else:
            r = -SL; reason = 'sl'; j = i_sl
        exit_ms = int(ts_arr[i + 1 + j]) if (i + 1 + j) < len(ts_arr) else t_ms
        out.append((t_ms, sym, float(r - COST), reason, exit_ms))
    if not out:
        continue
    # 池 N
    out.sort(key=lambda x: x[0])
    slots, keep = [], []
    for t_in, sym, r, reason, t_out in out:
        slots = [s for s in slots if s > t_in]
        if len(slots) >= POOL:
            continue
        slots.append(t_out)
        keep.append((t_in, sym, r, reason))
    if not keep:
        continue
    K = pd.DataFrame(keep, columns=['t_in', 'sym', 'r', 'reason'])
    eq = 1.0; peak = 1.0; mdd = 0.0
    for r in K.r.to_numpy():
        eq *= (1.0 + r / POOL)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    win = (K.r > 0).mean() * 100
    rows.append(dict(sl_scale=s_sl, tp_scale=s_tp, n=len(K),
                     win=win, mean=K.r.mean()*100,
                     eq=eq, mdd=mdd*100,
                     n_sl=(K.reason == 'sl').sum(), n_tp=(K.reason == 'tp').sum()))

R = pd.DataFrame(rows)
p('')
p('=' * 108)
p('%-9s %9s %8s %9s %12s %9s %8s %8s' % ('SL尺度', 'TP尺度', '笔数', '胜率', '单笔均值', '终值', '回撤', '止损数'))
p('-' * 108)
for _, r in R.sort_values('eq', ascending=False).iterrows():
    p('%-9s %9s %8d %8.2f%% %11.4f%% %9.4f %8.2f%% %8d' % (
        '%.0fm' % r['sl_scale'], '%.0fm' % r['tp_scale'], r['n'], r['win'], r['mean'], r['eq'], r['mdd'], r['n_sl']))
p('-' * 108)
best = R.sort_values('eq', ascending=False).iloc[0]
p('★ 最优：止损 %.0f 分钟确认 / 止盈 %.0f 分钟确认 → 终值 %.4f  回撤 %.2f%%  胜率 %.2f%%'
  % (best['sl_scale'], best['tp_scale'], best['eq'], best['mdd'], best['win']))
base = R[(R.sl_scale == SCALES[0]) & (R.tp_scale == SCALES[0])]
if len(base):
    b = base.iloc[0]
    p('  基线（都用 %.0f 分钟）= 终值 %.4f  回撤 %.2f%%  止损数 %d' % (b['sl_scale'], b['eq'], b['mdd'], b['n_sl']))
p('')
p('--- honest boundary ---')
p('1. 「N 分钟确认」= 只看每 N 分钟那根的收盘价，不算盘中影线；这是对实盘「收盘才决策」的近似。')
p('2. 零滑点；成本 %.2f%% 往返已扣。' % (COST*100))
p('3. 池 %d 并发，信号冲突按时间先后丢弃。' % POOL)
p('4. 单一样本 %s，含样本内参数。' % fname)

out = os.path.join(EVI, 'scale_sweep_%s.txt' % tf)
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
