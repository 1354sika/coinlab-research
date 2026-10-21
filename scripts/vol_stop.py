# -*- coding: utf-8 -*-
"""波动率止损：止损宽度 = k × σ4h（入场时最近240根1m收益标准差 × sqrt(240)）。
止盈固定 50% 用 5 分钟确认；止损用 240 分钟确认（t2 扫出的最优尺度）。
对照基线：固定 3.5% 止损，同样确认尺度。
"""
import os, sys
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'

fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
TP    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
COST  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0015
POOL  = int(sys.argv[4]) if len(sys.argv) > 4 else 20
DROP  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
VOLR  = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
KS    = [float(x) for x in (sys.argv[7].split(',') if len(sys.argv) > 7 else ['0','0.75','1.0','1.5','2.0','2.5','3.0'])]
SL_CONFIRM = 240   # 止损看每 240 分钟那根的收盘
TP_CONFIRM = 5     # 止盈看每 5 分钟那根的收盘

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

p('=' * 106)
p('VOL-STOP SWEEP  file=%s  TP=%.1f%%  COST=%.2f%%  POOL=%d' % (fname, TP*100, COST*100, POOL))
p('止损宽度 = k × σ4h（入场时最近240根1m收益std × sqrt(240)），截断在 [1%%, 50%%]')
p('确认尺度：止损 240 分钟 / 止盈 5 分钟')
p('k=0 表示固定 3.5%%（对照基线）')
p('=' * 106)

f = os.path.join(DATA, fname)
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = [c.strip().replace('\ufeff', '') for c in fh.readline().split(',')]
need = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(f, usecols=[c for c in need if c in hdr])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
df = df.sort_values(['symbol', 'open_time'])
tf = '5m' if '5m' in fname else ('1d' if '1d' in fname else ('1h' if '1h' in fname else '1m'))
BPM = {'1m': 1, '5m': 5, '1h': 60, '1d': 1440}[tf]
p('timeframe=%s  rows=%d  symbols=%d' % (tf, len(df), df.symbol.nunique()))

sigs = []   # (t_ms, sym, i, entry, sd4h)
for sym, b in df.groupby('symbol', sort=False):
    b = b.sort_values('open_time')
    n = len(b)
    if n < 300:
        continue
    c = b.close.to_numpy(np.float64)
    v = b.volume.to_numpy(np.float64)
    ts = b.open_time.to_numpy(np.int64)
    ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1.0
    sd = pd.Series(ret).rolling(240, min_periods=120).std().to_numpy()
    sd4h = sd * np.sqrt(240.0)
    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR) & np.isfinite(sd4h)
    for i in np.where(sig)[0]:
        if i + 3 < n:
            sigs.append((int(ts[i]), sym, i, float(c[i]), float(sd4h[i]), c, ts))
p('signals = %d' % len(sigs))

p('')
p('%-6s %-11s %9s %8s %9s %10s %9s %8s' % ('k', '止损宽度', '笔数', '胜率', '单笔均值', '终值', '回撤', '止损数'))
p('-' * 106)

step_sl = SL_CONFIRM * BPM
step_tp = TP_CONFIRM * BPM

for k in KS:
    out = []
    for t_ms, sym, i, entry, sd4h, c, ts in sigs:
        if k == 0:
            slw = 0.035
        else:
            slw = min(max(k * sd4h, 0.01), 0.50)
        path = c[i+1:]
        n = len(path)
        if n < 2:
            continue
        idx = np.arange(n)
        h_sl = np.where((idx % step_sl == 0) & (path <= entry * (1 - slw)))[0]
        h_tp = np.where((idx % step_tp == 0) & (path >= entry * (1 + TP)))[0]
        i_sl = int(h_sl[0]) if len(h_sl) else 10**9
        i_tp = int(h_tp[0]) if len(h_tp) else 10**9
        if i_sl == 10**9 and i_tp == 10**9:
            r = path[-1] / entry - 1.0; reason = 'open_end'; j = n - 1
        elif i_tp <= i_sl:
            r = TP; reason = 'tp'; j = i_tp
        else:
            r = -slw; reason = 'sl'; j = i_sl
        exit_ms = int(ts[i + 1 + j]) if (i + 1 + j) < len(ts) else t_ms
        out.append((t_ms, float(r - COST), reason, exit_ms))
    if not out:
        continue
    out.sort(key=lambda x: x[0])
    slots, keep = [], []
    for t_in, r, reason, t_out in out:
        slots = [s for s in slots if s > t_in]
        if len(slots) >= POOL:
            continue
        slots.append(t_out)
        keep.append((r, reason))
    if not keep:
        continue
    K = pd.DataFrame(keep, columns=['r', 'reason'])
    eq = 1.0; peak = 1.0; mdd = 0.0
    for r in K.r.to_numpy():
        eq *= (1.0 + r / POOL)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    med_slw = np.median([min(max(k * s[4], 0.01), 0.50) for s in sigs]) if k else 0.035
    p('%-6s %-11s %9d %7.2f%% %8.4f%% %10.4f %8.2f%% %8d' % (
        'fixed' if k == 0 else '%.2f' % k, '%.2f%%' % (med_slw*100), len(K),
        (K.r > 0).mean()*100, K.r.mean()*100, eq, mdd*100, (K.reason == 'sl').sum()))

p('')
p('--- honest boundary ---')
p('1. σ4h 用入场前 240 根 1m 收益算，无前视；k 越大止损越宽。')
p('2. 止损宽度截断在 [1%, 50%]，防止极端窄/宽。')
p('3. 止损 240 分钟确认、止盈 5 分钟确认（来自 t2 扫描的最优尺度）。')
p('4. 零滑点，成本 %.2f%% 往返；池 %d 并发。' % (COST*100, POOL))
p('5. 单一样本 %s，含样本内。' % fname)

out = os.path.join(EVI, 'volstop_%s.txt' % tf)
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
