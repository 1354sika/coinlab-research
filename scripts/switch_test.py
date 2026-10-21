# -*- coding: utf-8 -*-
"""分层切换实验：前半段用快确认（1分钟），本金到达 SWITCH 倍后切慢确认（4小时止损）。
对照：全程快确认 / 全程慢确认 / 翻倍后切换。
"""
import os, sys
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'

fname  = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
TP     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
SL     = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
COST   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0015
POOL   = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DROP   = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
VOLR   = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
SWITCH = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

p('=' * 104)
p('LAYER SWITCH  file=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d  switch=%.2fx'
  % (fname, TP*100, SL*100, COST*100, POOL, SWITCH))
p('A=快确认(止损1m/止盈1m)   B=慢确认(止损240m/止盈5m)')
p('=' * 104)

f = os.path.join(DATA, fname)
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = [c.strip().replace('\ufeff', '') for c in fh.readline().split(',')]
need = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(f, usecols=[c for c in need if c in hdr])
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
df = df.sort_values(['symbol', 'open_time'])
tf = '5m' if '5m' in fname else ('1d' if '1d' in fname else ('1h' if '1h' in fname else '1m'))
BPM = {'1m': 1, '5m': 5, '1h': 60, '1d': 1440}[tf]
p('timeframe=%s rows=%d symbols=%d' % (tf, len(df), df.symbol.nunique()))


def exit_of(c, ts, i, step_sl, step_tp):
    entry = c[i]
    path = c[i+1:]
    n = len(path)
    if n < 2:
        return None
    idx = np.arange(n)
    h_sl = np.where((idx % step_sl == 0) & (path <= entry * (1 - SL)))[0]
    h_tp = np.where((idx % step_tp == 0) & (path >= entry * (1 + TP)))[0]
    i_sl = int(h_sl[0]) if len(h_sl) else 10**9
    i_tp = int(h_tp[0]) if len(h_tp) else 10**9
    if i_sl == 10**9 and i_tp == 10**9:
        r = path[-1] / entry - 1.0; j = n - 1
    elif i_tp <= i_sl:
        r = TP; j = i_tp
    else:
        r = -SL; j = i_sl
    ex = int(ts[i + 1 + j]) if (i + 1 + j) < len(ts) else int(ts[i])
    return float(r - COST), ex


recs = []   # (t_in, retA, exitA, retB, exitB)
for sym, b in df.groupby('symbol', sort=False):
    b = b.sort_values('open_time')
    n = len(b)
    if n < 30:
        continue
    c = b.close.to_numpy(np.float64)
    v = b.volume.to_numpy(np.float64)
    ts = b.open_time.to_numpy(np.int64)
    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR)
    for i in np.where(sig)[0]:
        a = exit_of(c, ts, i, 1, 1)                       # 快
        bb = exit_of(c, ts, i, 240 * BPM, 5 * BPM)        # 慢
        if a is None or bb is None:
            continue
        recs.append((int(ts[i]), a[0], a[1], bb[0], bb[1]))
recs.sort(key=lambda x: x[0])
p('signals = %d' % len(recs))


def run(mode):
    """mode: 'A' 全程快 / 'B' 全程慢 / 'switch' 达 SWITCH 后切慢"""
    slots = []           # exit_ms 列表
    eq = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; switched_at = None
    for t_in, rA, eA, rB, eB in recs:
        slots = [s for s in slots if s > t_in]
        if len(slots) >= POOL:
            continue
        if mode == 'A':
            r, ex = rA, eA
        elif mode == 'B':
            r, ex = rB, eB
        else:
            if eq >= SWITCH:
                if switched_at is None:
                    switched_at = t_in
                r, ex = rB, eB
            else:
                r, ex = rA, eA
        eq *= (1.0 + r / POOL)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
        slots.append(ex); ntr += 1
    return dict(eq=eq, mdd=mdd * 100, n=ntr,
                sw=(pd.to_datetime(switched_at, unit='ms').strftime('%Y-%m-%d')
                    if switched_at else '-'))


p('')
p('%-22s %12s %10s %8s %12s' % ('模式', '终值', '回撤', '笔数', '切换日'))
p('-' * 104)
for mode, label in [('A', '全程快确认(1m)'), ('B', '全程慢确认(240m)'),
                    ('switch', '翻倍(%.1fx)后切慢' % SWITCH)]:
    s = run(mode)
    p('%-22s %12.4f %9.2f%% %8d %12s' % (label, s['eq'], s['mdd'], s['n'], s['sw']))

p('')
p('--- honest boundary ---')
p('1. 切换判定用「当前 equity 是否已达 SWITCH 倍」，切换后不再回退。')
p('2. 两套出场都预先算好，切换只影响选哪一套，不影响信号本身。')
p('3. 零滑点；成本 %.2f%% 往返；池 %d 并发。' % (COST*100, POOL))
p('4. 单一样本 %s，含样本内。' % fname)

out = os.path.join(EVI, 'switch_%s.txt' % tf)
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
