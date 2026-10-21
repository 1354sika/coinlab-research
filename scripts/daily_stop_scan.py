# -*- coding: utf-8 -*-
"""日频停手扫描：每天从当日起始资金出发，累计到 X 倍就停手，次日以新资金复利。
不看单笔输赢，只看"赢的那天锁多少、输的那天放多少"。
"""
import os, sys
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
HARG  = int(sys.argv[8]) if len(sys.argv) > 8 else 1440   # 0 = 不限时间（机器全天在线）
DOWN  = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0     # 日亏熔断：当日资金跌到 1-DOWN 就停手；0=不熔断

L = []
def p(s=''):
    L.append(str(s)); print(s, flush=True)

p('=' * 108)
p('DAILY STOP SCAN  file=%s  TP=%.1f%%  SL=%.2f%%  COST=%.2f%%  POOL=%d'
  % (fname, TP*100, SL*100, COST*100, POOL))
p('=' * 108)

f = os.path.join(DATA, fname)
with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
    hdr = [c.strip().replace('\ufeff', '') for c in fh.readline().split(',')]
need = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(f, usecols=[c for c in need if c in hdr])
p('rows = %d  symbols = %d' % (len(df), df.symbol.nunique()))

df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
df = df.sort_values(['symbol', 'open_time'])

# 时间框架推断
t0 = df.open_time.iloc[0]
tf = '5m' if fname.find('5m') >= 0 else ('1d' if fname.find('1d') >= 0 else ('1h' if fname.find('1h') >= 0 else '1m'))
H = HARG if HARG > 0 else 10**12
p('timeframe = %s   hold cap = %s' % (tf, ('UNLIMITED' if HARG == 0 else '%d bars' % HARG)))

trades = []
for sym, b in df.groupby('symbol', sort=False):
    b = b.sort_values('open_time')
    n = len(b)
    if n < 30:
        continue
    c = b.close.to_numpy(np.float64); h = b.high.to_numpy(np.float64)
    lo = b.low.to_numpy(np.float64); v = b.volume.to_numpy(np.float64)
    ts = b.open_time.to_numpy(np.int64)
    ret24 = np.full(n, np.nan)
    ret24[24:] = c[24:] / c[:-24] - 1.0
    vma = pd.Series(v).rolling(24, min_periods=12).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volr = np.where(vma > 0, v / vma, 1.0)
    sig = np.isfinite(ret24) & (ret24 <= -DROP) & (volr >= VOLR)
    for i in np.where(sig)[0]:
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(i + H, n)
        if end <= i + 1:
            continue
        hh = h[i+1:end]; ll = lo[i+1:end]
        tp_hit = np.where(hh >= entry * (1 + TP))[0]
        sl_hit = np.where(ll <= entry * (1 - SL))[0]
        i_tp = tp_hit[0] if len(tp_hit) else 10**9
        i_sl = sl_hit[0] if len(sl_hit) else 10**9
        if i_tp == 10**9 and i_sl == 10**9:
            if HARG > 0 and end - i - 1 < H:
                continue
            r = c[end-1] / entry - 1.0; j = end - 1
            reason = 'open_end' if HARG == 0 else 'timeout'
        elif i_tp <= i_sl:
            r = TP; j = i + 1 + i_tp; reason = 'tp'
        else:
            r = -SL; j = i + 1 + i_sl; reason = 'sl'
        trades.append((int(ts[i]), sym, float(r - COST), int(ts[j] - ts[i]), reason))

if not trades:
    p('NO SIGNAL'); sys.exit(0)

T = pd.DataFrame(trades, columns=['t_in', 'sym', 'r', 'hold_ms', 'reason'])
T['out_ms'] = T.t_in + T.hold_ms
T = T.sort_values('t_in').reset_index(drop=True)
p('signals = %d   span = %s ~ %s' % (
    len(T),
    pd.to_datetime(T.t_in.min(), unit='ms').strftime('%Y-%m-%d'),
    pd.to_datetime(T.t_in.max(), unit='ms').strftime('%Y-%m-%d')))

# 池 N 并发：占满则跳过
slots = []
taken = []
for _, row in T.iterrows():
    slots = [s for s in slots if s[0] > row.t_in]
    if len(slots) >= POOL:
        continue
    slots.append((row.out_ms, row.r))
    taken.append(row)
K = pd.DataFrame(taken).sort_values('t_in').reset_index(drop=True)
K['day'] = pd.to_datetime(K.t_in, unit='ms').dt.strftime('%Y-%m-%d')
p('pool took = %d  over %d natural days' % (len(K), K.day.nunique()))
p('')
p('--- single-trade (pool 内的真实成交笔) ---')
p('mean   %+.4f%%' % (K.r.mean()*100))
p('median %+.4f%%' % (K.r.median()*100))
p('win    %.2f%%   (%d / %d)' % ((K.r > 0).mean()*100, (K.r > 0).sum(), len(K)))
if (K.r > 0).any(): p('avg win  %+.3f%%' % (K.r[K.r > 0].mean()*100))
if (K.r <= 0).any(): p('avg loss %+.3f%%' % (K.r[K.r <= 0].mean()*100))
p('hold   median %.1f h' % (K.hold_ms.median()/3600000.0))
p('exit mix: ' + '  '.join('%s=%d' % (k, v) for k, v in K.reason.value_counts().items()))
p('')

# 逐日基础：每笔对总资金贡献 r/POOL
def simulate(X):
    """X = 当日资金到 X 倍就停手；X=inf 表示不停"""
    total = 1.0
    peak = 1.0
    mdd = 0.0
    hit_days = 0
    day_list = []
    for day, g in K.groupby('day', sort=True):
        q = 1.0
        nused = 0
        for r in g.r.to_numpy():
            q *= (1.0 + r / POOL)
            nused += 1
            if q >= X:
                break
            if DOWN > 0 and q <= 1.0 - DOWN:
                break
        day_list.append((day, q, nused))
        total *= q
        peak = max(peak, total)
        mdd = min(mdd, total / peak - 1.0)
        if q >= X:
            hit_days += 1
    D = pd.DataFrame(day_list, columns=['day', 'q', 'n'])
    return dict(total=total, mdd=mdd, hit=hit_days, days=len(D),
                med_n=D[D.q >= X].n.median() if hit_days else np.nan,
                med_q=D.q.median(), win_days=(D.q > 1).mean() * 100)

p('%-10s %14s %10s %10s %10s %10s' % ('stop X', 'final(134d)', 'maxDD', 'hit_days', 'med_n', 'day_win%'))
p('-' * 108)
for X in [1.03, 1.05, 1.08, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 2.00, np.inf]:
    s = simulate(X)
    label = 'none' if np.isinf(X) else '%.2fx' % X
    p('%-10s %14.4f %9.1f%% %10d %10s %9.1f%%' % (
        label, s['total'], s['mdd']*100, s['hit'],
        '-' if np.isnan(s['med_n']) else '%.0f' % s['med_n'], s['win_days']))

p('')
p('读法：stop X = 当日资金到 X 倍就停手，次日以新资金继续。')
p('      final = 134 天后的总资金倍数（本金 1.0 起）；maxDD = 期间最大回撤。')
p('      hit_days = 有多少天真的碰到了 X；med_n = 碰到 X 的天平均用了几笔。')
p('')
p('--- honest boundary -----------------------------------------------')
p('1. 逐笔按 r/POOL 线性计入当日资金（等权 1/20），未模拟逐仓保证金与资金费率。')
p('2. 零滑点；TP/SL 按触价即成交；成本 %.2f%% 已含一次往返。' % (COST*100))
p('3. 池 20 并发：信号来得比持仓快时会跳过，这就是真实约束。')
p('4. 单一样本（%s），含样本内参数；日切按 UTC。' % fname)

out = os.path.join(EVI, 'daily_stop_%s_tp%d_sl%d_h%d_d%d.txt' % (
    tf, int(round(TP*1000)), int(round(SL*1000)), HARG, int(round(DOWN*1000))))
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
p('')
p('written %s' % out)
