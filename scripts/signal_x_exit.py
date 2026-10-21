# -*- coding: utf-8 -*-
"""signal_x_exit · 43 个信号 × 出场网格 × 日目标停手，跑全量历史。

用户的真实口径（本轮已确认）：
  · 目标不是「每天固定 2 笔」，而是「当日累计达到 +2.01%（= 两次 1% 复利）就停手」
    达到后当日不再开新仓，次日以新资金继续；中间可以做任意多笔。
  · 所有信号算法 × 今天试过的出场算法，穷尽组合，跑全量历史。

口径：
  · 信号定义直接沿用 gambler_exhaust.py 的 43 个（r/volr/tbr/trr/vstd/hour）
  · 出场：之后每根 1m 的收盘价触线才成交（不用 high/low，无前视）
  · 成本 0.15% 往返
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI = r'D:\中信证券\coinlab\evidence'
FN = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
COST = 0.0015
MAXH = int(sys.argv[5]) if len(sys.argv) > 5 else 1440   # 最长持仓根数（1m 数据）
STEP = int(sys.argv[6]) if len(sys.argv) > 6 else 60      # 候选点采样间隔（根），60=每小时一个
DAY_TARGET = 1.0201  # 当日累计到 +2.01% 停手
N_SIG_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 12

t0 = time.time()
cols = ['symbol', 'open_time', 'close', 'high', 'low', 'volume', 'quote_volume', 'trades', 'taker_base']
df = pd.read_csv(os.path.join(DATA, FN), usecols=cols)
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
df = df.dropna(subset=['open_time'])
df['open_time'] = df['open_time'].astype('int64')
for c in cols[2:]:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
df = df.dropna()
print('rows %d  %.1fs' % (len(df), time.time() - t0))

# ---- 每币算 43 个信号的特征 ----
feat_frames = []
paths = {}
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    ot = g['open_time'].to_numpy()
    c = g['close'].to_numpy(np.float64)
    h = g['high'].to_numpy(np.float64)
    l = g['low'].to_numpy(np.float64)
    q = g['quote_volume'].to_numpy(np.float64)
    vol = g['volume'].to_numpy(np.float64)
    tr = g['trades'].to_numpy(np.float64)
    tb = g['taker_base'].to_numpy(np.float64)
    n = len(c)
    if n < 3000:
        continue
    paths[sym] = c
    def ret(k):
        r = np.full(n, np.nan); r[k:] = c[k:] / c[:-k] - 1.0; return r
    r1, r5, r15, r60, r240, r1440 = ret(1), ret(5), ret(15), ret(60), ret(240), ret(1440)
    qma = pd.Series(q).rolling(60, min_periods=10).mean().to_numpy()
    volr = np.where(qma > 0, q / np.maximum(qma, 1e-12), np.nan)
    tbr = np.where(vol > 0, tb / np.maximum(vol, 1e-12), np.nan)
    trma = pd.Series(tr).rolling(60, min_periods=10).mean().to_numpy()
    trr = np.where(trma > 0, tr / np.maximum(trma, 1e-12), np.nan)
    vstd = pd.Series(r1).rolling(15, min_periods=5).std().to_numpy()
    vstd1440 = pd.Series(r1).rolling(1440, min_periods=200).std().to_numpy()
    hour = ((ot // 3600000) % 24).astype('float64')
    F = np.column_stack([r1, r5, r15, r60, r240, r1440, volr, tbr, trr, vstd, vstd1440, hour])
    ok = np.isfinite(F).all(axis=1) & (np.arange(n) >= 300)
    idx = np.arange(300, n, STEP)
    idx = idx[ok[idx]]
    if len(idx) == 0:
        continue
    feat_frames.append(pd.DataFrame({
        'sym': sym, 'i': idx, 't_ms': ot[idx],
        'r1': r1[idx], 'r5': r5[idx], 'r15': r15[idx], 'r60': r60[idx],
        'r240': r240[idx], 'r1440': r1440[idx],
        'volr': volr[idx], 'tbr': tbr[idx], 'trr': trr[idx],
        'vstd': vstd[idx], 'vstd1440': vstd1440[idx], 'hour': hour[idx]}))

X = pd.concat(feat_frames, ignore_index=True)
del feat_frames, df
import gc; gc.collect()
print('feature rows %d  syms %d  %.1fs' % (len(X), X['sym'].nunique(), time.time() - t0))

# 分位数用全样本（诚实边界里注明）
def qz(col, p):
    return float(np.nanpercentile(X[col].to_numpy(), p))

SIG = [
    ('r1<-1%', X.r1 < -0.01), ('r1>+1%', X.r1 > 0.01),
    ('r5<-2%', X.r5 < -0.02), ('r5>+3%', X.r5 > 0.03),
    ('r15<-2%', X.r15 < -0.02), ('r15<-5%', X.r15 < -0.05), ('r15>+2%', X.r15 > 0.02),
    ('r60<-5%', X.r60 < -0.05), ('r60>+5%', X.r60 > 0.05), ('r60>+10%', X.r60 > 0.10),
    ('r240<-10%', X.r240 < -0.10), ('r240>+5%', X.r240 > 0.05), ('r240>+10%', X.r240 > 0.10),
    ('r1440<-20%', X.r1440 < -0.20), ('r1440>+20%', X.r1440 > 0.20),
    ('volr>1.5', X.volr > 1.5), ('volr>3', X.volr > 3), ('volr>6', X.volr > 6),
    ('volr<0.5', X.volr < 0.5), ('volr<0.3', X.volr < 0.3),
    ('vstd>p50', X.vstd > qz('vstd', 50)), ('vstd>p70', X.vstd > qz('vstd', 70)),
    ('vstd>p90', X.vstd > qz('vstd', 90)), ('vstd>p97', X.vstd > qz('vstd', 97)),
    ('vstd<p50', X.vstd < qz('vstd', 50)),
    ('vstd1440>p90', X.vstd1440 > qz('vstd1440', 90)),
    ('tbr>0.55', X.tbr > 0.55), ('tbr>0.6', X.tbr > 0.6), ('tbr>0.7', X.tbr > 0.7),
    ('tbr<0.4', X.tbr < 0.4), ('tbr<0.45', X.tbr < 0.45),
    ('trr>1', X.trr > 1), ('trr>3', X.trr > 3), ('trr<0.5', X.trr < 0.5), ('trr<0.3', X.trr < 0.3),
    ('hour 0-6', (X.hour >= 0) & (X.hour < 6)), ('hour 6-12', (X.hour >= 6) & (X.hour < 12)),
    ('hour 12-18', (X.hour >= 12) & (X.hour < 18)), ('hour 18-24', (X.hour >= 18) & (X.hour < 24)),
    ('r1>0', X.r1 > 0), ('r1<0', X.r1 < 0), ('r5>0', X.r5 > 0), ('r5<0', X.r5 < 0),
]
print('signals %d' % len(SIG))

TPs = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.010', '0.020', '0.030'])]
SLs = [float(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else ['0.010', '0.020', '0.030'])]

def simulate(mask):
    """给定信号 mask，对每个 TP/SL 跑一遍；返回每组统计。"""
    sub = X[mask]
    if len(sub) < 200:
        return []
    out = []
    for TP in TPs:
        for SL in SLs:
            rs = []
            ts = []
            for sym, g in sub.groupby('sym', sort=False):
                c = paths[sym]
                for i, tms in zip(g['i'].to_numpy(), g['t_ms'].to_numpy()):
                    ep = c[i]
                    up = ep * (1 + TP)
                    dn = ep * (1 - SL)
                    end = min(i + MAXH, len(c) - 1)
                    seg = c[i + 1:end + 1]
                    if len(seg) == 0:
                        continue
                    hu = np.where(seg >= up)[0]
                    hd = np.where(seg <= dn)[0]
                    ju = hu[0] if len(hu) else 10**9
                    jd = hd[0] if len(hd) else 10**9
                    j = min(ju, jd)
                    if j == 10**9:
                        px = seg[-1]
                    elif ju <= jd:
                        px = up
                    else:
                        px = dn
                    rs.append(px / ep - 1.0 - COST)
                    ts.append(int(tms) + (j + 1) * 60000)
            if len(rs) < 100:
                continue
            o = np.argsort(ts)
            r = np.array(rs)[o]
            t = np.array(ts)[o]
            # 日目标停手：按 UTC 日切，日内累计到 DAY_TARGET 就停
            days = (t // 86400000).astype('int64')
            eq_day = {}
            taken = 0
            day_ret = {}
            for rr, dd in zip(r, days):
                cur = eq_day.get(dd, 1.0)
                if cur >= DAY_TARGET:
                    continue
                cur *= (1.0 + rr)
                eq_day[dd] = cur
                taken += 1
            for dd, v in eq_day.items():
                day_ret[dd] = v - 1.0
            dr = np.array(list(day_ret.values()))
            # 无停手对照
            eq_all = float(np.prod(1.0 + r))
            out.append(dict(TP=TP, SL=SL, n=len(r), taken=taken,
                            days=len(day_ret), win=(r > 0).mean(), mean=r.mean(),
                            daily_mean=dr.mean(), hit=(dr >= DAY_TARGET - 1).mean(),
                            eq_all=eq_all, eq_stop=float(np.prod(1.0 + dr))))
    return out

L = []
L.append('43 信号 × 出场网格 × 日目标停手  file=%s' % FN)
L.append('=' * 118)
L.append('出场：收盘价触线才成交（无前视）｜成本 %.2f%% 往返｜最长持 24h｜日目标 +2.01%% 停手' % (COST * 100))
L.append('样本 %d 行 / %d 币 / %d 天' % (
    len(X), X['sym'].nunique(), (X['t_ms'].max() - X['t_ms'].min()) / 86400000.0))
L.append('')
hdr = '%-16s %-6s %-6s %7s %7s %7s %8s %10s %9s %9s %9s' % (
    '信号', 'TP', 'SL', '命中', '成交', '天', '胜率', '单笔均值', '日达标', '日终值', '无停手')
L.append(hdr)
L.append('-' * 118)

rows_all = []
for name, mask in SIG:
    res = simulate(mask)
    for x in res:
        rows_all.append((name,) + tuple(x.values()))
    # 每个信号只打最好的一组
    if res:
        b = max(res, key=lambda z: z['eq_stop'])
        L.append('%-16s %-6.1f%% %-6.1f%% %7d %7d %7d %7.1f%% %9.4f%% %8.1f%% %9.3f %9.3e' % (
            name, b['TP'] * 100, b['SL'] * 100, int(mask.sum()), b['taken'], b['days'],
            b['win'] * 100, b['mean'] * 100, b['hit'] * 100, b['eq_stop'], b['eq_all']))
    else:
        L.append('%-16s  (样本不足)' % name)

L.append('')
L.append('--- 判读 ---')
L.append('  · 「日达标」= 该信号下有多少比例的交易日能到 +2.01%')
L.append('  · 「日终值」= 每天到 +2.01% 停手、否则跑完的复利终值')
L.append('  · 「无停手」= 不设日目标的复利终值（对照）')
L.append('')
L.append('--- 诚实边界 ---')
L.append('1. 收盘确认，不用 high/low 触价；实盘还需排队与滑点，这里只扣 0.15% 固定成本。')
L.append('2. 信号命中点未去重（相邻分钟可能重复计同一波），样本重叠。')
L.append('3. 分位阈值用全样本算（含未来），对分位数类信号有轻微前视。')
L.append('4. 单一数据集，含样本内；未做时间切分外推。')

out = os.path.join(EVI, 'signal_x_exit.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwritten ' + out)
