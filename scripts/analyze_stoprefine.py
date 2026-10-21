# -*- coding: utf-8 -*-
"""止损规律细化重验 + 爆发点信号的止损优化
背景：此前 R3 结论「中间价止损触发率反而更高」是【全体样本平均】得出。
本脚本按 (a) 分化时刻 vs 非分化时刻  (b) 是否处于爆发点事件 分层重算触发率。
并给爆发点信号（分化+急跌5%+放量2倍）找最优止损距离与持有期组合。
"""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_502.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

FEE_T, FEE_M = 0.0020, 0.0004

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64', 'close': 'float32',
                        'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
hi = df.pivot(index='open_time', columns='symbol', values='high')
lo = df.pivot(index='open_time', columns='symbol', values='low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
say('matrix %d x %d  %.0fs' % (px.shape[0], px.shape[1], time.time() - t0))

r = px.pct_change()
disp = r.std(axis=1); disp_v = disp.values
R15 = (px / px.shift(15) - 1.0).values
VR = (qv / qv.rolling(60).mean()).values
C = px.values; L = lo.values
dq = np.nanquantile(disp_v[np.isfinite(disp_v)], [0.50, 0.85, 0.90, 0.95])

# ---------------- A) 爆发点信号的止损优化 ----------------
cond = (R15 < -0.05) & (VR > 2.0)
sel = disp_v > dq[2]
ev = []
for j in range(cond.shape[1]):
    ii = np.flatnonzero(cond[:, j] & sel)
    if len(ii) == 0:
        continue
    starts = np.r_[0, np.flatnonzero(np.diff(ii) > 5) + 1]
    for s in starts:
        k = ii[s]
        if k + 61 < C.shape[0]:
            ev.append((k, j))
say('\n事件数（分化P90 + 急跌5%% + 放量2倍，事件级）= %d' % len(ev))

SLS = [0.003, 0.005, 0.010, 0.020, 0.030, 0.050]
HOLD = [5, 15, 30, 60]

say('\n=== A) 爆发点信号：止损 x 持有期（事件级；扣 maker 0.04% / taker 0.20%）===')
say('  %-8s %6s %9s %10s %10s %9s' % ('止损', '持有', '毛均值%', '净taker%', '净maker%', '触发率%'))
for sl in [None] + SLS:
    for h in HOLD:
        rs = []; hits = 0; tot = 0
        for k, j in ev:
            e = C[k, j]
            if not np.isfinite(e) or e <= 0:
                continue
            path_c = C[k + 1:k + 1 + h, j]; path_l = L[k + 1:k + 1 + h, j]
            ret = path_c[-1] / e - 1.0
            if sl is not None:
                tot += 1
                if np.any(path_l <= e * (1 - sl)):
                    ret = -sl; hits += 1
            rs.append(ret)
        rs = np.array(rs); rs = rs[np.isfinite(rs)]
        if len(rs) < 50:
            continue
        say('  %-8s %6s %8.3f %10.3f %10.3f %9s' % (
            '无' if sl is None else '%.1f%%' % (sl * 100), '%dm' % h,
            100 * rs.mean(), 100 * (rs.mean() - FEE_T), 100 * (rs.mean() - FEE_M),
            '-' if sl is None else '%.1f' % (100 * hits / max(tot, 1))))

# ---------------- B) 止损口径：收盘 vs 中间价，分层 ----------------
say('\n=== B) 止损口径：收盘价 vs 中间价 (high+low)/2 的触发率 ===')
say('  （此前 R3 的结论是全体平均；这里按分化/非分化分层）')
HL = (hi + lo) / 2.0
for h in (15, 60):
    fwd_lo = lo.rolling(h).min().shift(-h)
    fwd_mid = HL.rolling(h).min().shift(-h)
    valid = fwd_lo.notna() & fwd_mid.notna()
    say('  持有 %d 分钟:' % h)
    for nm, layer in [('全体', None), ('分化时刻 disp>P90', disp > dq[2]), ('非分化时刻', disp <= dq[2])]:
        for sl in (0.005, 0.010):
            c_hit = (fwd_lo <= px * (1 - sl)) & valid
            m_hit = (fwd_mid <= px * (1 - sl)) & valid
            v = valid
            if layer is not None:
                c_hit = c_hit.loc[layer]; m_hit = m_hit.loc[layer]; v = v.loc[layer]
            tot = int(v.sum().sum())
            if tot == 0:
                continue
            rc = int(c_hit.sum().sum()) / tot
            rm = int(m_hit.sum().sum()) / tot
            say('    %-22s 止损%.1f%%  收盘=%.4f  中间价=%.4f  相对差=%+.1f%%' % (
                nm, sl * 100, rc, rm, 100 * (rm - rc) / max(rc, 1e-9)))

with open(os.path.join(EV, 'stoprefine.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
