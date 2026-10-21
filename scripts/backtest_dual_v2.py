# -*- coding: utf-8 -*-
"""D) 用 C 找到的严格门槛重跑组合回测（回答：门槛收紧后能不能转正）
门槛对比（做多侧；做空侧镜像）：
  V1 volr>2                      —— B 的旧版（已知 -98%）
  V2 volr>5                      —— C 的关键发现
  V3 volr>5 & 段长>=3
  V4 volr>5 & speed>=2%/min
  V5 volr>5 & 段长>=3 & speed>=2%/min
每个版本：多空组合、无前视、扣 taker(0.10%/边) 与 maker(0.02%/边)。
另测 V2 在「同时持仓上限」= 不限 / 20 / 5 下的差异（检验崩盘满仓假说）。
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

HOLD, THR, DQ = 60, 0.05, 0.90
FEE_T, FEE_M = 0.0010, 0.0002

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64)
R = px.pct_change().values.astype(np.float64)
disp_v = px.pct_change().std(axis=1).values
qv60 = qv.rolling(60).mean().values
QV = qv.values.astype(np.float64)
T, N = P.shape
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

s = np.sign(R); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
P0 = np.take_along_axis(P, start, axis=0)
streak_ret = P / P0 - 1.0
speed = streak_ret / seglen
volr = QV / qv60
d90 = float(np.nanpercentile(disp_v, DQ * 100))
sel = (disp_v > d90)[:, None] & np.ones((1, N), bool)


def build_pos(sig, sign, maxpos=10 ** 9):
    pos = np.zeros((T, N), np.float32)
    opens = 0
    for j in range(N):
        ii = np.flatnonzero(sig[:, j])
        if len(ii) == 0:
            continue
        last = -1
        for k in ii:
            if k <= last:
                continue
            if (np.abs(pos[k]) > 0).sum() >= maxpos:   # 同时持仓上限（近似）
                continue
            pos[k:k + HOLD, j] = sign
            opens += 1
            last = k + HOLD - 1
    return pos, opens


def run(SIG_L, SIG_S, tag, maxpos=10 ** 9):
    posL, nL = build_pos(SIG_L, +1, maxpos)
    posS, nS = build_pos(SIG_S, -1, maxpos)
    pos = posL + posS
    Rn = np.vstack([R[1:], np.full((1, N), np.nan)])
    pnl = np.nansum(pos[:-1] * Rn[:-1], axis=1)
    act = (np.abs(pos[:-1]) > 0).sum(axis=1).astype(np.float64)
    pnl_eq = np.where(act > 0, pnl / np.maximum(act, 1), 0.0)
    out = {'tag': tag, 'nL': nL, 'nS': nS, 'act_mean': act.mean(), 'act_max': act.max()}
    for nm, fee in (('taker', FEE_T), ('maker', FEE_M)):
        fs = np.zeros(T)
        # 用实际开仓位置扣费：近似用 pos 变化
        d = np.abs(np.diff(pos, axis=0, prepend=0))
        chg = (d > 1e-6).sum(axis=1)
        fs = chg * fee
        fse = np.where(act > 0, fs[:-1] / np.maximum(act, 1), 0.0)
        net = pnl_eq - fse
        nav = np.cumprod(1 + net)
        sd = net.std()
        out[nm + '_tot'] = nav[-1] - 1
        out[nm + '_sharpe'] = net.mean() / sd * np.sqrt(525600) if sd > 0 else np.nan
        pk = np.maximum.accumulate(nav)
        out[nm + '_mdd'] = ((nav - pk) / pk).min()
    return out


vers = [
    ('V1 volr>2 (旧,B)', lambda: ((streak_ret < -THR) & (volr > 2) & sel,
                                  (streak_ret > +THR) & (volr > 2) & sel)),
    ('V2 volr>5', lambda: ((streak_ret < -THR) & (volr > 5) & sel,
                           (streak_ret > +THR) & (volr > 5) & sel)),
    ('V3 volr>5 & 段长>=3', lambda: ((streak_ret < -THR) & (volr > 5) & (seglen >= 3) & sel,
                                     (streak_ret > +THR) & (volr > 5) & (seglen >= 3) & sel)),
    ('V4 volr>5 & speed>=2%/min', lambda: ((streak_ret < -THR) & (volr > 5) & (np.abs(speed) >= 0.02) & sel,
                                           (streak_ret > +THR) & (volr > 5) & (np.abs(speed) >= 0.02) & sel)),
    ('V5 volr>5 & 段>=3 & speed>=2%', lambda: ((streak_ret < -THR) & (volr > 5) & (seglen >= 3) & (np.abs(speed) >= 0.02) & sel,
                                               (streak_ret > +THR) & (volr > 5) & (seglen >= 3) & (np.abs(speed) >= 0.02) & sel)),
]

say('\n=== D) 门槛收紧后的组合回测（持 %d 分钟，无前视）===' % HOLD)
say('  %-30s %6s %6s %7s %8s %9s %7s %8s' % (
    '版本', 'n多', 'n空', '均持仓', '夏普', '净maker', '回撤', '胜率'))
for tag, fn in vers:
    SL, SS = fn()
    r = run(SL, SS, tag)
    say('  %-30s %6d %6d %7.2f %8.2f %9.1f%% %7.1f%%' % (
        tag, r['nL'], r['nS'], r['act_mean'],
        r['maker_sharpe'], 100 * r['maker_tot'], 100 * r['maker_mdd']))
    say('       taker: 总净=%+.1f%%  夏普=%.2f  回撤=%.1f%%' % (
        100 * r['taker_tot'], r['taker_sharpe'], 100 * r['taker_mdd']))

say('\n=== V2 在不同「同时持仓上限」下（检验崩盘满仓假说）===')
SL2, SS2 = vers[1][1]()
for mp in (10 ** 9, 20, 5):
    r = run(SL2, SS2, 'V2 maxpos=%s' % ('无限' if mp > 1e8 else mp), maxpos=mp)
    say('  maxpos=%-6s 开仓 多%d/空%d  平均持仓=%.2f  最大=%d  maker总净=%+.1f%%  夏普=%.2f  回撤=%.1f%%' % (
        '无限' if mp > 1e8 else mp, r['nL'], r['nS'], r['act_mean'], r['act_max'],
        100 * r['maker_tot'], r['maker_sharpe'], 100 * r['maker_mdd']))

with open(os.path.join(EV, 'backtest_dual_v2.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
