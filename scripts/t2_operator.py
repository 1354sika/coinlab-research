# -*- coding: utf-8 -*-
"""把 T2 做成正式算子：市场门控（市场 15 分钟跌幅 > 2%）—— 从"事后解释"升级为"前置开关"

算子 OP_LIQUIDATION_SNAP（强平踩踏回补）：
  入场（同时满足）：
    A 段内跌幅 < -DEP        （急跌）
    B 量比 > VOLR            （巨量）
    C 横截面分化 > P90       （disp，之前已确认）
    D 【T2】市场 15 分钟跌幅 < -2%   ← 本脚本的主角
    E 【T1】同分钟触发数 >= 4        （系统性去杠杆）
  方向：做多    持有：60 分钟    出场：时间到期

实验：
  ① 消融：逐个关掉 A/B/C/D/E，看哪个门控是真贡献
  ② 完整回测：事件去重 + 无前视 + 扣费（maker/taker）
  ③ 分月稳定性
  ④ 出图
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

DEP, VOLR, DQ, MKT_GATE, CONCUR, HOLD = 0.05, 10.0, 0.90, -0.02, 4, 60
FEE_M, FEE_T = 0.0002, 0.0010

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64',
                        'close': 'float32', 'quote_volume': 'float32'})
px = df.pivot(index='open_time', columns='symbol', values='close')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume')
del df
P = px.values.astype(np.float64); QV = qv.values.astype(np.float64)
T, N = P.shape; tms = px.index.values
syms = list(px.columns)
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret1 = np.full((T, N), np.nan)
ret1[1:] = P[1:] / P[:-1] - 1.0
qv60 = pd.DataFrame(QV).rolling(60).mean().values
volr = QV / qv60
disp_v = np.nanstd(ret1, axis=1)
mkt_r = np.nanmedian(ret1, axis=1)
mkt15 = pd.Series(mkt_r).rolling(15).sum().values

s = np.sign(ret1); s[~np.isfinite(s)] = 0
is_start = np.vstack([np.ones((1, N), bool), (s[1:] != s[:-1])])
rows = np.arange(T)[:, None] * np.ones((1, N), np.int64)
start = np.maximum.accumulate(np.where(is_start, rows, 0), axis=0)
seglen = (rows - start + 1).astype(np.float64)
streak_ret = P / np.take_along_axis(P, start, axis=0) - 1.0
speed = streak_ret / seglen

FWD = np.full((T, N), np.nan)
FWD[:-HOLD] = P[HOLD:] / P[:-HOLD] - 1.0

d90 = float(np.nanpercentile(disp_v[60:], DQ * 100))
ones = np.ones((1, N), bool)
C_dep = streak_ret < -DEP
C_vol = volr > VOLR
C_disp = (disp_v > d90)[:, None] & ones
C_mkt = (mkt15 < MKT_GATE)[:, None] & ones
BASE = C_dep & C_vol & C_disp
concur = BASE.sum(axis=1)[:, None] * np.ones((1, N))
C_concur = concur >= CONCUR
say('disp P90=%.4f%%  mkt15<-2%% 的分钟占比=%.1f%%' % (
    100 * d90, 100 * np.nanmean(mkt15 < MKT_GATE) if np.isfinite(mkt15).any() else 0))

FULL = BASE & C_mkt & C_concur


def dedup(SIG, hold=HOLD):
    """同一币在持仓期（hold 分钟）内不重复开仓"""
    out = np.zeros_like(SIG)
    last = np.full(N, -10 ** 9, np.int64)
    idx = np.where(SIG.any(axis=1))[0]
    for i in idx:
        row = SIG[i]
        sel = row & ((i - last) >= hold)
        if sel.any():
            out[i] = sel
            last[sel] = i
    return out


def stats(SIG, hold=HOLD):
    S = dedup(SIG, hold)
    y = FWD[S]; y = y[np.isfinite(y)]
    if len(y) < 20:
        return dict(n=len(y))
    g = y.mean()
    return dict(n=len(y), ev=int(S.sum()), gross=100 * g, median=100 * np.median(y),
                win=100 * (y > 0).mean(), netm=100 * (g - 2 * FEE_M),
                nett=100 * (g - 2 * FEE_T),
                sd=100 * y.std(), S=S, y=y)


say('\n' + '=' * 82)
say('【一】消融：逐条门控是不是真贡献（全部门控 vs 逐个关掉）')
say('  %-34s %6s %7s %9s %9s %9s %7s' % ('组合', '事件', '开仓', '毛均值%', '净maker%', '净taker%', '胜率%'))
combos = [
    ('全部（A急跌+B量比+C分化+D市场+E多币）', FULL),
    ('关 D（市场门控）', BASE & C_concur),
    ('关 E（多币同时）', BASE & C_mkt),
    ('关 D 和 E', BASE),
    ('关 C（分化）', C_dep & C_vol & C_mkt & C_concur),
    ('关 B（量比）', C_dep & C_disp & C_mkt & C_concur),
    ('关 A（跌幅，改 >-2%）', (streak_ret < -0.02) & C_vol & C_disp & C_mkt & C_concur),
]
res = {}
for nm, SG in combos:
    d = stats(SG)
    if d.get('ev', 0) >= 20:
        res[nm] = d
        say('  %-34s %6d %7d %8.3f%% %8.3f%% %8.3f%% %6.1f%%' % (
            nm, d['ev'], d['n'], d['gross'], d['netm'], d['nett'], d['win']))
    else:
        say('  %-34s 样本不足' % nm)

say('\n【二】完整回测：全部算子（持 60 分钟，事件去重，无前视）')
d = res.get(combos[0][0])
if d:
    say('  开仓 %d 次  毛均值=%+.3f%%  中位=%+.3f%%  胜率=%.1f%%  单笔波动=%.2f%%' % (
        d['n'], d['gross'], d['median'], d['win'], d['sd']))
    say('  taker 净=%+.3f%%/笔    maker 净=%+.3f%%/笔' % (d['nett'], d['netm']))
    y = d['y']
    # 逐月
    mo = pd.to_datetime(tms, unit='ms', utc=True).strftime('%Y-%m')
    # 找每个事件的时间：重算
    S = dedup(FULL)
    ii, jj = np.where(S)
    yv = FWD[ii, jj]; ok = np.isfinite(yv)
    ii, jj, yv = ii[ok], jj[ok], yv[ok]
    mos = pd.to_datetime(tms[ii], unit='ms', utc=True).strftime('%Y-%m')
    say('\n【三】逐月稳定性（maker 口径）')
    say('  %-9s %5s %10s %10s %8s' % ('月份', 'n', '毛均值%', '净maker%', '胜率%'))
    rows_m = []
    for m in sorted(set(mos)):
        msk = mos == m
        if msk.sum() < 3:
            continue
        yy = yv[msk]
        net = 100 * (yy.mean() - 2 * FEE_M)
        rows_m.append((m, msk.sum(), 100 * yy.mean(), net, 100 * (yy > 0).mean()))
        say('  %-9s %5d %9.3f%% %9.3f%% %7.1f%%' % rows_m[-1])
    pos = sum(1 for r in rows_m if r[3] > 0)
    say('  → 正收益月 %d/%d' % (pos, len(rows_m)))

    # 累计净值（按事件时间序列等权，每笔风险 1 单位）
    order = np.argsort(ii)
    eq = np.cumprod(1 + 0.01 * (yv[order] - 2 * FEE_M))  # 每笔只投 1% 资金（风险预算口径）
    say('\n【四】净值（每笔投 1% 资金、按事件顺序，maker）')
    say('  累计净值= %.3fx   最终 %+.1f%%   最大回撤=%.1f%%' % (
        eq[-1], 100 * (eq[-1] - 1), 100 * (eq / np.maximum.accumulate(eq) - 1).min()))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax = ax.ravel()
    names = [c[0] for c in combos if c[0] in res]
    xs = [res[n]['netm'] for n in names]
    ax[0].barh(range(len(names)), xs)
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels([n[:20] for n in names], fontsize=7)
    ax[0].axvline(0, color='k', lw=.8); ax[0].set_title('ablation: net(maker) % per trade')
    ax[1].plot(np.arange(len(eq)), eq); ax[1].set_yscale('log')
    ax[1].set_title('equity (per-event, maker)'); ax[1].set_xlabel('event #')
    ax[2].hist(100 * yv, bins=60); ax[2].axvline(0, color='k', lw=.8)
    ax[2].set_title('per-event return %% distribution')
    mm = [r[0] for r in rows_m]; vv = [r[3] for r in rows_m]
    ax[3].bar(range(len(mm)), vv); ax[3].set_xticks(range(len(mm)))
    ax[3].set_xticklabels(mm, rotation=70, fontsize=7)
    ax[3].axhline(0, color='k', lw=.8); ax[3].set_title('monthly net(maker) %')
    plt.tight_layout()
    p = os.path.join(EV, 't2_operator.png'); plt.savefig(p, dpi=125)
    say('\nsaved: %s' % p)

with open(os.path.join(EV, 't2_operator.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
