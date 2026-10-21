# -*- coding: utf-8 -*-
"""gambler_rule.py — q5：把「位置 + 市场齐涨 + 量能」合成可执行规则，测期望净与出手频率

上游证据：
    q1 job-24708-14/17/19：位置(over_now / dist_hi288)是唯一有信息的变量；但概率高≠期望高
    q2 job-24708-20/21：散户侧（局外，未启动）无脑入场 -0.075%/笔；
       5 条规则组合在「币留出 + 置换检验」后仍为正（C2 +0.91%、C7 +0.61%、C3/C5 各约 +0.4%）

【本脚本】把这些组合固化成 7 条可执行规则，给出一张能直接照做的规则卡：
    每条规则报：总笔数、期望净、中位、胜率、精确率、
                **时间留出 / 币留出 / 双重留出** 的期望净、
                **平均每天出手几次**（按抽样比例还原到全市场）、
                逐年稳定性（2023~2026）
【口径】与 retail30 完全一致，可比：
    局外点（相对 288 根低点涨幅 <15%）→ 未来 48h 触及 +30% 为止盈，-15% 止损，48h 超时
    扣费 0.09%；同窗口内高低都触 → 保守按止损
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288; W = 576; TARGET = 0.30; SL = 0.15; FEE = 0.0009
KEEP = 4000

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 120)
log('gambler_rule · 可执行规则卡（位置 × 市场齐涨 × 量能）')
log('=' * 120)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32', 'quote_volume': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
syms = list(C.columns); ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
R12 = C.pct_change(12).values
up_cnt = (np.nan_to_num(R12, nan=0.0) >= 0.05).sum(axis=1)
mkt_up = pd.Series(up_cnt).rolling(24, min_periods=1).max().values
del R12
log(f'币 {N}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

FN = ['r12', 'r288', 'over_now', 'pos288', 'dist_hi288', 'volr6', 'volr24', 'volr96', 'mkt_up']
XS, YS, WS, TID, CO, SCALE = [], [], [], [], [], 0.0

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4 + W:
        continue
    lo288 = pd.Series(l).rolling(WDAY, min_periods=72).min().values
    hi288 = pd.Series(h).rolling(WDAY, min_periods=72).max().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
        dhi = c / np.maximum(hi288, 1e-12) - 1.0
        pos288 = (c - lo288) / np.maximum(hi288 - lo288, 1e-12)
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / np.maximum(c[:-12], 1e-12) - 1.0
    r288 = np.full(n, np.nan); r288[288:] = c[288:] / np.maximum(c[:-288], 1e-12) - 1.0
    base_q = pd.Series(q).rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    volr96 = pd.Series(volr).rolling(96, min_periods=24).mean().values

    m = (over < 0.15) & np.isfinite(volr24) & (np.arange(n) < n - W - 1)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        continue
    SCALE += len(idx) / max(min(len(idx), KEEP), 1)     # 抽样还原倍数
    if len(idx) > KEEP:
        idx = idx[::max(1, len(idx) // KEEP)]
    wh = np.lib.stride_tricks.sliding_window_view(h, W)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, W)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, W)[idx]
    c0 = c[idx]
    hit_tp = wh >= (c0 * 1.30)[:, None]; hit_sl = wl <= (c0 * 0.85)[:, None]
    it = np.argmax(hit_tp, 1); il = np.argmax(hit_sl, 1)
    any_tp = hit_tp.any(1); any_sl = hit_sl.any(1)
    w_ = np.where(any_sl & (~any_tp | (il <= it)), -SL,
                  np.where(any_tp, TARGET, wc[:, -1] / c0 - 1.0)) - FEE
    y_ = ((wh.max(1) / c0 - 1.0) >= TARGET).astype('int8')
    posv = np.searchsorted(ts, ot)[idx]
    cols = [r12[idx], r288[idx], over[idx], pos288[idx], dhi[idx],
            volr6[idx], volr24[idx], volr96[idx], mkt_up[posv]]
    XS.append(np.column_stack([np.nan_to_num(v, nan=0.0, posinf=0.0) for v in cols]).astype('float32'))
    YS.append(y_); WS.append(w_.astype('float32'))
    TID.append(posv.astype('int64')); CO.append(np.full(len(idx), sym))
del RAW, C; gc.collect()

X = np.vstack(XS); Y = np.concatenate(YS); Wv = np.concatenate(WS)
TID = np.concatenate(TID); CO = np.concatenate(CO)
del XS, YS, WS; gc.collect()
F = {nm: X[:, j] for j, nm in enumerate(FN)}
te = TID >= t_cut
days_all = len(np.unique(ts[TID] // 86_400_000))
log(f'样本 {len(Y):,}（全量折算 ×{SCALE/len(np.unique(CO)):.1f}）   测试段 {int(te.sum()):,}   '
    f'币 {len(np.unique(CO))}   独立天 {days_all}   耗时 {time.time()-t0:.0f}s')

coins = np.array(sorted(np.unique(CO)))
rng0 = np.random.default_rng(11)
half = set(rng0.choice(coins, size=len(coins)//2, replace=False).tolist())
coin_in = np.isin(CO, list(half))

RULES = {
    'R1 mkt_up>=2 & r288>=5%':          (F['mkt_up'] >= 2) & (F['r288'] >= 0.05),
    'R2 r288>=5% & mkt_up>=1':          (F['r288'] >= 0.05) & (F['mkt_up'] >= 1),
    'R3 mkt_up>=2 & volr6>=1.8':        (F['mkt_up'] >= 2) & (F['volr6'] >= 1.8),
    'R4 mkt_up>=2 & volr24>=1.8':       (F['mkt_up'] >= 2) & (F['volr24'] >= 1.8),
    'R5 over>=8% & r288<=-6.3%':        (F['over_now'] >= 0.08) & (F['r288'] <= -0.063),
    'R6 mkt_up>=2 & r288>=5% & volr6>=1.8':
        (F['mkt_up'] >= 2) & (F['r288'] >= 0.05) & (F['volr6'] >= 1.8),
    'R7 mkt_up>=2 & r288>=3% & dist_hi>=-8%':
        (F['mkt_up'] >= 2) & (F['r288'] >= 0.03) & (F['dist_hi288'] >= -0.08),
    'Z  全部局外点（对照）': np.ones(len(Y), bool),
}

def cell(mask):
    if mask.sum() == 0:
        return None
    v = Wv[mask]
    d = len(np.unique(ts[TID[mask]] // 86_400_000))
    return dict(n=int(mask.sum()), exp=float(np.nanmean(v)), med=float(np.nanmedian(v)),
                wr=float(np.nanmean(v > 0)), rate=float(Y[mask].mean()),
                coins=int(len(np.unique(CO[mask]))), days=int(d),
                per_day=float(mask.sum() * (SCALE / len(np.unique(CO))) / max(d, 1)))

log('\n' + '=' * 120)
log('【规则卡】每格：笔数 / 期望净% / 中位% / 胜率% / 命中率%(48h内触及+30%) / 币数 / 独立天 / 每天出手次数')
log('=' * 120)
hdr = f'  {"规则":<40}{"全样本":>34}{"时间留出":>34}{"币留出":>34}'
log(hdr)
for name, m in RULES.items():
    parts = []
    for mask in (m, m & te, m & te & ~coin_in):
        s = cell(mask)
        parts.append('—' if s is None else
                     f'{s["n"]:,}/{100*s["exp"]:+.2f}/{100*s["med"]:+.1f}/'
                     f'{100*s["wr"]:.0f}/{100*s["rate"]:.1f}/{s["coins"]}/{s["days"]}/{s["per_day"]:.1f}')
    log(f'  {name:<40}{parts[0]:>34}{parts[1]:>34}{parts[2]:>34}')

log('\n' + '=' * 120)
log('【逐年稳定性】时间留出后的逐年期望净（%，括号内为笔数）')
log('=' * 120)
yr = pd.to_datetime(ts[TID], unit='ms').year
log(f'  {"规则":<40}' + ''.join(f'{y:>16}' for y in sorted(np.unique(yr))))
for name, m in RULES.items():
    cells = []
    for y in sorted(np.unique(yr)):
        mm = m & te & (yr == y)
        cells.append('—' if mm.sum() < 20 else f'{100*np.nanmean(Wv[mm]):+.2f}({int(mm.sum())})')
    log(f'  {name:<40}' + ''.join(f'{c:>16}' for c in cells))

log('\n' + '=' * 120)
log('【组合最优】三重留出（时间 + 币）下，期望净 > 0 的规则')
log('=' * 120)
best = []
for name, m in RULES.items():
    s = cell(m & te & ~coin_in)
    if s and s['n'] >= 100:
        best.append((name, s))
best.sort(key=lambda x: -x[1]['exp'])
for name, s in best:
    mark = ' ★' if s['exp'] > 0 else ''
    log(f'  {name:<40} 笔{s["n"]:>7,}  期望净 {100*s["exp"]:>+7.2f}%  中位 {100*s["med"]:>+7.2f}%  '
        f'胜率 {100*s["wr"]:>5.1f}%  命中 {100*s["rate"]:>4.1f}%  {s["coins"]}币/{s["days"]}天  '
        f'每天 {s["per_day"]:.1f} 次{mark}')

if best and best[0][1]['exp'] > 0:
    n, s = best[0]
    log('\n' + '=' * 120)
    log(f'【规则卡最终推荐】{n}')
    log('=' * 120)
    log(f'  触发条件见上方公式；入场用市价（滑点按 0.05% 单边已计入总费 0.09%）')
    log(f'  退出：+30% 止盈 / -15% 止损 / 48h 到期平仓')
    log(f'  实测（时间+币双重留出）：笔数 {s["n"]:,}，期望净 {100*s["exp"]:+.2f}%/笔，'
        f'中位 {100*s["med"]:+.2f}%，胜率 {100*s["wr"]:.1f}%，命中率 {100*s["rate"]:.1f}%')
    log(f'  覆盖 {s["coins"]} 个币 / {s["days"]} 个独立日，平均每天出手 {s["per_day"]:.1f} 次')
    log(f'  含义：每投入 1 元、每笔平均赚 {s["exp"]:.4f} 元；要拿到 +30% 必须命中那 '
        f'{100*s["rate"]:.1f}% 的尾部')

out = os.path.join(OUT, 'rule_card.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
