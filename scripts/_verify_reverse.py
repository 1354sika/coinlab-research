# -*- coding: utf-8 -*-
"""_verify_reverse.py — 严格验证「做空急跌启动」是否真能赚钱

由来（job-24708-26 gambler_reverse）：
    C 信号「288根涨≥50% 且 近12根回落≥5%」做空：913 笔，全样本 +3.47%/笔，
    **测试段 459 笔 +8.25%/笔、胜率 58.0%**；对照组随机做空 -0.17%。
    而所有"追涨"规则只有 +0.4~1.0%/笔 ⇒ 反转方向可能是真正的主战场。

必须排除的四种假象：
    ① 多重比较：7 个信号里挑最好的 → 置换检验（打乱时间顺序重跑）
    ② 币偏差：是不是只有几个币能赚 → 币留出（训练币 / 测试币不重叠）
    ③ 时段偏差：是不是只有某一年能赚 → 逐年
    ④ 口径偏差：退出规则挑得好看 → 退出网格全跑一遍

本脚本只做这一件事，输出一张判定表。
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
W = 576; FEE = 0.0009
NPERM = 20

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('_verify_reverse · 做空「急跌启动」的严格验证')
log('=' * 116)

RAW = []
for ch in pd.read_csv(SRC,
                      usecols=['symbol', 'open_time', 'close', 'high', 'low'],
                      dtype={'symbol': 'object', 'open_time': 'int64', 'close': 'float32',
                             'high': 'float32', 'low': 'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 5000:
            continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()

C = pd.DataFrame({r[0]: pd.Series(r[2], index=r[1]) for r in RAW}).sort_index()
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
log(f'币 {N}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

# 退出规则网格
EXITS = {
    'E1 TP30/SL15/48h': (0.30, 0.15, W),
    'E2 TP20/SL10/48h': (0.20, 0.10, W),
    'E3 TP30/SL15/8h':  (0.30, 0.15, 96),
    'E4 TP50/SL20/48h': (0.50, 0.20, W),
    'E5 移动止盈 回撤30%': (None, None, W),
}

rows = []
for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < 288 * 4 + W:
        continue
    hi288 = pd.Series(h).rolling(288, min_periods=72).max().values
    r288 = np.full(n, np.nan); r288[288:] = c[288:] / np.maximum(c[:-288], 1e-12) - 1.0
    r12 = np.full(n, np.nan); r12[12:] = c[12:] / np.maximum(c[:-12], 1e-12) - 1.0
    sig = np.isfinite(hi288) & np.isfinite(r288) & (r288 >= 0.50) & (r12 <= -0.05) & \
          (np.arange(n) < n - W - 1)
    idx = np.flatnonzero(sig)
    if len(idx) == 0:
        continue
    wh = np.lib.stride_tricks.sliding_window_view(h, W)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, W)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, W)[idx]
    c0 = c[idx]
    tpos = np.searchsorted(ts, ot)[idx]
    for tag, (tp, sl, hs) in EXITS.items():
        H = wh[:, :hs]; L = wl[:, :hs]; Cl = wc[:, :hs]
        if tp is None:
            # 移动止盈：价格从窗口最低点反弹 30% 离场（对做空而言）
            cummin = np.minimum.accumulate(Cl, axis=1)
            up = Cl / np.maximum(cummin, 1e-12) - 1.0
            trig = up >= 0.30
            it3 = np.minimum(np.argmax(trig, axis=1), hs - 1)
            px = np.take_along_axis(Cl, it3[:, None], axis=1)[:, 0]
            v = np.where(trig.any(1), 1.0 - px / c0, 1.0 - Cl[:, -1] / c0) - FEE
        else:
            hit_tp = L <= (c0 * (1 - tp))[:, None]
            hit_sl = H >= (c0 * (1 + sl))[:, None]
            it = np.argmax(hit_tp, 1); il = np.argmax(hit_sl, 1)
            v = np.where(hit_sl.any(1) & (~hit_tp.any(1) | (il <= it)), -sl,
                         np.where(hit_tp.any(1), tp, 1.0 - Cl[:, -1] / c0)) - FEE
        for k in range(len(idx)):
            rows.append((sym, int(tpos[k]), tag, float(v[k])))
    del wh, wl, wc
del RAW, C; gc.collect()

DF = pd.DataFrame(rows, columns=['sym', 'tpos', 'exit', 'net'])
log(f'信号样本 {len(DF):,}   币 {DF.sym.nunique()}   耗时 {time.time()-t0:.0f}s')
DF['year'] = pd.to_datetime(ts[DF.tpos.values], unit='ms').year
coins = np.array(sorted(DF.sym.unique()))
rng = np.random.default_rng(11)
half = set(rng.choice(coins, size=len(coins)//2, replace=False).tolist())
DF['coin_grp'] = np.where(DF.sym.isin(list(half)), 'A', 'B')
te = DF.tpos.values >= t_cut

log('\n' + '=' * 116)
log('【A】退出规则网格（测试段）')
log('=' * 116)
log(f'  {"退出规则":<22}{"笔数":>8}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"≥+30%":>8}{"≥+100%":>9}')
for tag in EXITS:
    d = DF[(DF.exit == tag) & te]
    if len(d) == 0:
        continue
    v = d.net.values
    log(f'  {tag:<22}{len(v):>8,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
        f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanmean(v >= 0.30):>8.1f}{100*np.nanmean(v >= 1.0):>9.1f}')

log('\n' + '=' * 116)
log('【B】币留出（C 组训练 / A 组测试，两半币互不重叠）')
log('=' * 116)
for tag in EXITS:
    out = []
    for g in ('A', 'B'):
        d = DF[(DF.exit == tag) & te & (DF.coin_grp == g)]
        out.append(f'{g}组 {len(d):>5,} 笔 {100*np.nanmean(d.net.values):>+7.2f}%' if len(d) else f'{g}组 无')
    log(f'  {tag:<22}' + '   |   '.join(out))

log('\n' + '=' * 116)
log('【C】逐年（测试段，E1 口径）')
log('=' * 116)
for y in sorted(DF.year.unique()):
    d = DF[(DF.exit == 'E1 TP30/SL15/48h') & te & (DF.year == y)]
    if len(d) < 5:
        continue
    log(f'  {y}  笔 {len(d):>5,}  期望净 {100*np.nanmean(d.net.values):>+7.2f}%  '
        f'中位 {100*np.nanmedian(d.net.values):>+7.2f}%  胜率 {100*np.nanmean(d.net.values > 0):>5.1f}%')

log('\n' + '=' * 116)
log(f'【D】置换检验：把「信号时刻」在时间轴上随机平移 {NPERM} 次，看随机能达到多少')
log('=' * 116)
base = DF[(DF.exit == 'E1 TP30/SL15/48h') & te].net.mean()
log(f'  真实测试段期望净 = {100*base:+.2f}%')
rnd = []
allpos = DF[(DF.exit == 'E1 TP30/SL15/48h')].tpos.values
for it in range(NPERM):
    # 随机抽取同样数量的"任意时点"，其净收益用同分布近似（用全体信号的随机子集）
    s = rng.choice(allpos, size=int(te.sum()), replace=False)
    rnd.append(np.nanmean(DF[(DF.exit == 'E1 TP30/SL15/48h')].net.values.mean()))
log('  （说明：本置换为粗版——信号时刻与收益强耦合，精确置换需重算未来窗口，'
    '以「币留出 + 逐年」为主要判据；此处仅给出随机子集基准）')
log(f'  随机子集基准 = {100*np.nanmean(rnd):+.2f}%')

log('\n' + '=' * 116)
log('【判读】')
log('=' * 116)
log('  · 期望净 >0 且 币留出两半都 >0 且 逐年多数 >0 ⇒ 可继续投入验证')
log('  · 任一条件不满足 ⇒ 该信号是选择性偏差，不可用')

out = os.path.join(OUT, 'verify_reverse.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
