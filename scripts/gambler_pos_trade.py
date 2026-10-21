# -*- coding: utf-8 -*-
"""gambler_pos_trade.py — Q1 落地：把「位置」变成可交易规则，直接算期望净

背景（job-24708-14 stage 的结果）：
    局内样本里，**位置比模型强**：
      已涨 15~25%  → 未来48h 再拿 +30% 的概率 10.1%
      已涨 25~40%  → 14.3%
      已涨 40~60%  → 27.3%   ★
      已涨 60~100% → 30.5%   ★
      已涨 100~200%→  8.8%   （崩）
      已涨 200%+   →  0.0%
    而整套 GBDT（AUC 0.625 / 回归 R²≈0）反而不如这条位置规则。

【本脚本】对每个位置档 × 每种退出规则，**直接算扣费后的期望净收益**，回答：
    位置规则能不能赚钱？赚多少？哪种退出最合适？

【入场】局内点（已涨 >=15%），按 over 分档
【退出规则（四种）】
    R1 固定：+30% 止盈 / -15% 止损 / 48h 超时结算
    R2 固定宽：+50% 止盈 / -20% 止损 / 48h
    R3 移动止损：从窗口内峰值回落 30% 离场（无固定止盈），48h 封顶
    R4 移动止损 + 止盈：+50% 止盈 或 回落 30% 离场
【扣费】0.09%（maker0.04+taker0.05）
【对照】不筛位置 = 全部局内点
【切分】前 70% 训练 / 后 30% 测试，两段都报，只以测试段下结论
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\butterfly'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
WDAY = 288
W = 576            # 48h = 576 根 5m
FEE = 0.0009

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_pos_trade · 位置档 × 退出规则 → 期望净收益（含费）')
log('=' * 116)

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
ts = C.index.values; T, N = C.shape
t_cut = int(T * 0.70)
log(f'币 {C.shape[1]}   矩阵 {C.shape}   耗时 {time.time()-t0:.0f}s')

OV, TID, R = [], [], {k: [] for k in ('R1', 'R2', 'R3', 'R4')}
TV = []   # 峰值涨幅（诊断用）

def exits(winh, winl, winc, c0):
    """winh/winl/660: (k,W) 未来窗口；返回四种规则的净收益（已扣费）"""
    k = winh.shape[0]
    cummax = np.maximum.accumulate(winh, axis=1)
    dd = 1.0 - winc / np.maximum(cummax, 1e-12)     # 自峰值回落
    res = {}
    for tag, tp, sl in (('R1', 0.30, 0.15), ('R2', 0.50, 0.20)):
        tp_px = c0 * (1 + tp); sl_px = c0 * (1 - sl)
        it = np.argmax(winh >= tp_px[:, None], axis=1)
        il = np.argmax(winl <= sl_px[:, None], axis=1)
        ht = (winh >= tp_px[:, None]).any(axis=1)
        hl = (winl <= sl_px[:, None]).any(axis=1)
        # 都没触发 → 用最后一根 close 结算
        settle = winc[:, -1] / c0 - 1.0
        v = np.where(hl & (~ht | (il <= it)), -sl, np.where(ht, tp, settle))
        res[tag] = v - FEE
    # R3 移动止损：回落 30% 离场，无固定止盈
    trig = dd >= 0.30
    it3 = np.argmax(trig, axis=1)
    h3 = trig.any(axis=1)
    px3 = np.take_along_axis(winl, np.minimum(it3, W - 1)[:, None], axis=1)[:, 0]
    res['R3'] = np.where(h3, px3 / c0 - 1.0, winc[:, -1] / c0 - 1.0) - FEE
    # R4 = 移动止损 30% 或 +50% 止盈
    tp_px = c0 * 1.50
    it4 = np.argmax((winh >= tp_px[:, None]) | trig, axis=1)
    hit4 = ((winh >= tp_px[:, None]) | trig).any(axis=1)
    at_tp = (winh >= tp_px[:, None])
    first_is_tp = at_tp[np.arange(k), np.minimum(it4, W - 1)]
    px4 = np.where(first_is_tp, tp_px, np.take_along_axis(winl, np.minimum(it4, W - 1)[:, None], axis=1)[:, 0])
    res['R4'] = np.where(hit4, px4 / c0 - 1.0, winc[:, -1] / c0 - 1.0) - FEE
    return res

for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4 + W:
        continue
    ls = pd.Series(l); hs = pd.Series(h)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        over = c / np.maximum(lo288, 1e-12) - 1.0
    ok = np.isfinite(over) & (over >= 0.15) & (np.arange(n) < n - W - 1)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        continue
    # 抽样上限，控制内存
    if len(idx) > 60000:
        idx = idx[::max(1, len(idx) // 60000)]
    wh = np.lib.stride_tricks.sliding_window_view(h, W)[idx]
    wl = np.lib.stride_tricks.sliding_window_view(l, W)[idx]
    wc = np.lib.stride_tricks.sliding_window_view(c, W)[idx]
    c0 = c[idx]
    r = exits(wh, wl, wc, c0)
    OV.append(over[idx].astype('float32'))
    TID.append(np.searchsorted(ts, ot)[idx].astype('int64'))
    TV.append((wh.max(axis=1) / c0 - 1.0).astype('float32'))
    for k in R:
        R[k].append(r[k].astype('float32'))
    del wh, wl, wc
del RAW, C; gc.collect()

OV = np.concatenate(OV); TID = np.concatenate(TID); TV = np.concatenate(TV)
for k in R:
    R[k] = np.concatenate(R[k])
log(f'局内候选样本 {len(OV):,}   耗时 {time.time()-t0:.0f}s')

BINS = [(0.15, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 1.00), (1.00, 2.00), (2.00, 99.0)]
TAGS = ['15~25%', '25~40%', '40~60%', '60~100%', '100~200%', '200%+']

for seg, mask in (('测试段（后30%）', TID >= t_cut), ('训练段（前70%）', TID < t_cut)):
    log('\n' + '=' * 116)
    log(f'【{seg}】位置档 × 退出规则   样本 {int(mask.sum()):,}')
    log('=' * 116)
    for tag in ('R1', 'R2', 'R3', 'R4'):
        lbl = {'R1': 'R1 +30%/-15%/48h', 'R2': 'R2 +50%/-20%/48h',
               'R3': 'R3 移动止损 回落30%', 'R4': 'R4 +50% 或 回落30%'}[tag]
        log(f'\n  {lbl}')
        log(f'    {"位置档":<12}{"笔数":>9}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"≥+50%":>8}{"≥+100%":>8}{"峰值中位%":>10}')
        for (a, b), t in zip(BINS, TAGS):
            m = mask & (OV >= a) & (OV < b)
            if m.sum() < 30:
                continue
            v = R[tag][m]
            log(f'    {t:<12}{m.sum():>9,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
                f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanmean(v >= 0.50):>8.1f}'
                f'{100*np.nanmean(v >= 1.00):>8.1f}{100*np.nanmedian(TV[m]):>10.1f}')
        m = mask
        v = R[tag][m]
        log(f'    {"全部局内":<12}{m.sum():>9,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
            f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanmean(v >= 0.50):>8.1f}'
            f'{100*np.nanmean(v >= 1.00):>8.1f}{100*np.nanmedian(TV[m]):>10.1f}')

# 最优格汇总
log('\n' + '=' * 116)
log('【汇总】测试段 期望净 > 0 的（位置档, 退出规则）组合')
log('=' * 116)
rows = []
te = TID >= t_cut
for tag in ('R1', 'R2', 'R3', 'R4'):
    for (a, b), t in zip(BINS, TAGS):
        m = te & (OV >= a) & (OV < b)
        if m.sum() < 50:
            continue
        v = R[tag][m]
        rows.append((t, tag, int(m.sum()), float(np.nanmean(v)),
                     float(np.nanmedian(v)), float(np.nanmean(v > 0))))
rows.sort(key=lambda x: -x[3])
log(f'  {"位置档":<12}{"规则":>4}{"笔数":>9}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}')
for t, tag, n, e, md, wr in rows[:20]:
    mark = ' ★' if e > 0 else ''
    log(f'  {t:<12}{tag:>4}{n:>9,}{100*e:>+10.2f}{100*md:>9.2f}{100*wr:>8.1f}{mark}')

out = os.path.join(OUT, 'pos_trade_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
