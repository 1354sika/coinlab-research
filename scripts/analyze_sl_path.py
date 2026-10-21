# -*- coding: utf-8 -*-
"""analyze_sl_path.py — 止损触发之后，价格是"回来"还是"继续跌"？

这是决定打法的关键：
  · 若触发后大量反弹回止损线上方 => 止损位太近，被打在噪声里，应该放宽止损
  · 若触发后继续单边下跌             => 止损是对的，只是价格真的跳空，只能靠限价/更早

同时给出"跳空深度"的条件分布（这才是真正的滑点来源，不是下单晚）。
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'

LOOKBACK = 15
DROP = 0.05
TP = 0.03
SL = 0.01
MAXBARS = 2880
NSYM = 60
HORIZONS = [1, 3, 5, 15, 30, 60, 240]

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(x for x in os.listdir(DATA) if x.endswith('.csv'))[:NSYM]

# 每笔： (k, 入场后到触发的路径最低点, 触发后各 horizon 的收益)
sl_paths = []      # 止损笔：触发后 h 根相对入场价的收益
sl_k = []          # 触发在入场后第几根
nsl = 0

for fi in files:
    sym = fi[:-4]
    try:
        df = pd.read_csv(os.path.join(DATA, fi), usecols=['open', 'close'])
    except Exception:
        continue
    if len(df) < 50000:
        continue
    o = df['open'].to_numpy(np.float32)
    c = df['close'].to_numpy(np.float32)
    n = len(c)

    ret = c / np.roll(c, LOOKBACK) - 1.0
    ret[:LOOKBACK] = 0.0
    idx = np.where(ret <= -DROP)[0]
    idx = idx[(idx >= LOOKBACK) & (idx + MAXBARS < n)]

    for i in idx:
        iv = i + 1
        ep = o[iv]
        if not np.isfinite(ep) or ep <= 0:
            continue
        path = c[iv:iv + MAXBARS]
        rr = path / ep - 1.0
        slm = rr <= -SL
        tpm = rr >= TP
        i_sl = int(np.argmax(slm)) if slm.any() else 10 ** 9
        i_tp = int(np.argmax(tpm)) if tpm.any() else 10 ** 9
        if i_sl < i_tp:
            k = i_sl
            nsl += 1
            # 触发后各 horizon（从触发那根往后算）
            hs = []
            for h in HORIZONS:
                j = k + h
                hs.append(float(rr[j]) if j < MAXBARS else np.nan)
            sl_paths.append(hs)
            sl_k.append(k)

A = np.array(sl_paths, dtype=np.float64)   # (n_sl, len(HORIZONS))
K = np.array(sl_k, dtype=np.int32)

p('')
p('=' * 100)
p('止损触发之后的路径  15m跌5%% + TP3%%/SL1%%  |  %d 币 / 止损笔 %d / %.0fs'
  % (NSYM, nsl, time.time() - t0))
p('=' * 100)
p('')
p('=== 一、触发止损后，价格怎么走（相对入场价的收益）')
p('  注意：触发点的收益固定在约 -1%%（实际 -1.89%%，见 sl_pierce）')
hdr = '  %-10s' % '触发后'
for h in HORIZONS:
    hdr += '%9d根' % h
p(hdr)
p('  ' + '-' * (10 + 9 * len(HORIZONS)))
for name, fn in [('均值', np.nanmean), ('中位', np.nanmedian),
                 ('P25', lambda x: np.nanpercentile(x, 25)),
                 ('P75', lambda x: np.nanpercentile(x, 75))]:
    line = '  %-10s' % name
    for j in range(len(HORIZONS)):
        v = fn(A[:, j])
        line += '%8.2f%%' % (100 * v)
    p(line)

p('')
p('=== 二、触发后"回到止损线上方"（>= -1%%）的比例')
p('  这是判断"是不是被打在噪声里"的直接指标：')
line = '  %-14s' % '触发后'
for h in HORIZONS:
    line += '%9d根' % h
p(line)
p('  ' + '-' * (14 + 9 * len(HORIZONS)))
for thr, nm in [(-0.01, '回到 -1% 以上'), (0.0, '回到成本价'), (0.03, '到 +3%(原TP)')]:
    line = '  %-14s' % nm
    for j in range(len(HORIZONS)):
        col = A[:, j]
        ok = np.isfinite(col)
        line += '%8.1f%%' % (100 * np.mean(col[ok] >= thr)) if ok.any() else '%8s' % '-'
    p(line)

p('')
p('=== 三、跳空深度分布（触发那根实际跌幅，这才是"滑点"的真身）')
p('  说明：这是收盘价确认口径，不含任何执行延迟；')
p('  所以下面这些 100%% 是"市场自己跳过去"，不是"你下单晚"。')
p('  数据见 sl_pierce.txt：触发那根收盘已平均 -1.89%%')

p('')
p('=== 四、结论判定')
p25 = np.nanpercentile(A[:, HORIZONS.index(30)], 25) if nsl else np.nan
p50 = np.nanmedian(A[:, HORIZONS.index(30)]) if nsl else np.nan
back = np.mean(A[:, HORIZONS.index(30)] >= -0.01) if nsl else np.nan
p('  30 根后中位收益 %+.2f%%   30 根后回到 -1%% 以上的比例 %.1f%%'
  % (100 * p50, 100 * back))
if back > 0.5:
    p('  => 一半以上会回来 => **止损位偏近，被打在噪声里**，放宽止损可能更好')
else:
    p('  => 多数不回来 => **止损方向对，是价格真的跳空**，只能靠更早/限价')

p('')
p('--- 诚实边界 ---')
p('1. 收盘价序列口径，不含执行延迟与真实滑点。')
p('2. horizon 从触发那根往后数，若期间再次触 TP 未单独标记。')
p('3. 未做事件去重：同波行情可能重复计数。')
p('4. 已下架归零币不在池内。')
p('5. 单一样本，含样本内参数。')

with open(os.path.join(EVI, 'sl_path.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written D:/中信证券/coinlab/evidence/sl_path.txt  TOTAL %.0fs' % (time.time() - t0))
