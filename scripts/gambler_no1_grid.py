# -*- coding: utf-8 -*-
"""gambler_no1_grid · 一号大周期算法 · 止盈三档网格（+8% / +10% / +15%）

所有时间参数单位一律「分钟」（见 变量编号表 v1.1 §0），根数由 V02 自动换算：
  低点 = 过去 V05=240min 最低（纯回看，无前视）→ 1m: 240 根 / 5m: 48 根
  入场 = 低点后 V09=2880min 内，high >= 低点 × 1.03，入场价 = 低点 × 1.03
  SL   = 低点 × 0.95（相对入场 -7.767%，固定不动）
  TP   = 入场价 × (1 + {8%, 10%, 15%})
  持有 = V16=1440min（24h）→ 1m: 1440 根 / 5m: 288 根
  成本 0.50%/笔，本金 100，仓位 10%，不筛任何条件

用法：python -X utf8 gambler_no1_grid.py [--gran 1m|5m]
"""
import os, time, argparse
import numpy as np
import pandas as pd
from data_registry import resolve

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

# 变量编号见 变量编号表_一号大周期算法_v1.1.md —— 引用参数必须带编号
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='D01', help='D 编号：D01~D08（见《数据资产编号表_v1.0.md》）')
ARGS = ap.parse_args()
DCODE, DATA_PATH, GRAN, DDESC = resolve(ARGS.data)   # [V01][V02] 数据与粒度由 D 编号唯一决定

VW_MIN = 240            # [V05] 低点窗口 240 分钟 = 4h（用户定，不可改）
CONF_MIN = 2880         # [V09] 确认窗口 2880 分钟 = 48h
HOLD_MIN = 1440         # [V16] 持有上限 1440 分钟 = 24h（用户定，不可改）
VW = VW_MIN // GRAN     # → 1m: 240 根 / 5m: 48 根
CONF = CONF_MIN // GRAN # → 1m: 2880 根 / 5m: 576 根
HOLD = HOLD_MIN // GRAN # → 1m: 1440 根 / 5m: 288 根
ENTRY_RISE = 0.03       # [V10] 入场触发 3%
SL_MULT = 0.95          # [V15] 止损 = 低点×0.95（相对入场 -7.77%）
TPS = [0.08, 0.10, 0.15]  # [V14] 止盈三档（相对入场价）
FEE = 0.005             # [V19] 成本 0.50%/笔
START = 100.0           # [V20] 本金
POS = 0.10              # [V21] 每笔仓位
MAXSYM = int(os.environ.get('SYMS', '60'))  # [V03] 币数上限

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_no1_grid · 一号算法 · 止盈三档（+8% / +10% / +15%），止损固定 -7.77%')
log('=' * 100)
log(f'数据 {DCODE} = {DDESC} | 粒度 {GRAN}m')
log(f'低点窗口 {VW_MIN}min = {VW} 根(4h) | 确认窗口 {CONF_MIN}min = {CONF} 根(48h) | 持有 {HOLD_MIN}min = {HOLD} 根(24h)')
log(f'入场 = 低点 × {1+ENTRY_RISE:.2f} | 止损 = 低点 × {SL_MULT:.2f}'
    f'（相对入场 {SL_MULT/(1+ENTRY_RISE)-1:+.2%}）')
log(f'成本 {FEE*100:.2f}%/笔 | 本金 {START:.0f} | 仓位 {POS:.0%}')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(DATA_PATH, usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, n):
    lo_w = pd.Series(l).rolling(VW, min_periods=1).min().values
    idx = []
    i = VW
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            idx.append(i); i += max(5, VW // 8)
        else:
            i += 1
    return idx


ROWS = []
for sym, (o, h, l, c, ot) in RAWS.items():
    n = len(c)
    for lo_i in find_lows(l, n):
        lo = l[lo_i]
        if lo <= 0: continue
        trig = lo * (1 + ENTRY_RISE)
        j = lo_i + 1
        end = min(n - 1, lo_i + CONF)
        hit = -1
        while j <= end:
            if h[j] >= trig:
                hit = j; break
            if l[j] < lo: break
            j += 1
        if hit < 0: continue
        ROWS.append((ot[hit], sym, lo, hit, n))
ROWS.sort(key=lambda x: x[0])
log(f'入场点 {len(ROWS):,}  耗时 {time.time()-t0:.0f}s')

ts = np.array([r[0] for r in ROWS])
cut = ts[int(len(ts) * 0.75)]
te_mask = ts >= cut
log(f'训练 {int((~te_mask).sum()):,}  测试 {int(te_mask.sum()):,}')


def sim(tp_pct):
    out = []
    for (ot, sym, lo, hit, n) in ROWS:
        h, l, c = RAWS[sym][1], RAWS[sym][2], RAWS[sym][3]
        entry = lo * (1 + ENTRY_RISE)
        tp_px = entry * (1 + tp_pct)
        sl_px = lo * SL_MULT
        ret = None; kind = 'to'
        for k in range(hit + 1, min(hit + HOLD + 1, n)):
            ht = h[k] >= tp_px; hs = l[k] <= sl_px
            if ht and hs: ret = sl_px / entry - 1; kind = 'sl'; break
            if ht: ret = tp_px / entry - 1; kind = 'tp'; break
            if hs: ret = sl_px / entry - 1; kind = 'sl'; break
        if ret is None:
            kk = min(hit + HOLD, n - 1)
            ret = c[kk] / entry - 1; kind = 'to'
        out.append((ot, ret - FEE, kind))
    return out


for tp in TPS:
    allt = sim(tp)
    te = [t for i, t in enumerate(allt) if te_mask[i]]
    rets = np.array([t[1] for t in te])
    kinds = {}
    for t in te:
        kinds[t[2]] = kinds.get(t[2], 0) + 1
    tot = len(rets) or 1
    cap = START; peak = START; mdd = 0.0
    for r in rets:
        cap *= (1 + POS * r); peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak)
    log(f'\n【TP +{tp:.0%} / SL {SL_MULT/(1+ENTRY_RISE)-1:+.2%} / 24h】')
    log(f'  笔数 {len(rets):,}  均值 {rets.mean():+.2%}  中位 {np.median(rets):+.2%}'
        f'  胜率 {(rets>0).mean():.1%}')
    log(f'  {START:.0f} → {cap:,.2f} 元  回撤 {mdd:.1%}  最差 {rets.min():+.1%}')
    lab = {'tp': '止盈', 'sl': '止损', 'to': '超时'}
    log('  出口: ' + '  '.join(f'{lab.get(k,k)} {v:,}（{v/tot:.1%}）' for k, v in sorted(kinds.items())))
    p_need = abs(SL_MULT/(1+ENTRY_RISE)-1) + FEE
    p_need = p_need / (tp + abs(SL_MULT/(1+ENTRY_RISE)-1))
    log(f'  盈亏平衡胜率 {p_need:.1%}（实际 {(rets>0).mean():.1%}）')

out = os.path.join(OUT, f'no1_grid_{DCODE}_summary.txt')  # 报告名带 D 编号，永不覆盖
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
