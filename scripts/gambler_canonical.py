# -*- coding: utf-8 -*-
"""gambler_canonical · 唯一口径：参数全部按「分钟」定义，粒度只决定换算成几根

【硬约束】大周期持有上限 = 24 小时（1440 分钟），绝不持 5 天。
参数定义见 evidence/cycle/口径定义_唯一标准_2026-10-17.md

用法：python -X utf8 gambler_canonical.py --data D01
      python -X utf8 gambler_canonical.py --data D07   # 304 币 × 3 年，含熊市
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from data_registry import resolve

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'

# ============ 唯一口径（分钟）· 变量编号见 变量编号表_一号大周期算法_v1.0.md ============
VW_MIN = 240        # [V05] 低点窗口 240min = 4h（用户定，不可改）
CONF_MIN = 2880     # [V09] 确认窗口 2880min = 48h
HOLD_BIG_MIN = 1440 # [V16] 大周期持有上限 1440min = 24h（用户定，不可改）
HOLD_SML_MIN = 60   # [小周期] 持有 60min = 1h
ENTRY_RISE = 0.03   # [V10] 入场触发 3%（相对低点）
BIG_TP_MULT = 1.30  # [V14] 止盈 = 低点×1.30（相对入场 +26.21%）
BIG_SL_MULT = 0.95  # [V15] 止损 = 低点×0.95（相对入场 -7.77%）
SML_TP = 0.05       # [小周期] 止盈 +5%
SML_SL = 0.02       # [小周期] 止损 -2%
FEE = 0.005         # [V19] 成本 0.50%/笔
START = 100.0       # [V20] 本金
POS = 0.10          # [V21] 每笔仓位
MAXSYM = int(os.environ.get('SYMS', '60'))  # [V03] 币数上限

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='D01', help='D 编号：D01~D08（见《数据资产编号表_v1.0.md》）')
args = ap.parse_args()
DCODE, DATA_PATH, GRAN, DDESC = resolve(args.data)   # [V01][V02] 数据与粒度由 D 编号唯一决定

# 换算成根
VW = max(1, VW_MIN // GRAN)
CONF = max(1, CONF_MIN // GRAN)
HOLD_BIG = max(1, HOLD_BIG_MIN // GRAN)
HOLD_SML = max(1, HOLD_SML_MIN // GRAN)

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log(f'gambler_canonical · 数据 {DCODE} = {DDESC} · 粒度 {GRAN}m · 唯一口径（按分钟定义）')
log('=' * 100)
log(f'低点窗口 {VW_MIN}min = {VW} 根 | 确认 {CONF_MIN}min = {CONF} 根')
log(f'大周期持有 {HOLD_BIG_MIN}min = {HOLD_BIG} 根 (必须 24h) | 小周期 {HOLD_SML_MIN}min = {HOLD_SML} 根')
log(f'大周期 低点×{BIG_TP_MULT}/{BIG_SL_MULT} | 小周期 入场×{1+SML_TP}/{1-SML_SL}')
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
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
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

# 收集入场点：低点确认 + 之后涨 3%
ROWS = []
for sym, (o, h, l, c, v, tb, ot) in RAWS.items():
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
        entry = trig
        ROWS.append((ot[hit], sym, lo, entry, hit, n))
ROWS.sort(key=lambda x: x[0])
log(f'入场点 {len(ROWS):,}  耗时 {time.time()-t0:.0f}s')

# 按时间切分
ts = np.array([r[0] for r in ROWS])
if GRAN == 1:
    cut = ts[int(len(ts) * 0.75)]
    tr_mask = ts < cut; te_mask = ~tr_mask
else:
    cut = pd.Timestamp('2024-10-01').value // 10**6
    tr_mask = ts < cut; te_mask = ~tr_mask
log(f'训练 {tr_mask.sum():,}  测试 {te_mask.sum():,}')

def sim_big(rows):
    out = []
    for (ot, sym, lo, entry, hit, n) in rows:
        o, h, l, c = RAWS[sym][0], RAWS[sym][1], RAWS[sym][2], RAWS[sym][3]
        tp = lo * BIG_TP_MULT; sl = lo * BIG_SL_MULT
        ret = None; exit_k = None; kind = 'to'
        for k in range(hit + 1, min(hit + HOLD_BIG + 1, n)):
            ht = h[k] >= tp; hs = l[k] <= sl
            if ht and hs: ret = sl / entry - 1; exit_k = k; kind = 'sl'; break
            if ht: ret = tp / entry - 1; exit_k = k; kind = 'tp'; break
            if hs: ret = sl / entry - 1; exit_k = k; kind = 'sl'; break
        if ret is None:
            kk = min(hit + HOLD_BIG, n - 1)
            ret = c[kk] / entry - 1; exit_k = kk; kind = 'to'
        exit_ot = ot + int(exit_k - hit) * GRAN * 60000
        out.append((ot, sym, ret - FEE, hit, n, exit_ot, kind))
    return out

def sim_sml(rows):
    out = []
    for (ot, sym, lo, entry, hit, n) in rows:
        o, h, l, c = RAWS[sym][0], RAWS[sym][1], RAWS[sym][2], RAWS[sym][3]
        tp = entry * (1 + SML_TP); sl = entry * (1 - SML_SL)
        ret = None; exit_k = None; kind = 'to'
        for k in range(hit + 1, min(hit + HOLD_SML + 1, n)):
            ht = h[k] >= tp; hs = l[k] <= sl
            if ht and hs: ret = -SML_SL; exit_k = k; kind = 'sl'; break
            if ht: ret = SML_TP; exit_k = k; kind = 'tp'; break
            if hs: ret = -SML_SL; exit_k = k; kind = 'sl'; break
        if ret is None:
            kk = min(hit + HOLD_SML, n - 1)
            ret = c[kk] / entry - 1; exit_k = kk; kind = 'to'
        exit_ot = ot + int(exit_k - hit) * GRAN * 60000
        out.append((ot, sym, ret - FEE, hit, n, exit_ot, kind))
    return out

def h3_filter(trades_all, is_test):
    """h3>=1.0：同币最近 3 次「已平仓」结果全部止盈。
    防前视：入场时刻还没平仓的交易，结果不得用于判断。"""
    trades_all = sorted(trades_all, key=lambda x: x[0])
    pending = {}   # sym -> [(exit_ot, win)] 未平仓
    closed = {}    # sym -> [win] 已平仓
    keep = []
    audit = {'checks': 0, 'pending_at_entry': 0, 'h3_pass': 0}
    for i, (ot, sym, ret, hit, n, exit_ot, kind) in enumerate(trades_all):
        # 先把 exit_ot <= 当前入场时间的转成已平仓
        still = []
        for (e_ot, win) in pending.get(sym, []):
            if e_ot <= ot:
                closed.setdefault(sym, []).append(win)
            else:
                still.append((e_ot, win))
        pending[sym] = still
        c = closed.get(sym, [])
        h3 = (len(c) >= 3 and all(c[-3:]))
        audit['checks'] += 1
        if still: audit['pending_at_entry'] += 1
        if h3:
            audit['h3_pass'] += 1
            if is_test[i]:
                keep.append((ot, sym, ret))
        pending.setdefault(sym, []).append((exit_ot, ret > 0))
    return keep, audit

def run_line(sim_fn, name):
    allt = sim_fn(ROWS)
    is_te = [bool(te_mask[i]) for i in range(len(ROWS))]
    # ROWS 与 sim 输出顺序一致
    base = np.array([t[2] for i, t in enumerate(allt) if is_te[i]])
    cap0 = START; peak0 = START; mdd0 = 0.0
    for r in base:
        cap0 *= (1 + POS * r); peak0 = max(peak0, cap0); mdd0 = max(mdd0, (peak0 - cap0) / peak0)
    log(f'\n【{name}·不筛基线】笔数 {len(base):,}  均值 {base.mean():+.2%}  中位 {np.median(base):+.2%}  胜率 {(base>0).mean():.1%}')
    log(f'  {START:.0f} → {cap0:,.2f} 元  回撤 {mdd0:.1%}  最差 {base.min():+.1%}')
    kinds = {}
    for i, t in enumerate(allt):
        if is_te[i]:
            kinds[t[6]] = kinds.get(t[6], 0) + 1
    tot = sum(kinds.values()) or 1
    label = {'tp': '止盈', 'sl': '止损', 'to': '超时'}
    log('  出口分布: ' + '  '.join(f'{label.get(k,k)} {v:,}（{v/tot:.1%}）' for k, v in sorted(kinds.items())))
    keep, audit = h3_filter(allt, is_te)
    log(f'  【前视审计】判断 {audit["checks"]:,} 次 | 入场时前序尚未平仓 {audit["pending_at_entry"]:,} 次'
        f' | h3 成立 {audit["h3_pass"]:,} 次')
    if not keep:
        log(f'\n【{name}】无交易'); return
    rets = np.array([r[2] for r in keep])
    cap = START; peak = START; mdd = 0.0
    for r in rets:
        cap *= (1 + POS * r)
        peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak)
    log(f'\n【{name}】h3>=1.0  笔数 {len(rets):,}')
    log(f'  单笔均值 {rets.mean():+.2%}  中位 {np.median(rets):+.2%}  胜率 {(rets>0).mean():.1%}')
    log(f'  {START:.0f} → {cap:,.2f} 元  回撤 {mdd:.1%}  最差 {rets.min():+.1%}')
    # 按月
    months = {}
    for r in keep:
        m = pd.Timestamp(r[0], unit='ms').strftime('%Y-%m')
        months.setdefault(m, []).append(r[2])
    log('  按月：')
    for m in sorted(months):
        a = np.array(months[m])
        log(f'    {m}  {len(a):>5} 笔  均值 {a.mean():+.2%}  胜率 {(a>0).mean():.1%}')

run_line(sim_big, '大周期 (24h)')
run_line(sim_sml, '小周期 (1h)')

out = os.path.join(OUT, f'canonical_{DCODE}_summary.txt')  # 报告名带 D 编号，永不覆盖
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
