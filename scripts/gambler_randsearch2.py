# -*- coding: utf-8 -*-
"""gambler_randsearch2.py — 随机搜索的发现 → 交易模拟 + 双重留出验证
承 randsearch（job-47368-42）：60,000 次随机组合（训练段搜→测试段验），挖出人工从未想到的
  维度 **hour18~24（UTC 18:00~24:00）**：
    hour18~24 & r288>=30% & volr6>=5      训练 13.183% → 测试 15.315%（14.17x）
    hour18~24 & volr24>=5 & r288>=30%     训练 12.601% → 测试 15.812%（14.63x）
    hour18~24 & r96>=10% & r288>=30%      测试 18.232%（16.87x）
  过拟合诊断：top-100 中 67% 测试段仍 >2x，33% 崩掉

本脚本三步：
  ① 对 top 组合做交易模拟（+50%TP/-15%SL/24hTO，扣 0.09%）
  ② 单独检验 hour18~24 的独立贡献（去掉其他条件）
  ③ **币留出**（奇序币训练 → 偶序币测试）+ **双留出**，排除"少数币主导"
"""
import os, gc, time
import numpy as np
import pandas as pd

t0t = time.time()
D = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
SRC = os.path.join(D, 'klines_5m_3y.csv')
OUT = os.path.join(EV, 'gambler_randsearch2.txt')
WDAY = 288
H48 = WDAY * 2
FEE = 0.0004 + 0.0005
TP, SL, TO = 0.50, -0.15, WDAY

buf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    buf.append(s)

log('=' * 100)
log('gambler_randsearch2 · 33 随机搜索的发现 → 交易模拟 + 币留出验证')
log('=' * 100)

RAW = []
for ch in pd.read_csv(SRC, usecols=['symbol','open_time','close','high','low','quote_volume'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32',
                             'high':'float32','low':'float32','quote_volume':'float32'}, chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 3000: continue
        g = g.sort_values('open_time')
        RAW.append((sym, g['open_time'].values.astype('int64'),
                    g['close'].values.astype('float64'), g['high'].values.astype('float64'),
                    g['low'].values.astype('float64'), g['quote_volume'].values.astype('float64')))
    del ch; gc.collect()
log(f'币 {len(RAW)}   读盘 {time.time()-t0t:.0f}s')

def simulate(lo, c, h, l, tp=TP, sl=SL, to=TO):
    n = len(c)
    end = min(n - 1, lo + to)
    if end <= lo + 1: return None
    tp_p = c[lo] * (1 + tp); sl_p = c[lo] * (1 + sl)
    ht = np.flatnonzero(h[lo+1:end+1] >= tp_p)
    hs = np.flatnonzero(l[lo+1:end+1] <= sl_p)
    it = ht[0] if len(ht) else 10**9
    isl = hs[0] if len(hs) else 10**9
    if it == 10**9 and isl == 10**9: return c[end] / c[lo] - 1.0 - FEE, 'timeout'
    if it <= isl: return tp - FEE, 'tp'
    return sl - FEE, 'sl'

B = {}      # 交易结果
DET = []    # 检测结果：(组名, symbol, t, true, 是否训练段)
def add(k, r, kind):
    B.setdefault(k, []).append((r, kind))

groups = ['G1 hour18~24 & r288>=30% & volr6>=5',
          'G2 hour18~24 & volr24>=5 & r288>=30%',
          'G3 hour18~24 & r96>=10% & r288>=30%',
          'G4 仅 hour18~24',
          'G5 仅 r288>=30%',
          'G6 hour18~24 & r288>=30%']
ALLT = []
for (sym, ot, c, h, l, q) in RAW:
    n = len(c)
    if n < WDAY * 4: continue
    hs = pd.Series(h); ls = pd.Series(l); qs = pd.Series(q)
    lo288 = ls.rolling(WDAY, min_periods=72).min().values
    over = c / np.maximum(lo288, 1e-12) - 1.0
    r96 = np.full(n, np.nan); r96[96:] = c[96:] / c[:-96] - 1.0
    r288 = np.full(n, np.nan); r288[WDAY:] = c[WDAY:] / c[:-WDAY] - 1.0
    base_q = qs.rolling(WDAY * 30, min_periods=WDAY).mean().values
    volr = np.where(base_q > 0, q / np.maximum(base_q, 1e-12), np.nan)
    volr6 = pd.Series(volr).rolling(6, min_periods=3).mean().values
    volr24 = pd.Series(volr).rolling(24, min_periods=6).mean().values
    hh = (ot // 3600000) % 24
    h1824 = (hh >= 18) | (hh < 0)          # UTC 18~24
    fmax = hs.rolling(H48, min_periods=1).max().shift(-H48).values
    f_up = fmax / c - 1.0
    dd = np.full(n, np.nan)
    ok = np.isfinite(r96) & np.isfinite(r288) & np.isfinite(volr6) & np.isfinite(volr24) & np.isfinite(f_up)
    base = (over >= 0.03) & (over <= 0.40) & (volr24 >= 1)
    sel = ok & base
    idx = np.flatnonzero(sel)
    if len(idx) > 0:
        for t in idx[::3]:
            e = min(n - 1, t + H48); seg = h[t+1:e+1]
            if len(seg) < 50: continue
            ipk = int(np.argmax(seg)); pk = seg[ipk]
            e2 = min(n - 1, t + 1 + ipk + WDAY)
            if e2 > t + 1 + ipk: dd[t] = 1.0 - l[t+1+ipk:e2+1].min() / max(pk, 1e-12)
    RULES = [
        (groups[0], sel & h1824 & (r288 >= 0.30) & (volr6 >= 5)),
        (groups[1], sel & h1824 & (volr24 >= 5) & (r288 >= 0.30)),
        (groups[2], sel & h1824 & (r96 >= 0.10) & (r288 >= 0.30)),
        (groups[3], sel & h1824),
        (groups[4], sel & (r288 >= 0.30)),
        (groups[5], sel & h1824 & (r288 >= 0.30)),
    ]
    for nm, m in RULES:
        cand = np.flatnonzero(m)
        last = -10 ** 9
        for t in cand:
            if t - last < 12: continue
            r = simulate(t, c, h, l)
            if r is None: continue
            add(nm, r[0], r[1]); last = t
    # 记录检测样本（用于双重留出）
    mm = sel & h1824 & (r288 >= 0.30) & (volr6 >= 5)
    for t in np.flatnonzero(np.isfinite(dd) & mm):
        DET.append((sym, int(ot[t]), 1 if (f_up[t] >= 0.50 and dd[t] >= 0.30) else 0))
    ALLT.append((sym, ot))
    del hs, ls, qs; gc.collect()

log('')
log('=' * 100)
log(f'【① 交易模拟】+{100*TP:.0f}% 止盈 / {100*SL:.0f}% 止损 / 24h 超时；扣 {100*FEE:.2f}%')
log('=' * 100)
log(f'  {"组":<44} {"笔数":>7} {"期望净%":>9} {"中位%":>8} {"胜率%":>7} {"止盈%":>7} {"止损%":>7} {"超时%":>7}')
for k in groups:
    arr = B.get(k, [])
    if not arr:
        log(f'  {k:<44} {0:>7}'); continue
    rs = np.array([x[0] for x in arr]); kinds = [x[1] for x in arr]
    log(f'  {k:<44} {len(rs):>7,} {100*rs.mean():>8.3f}% {100*np.median(rs):>7.2f}% '
        f'{100.0*(rs > 0).mean():>6.1f}% '
        f'{100.0*np.mean([x=="tp" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="sl" for x in kinds]):>6.1f}% '
        f'{100.0*np.mean([x=="timeout" for x in kinds]):>6.1f}%')

# ---------- ② / ③ 双重留出验证 ----------
if DET:
    det = pd.DataFrame(DET, columns=['sym', 't', 'y'])
    base = det['y'].mean()
    log('')
    log('=' * 100)
    log(f'【② 双留出验证（G1 条件）】检测样本 {len(det):,}   基准真局率 {100*base:.3f}%')
    log('=' * 100)
    # 时间留出
    cut = det['t'].quantile(0.70)
    trm = det['t'] <= cut; tem = ~trm
    if tem.sum() > 20:
        rt = det[tem]['y'].mean()
        log(f'  时间留出：训练 {int(trm.sum()):,} 测试 {int(tem.sum()):,} → 测试真局率 {100*rt:.3f}%（{rt/base:.2f}x）')
    # 币留出
    syms_sorted = sorted(det['sym'].unique())
    even = set(syms_sorted[::2])
    trc = det['sym'].isin(even); tec = ~trc
    if tec.sum() > 20:
        rc = det[tec]['y'].mean()
        log(f'  币留出：训练 {int(trc.sum()):,}（{len(even)}币） 测试 {int(tec.sum()):,}（{len(syms_sorted)-len(even)}币）'
            f' → 测试真局率 {100*rc:.3f}%（{rc/base:.2f}x）')
    # 双留出
    both = trm & trc; bothte = tem & tec
    if bothte.sum() > 20:
        rb = det[bothte]['y'].mean()
        log(f'  双重留出（时间+币都不重叠）：训练 {int(both.sum()):,} 测试 {int(bothte.sum()):,}'
            f' → 测试真局率 {100*rb:.3f}%（{rb/base:.2f}x）')
    # UTC 时段再确认：单独看 hour18-24 vs 其他
    log('')
    log('  对照：仅"r288>=30% & volr6>=5"（不限时段）与"限时段"的差异需另测，见下')

log('')
log('=' * 100)
log('【判读】')
log('=' * 100)
log('  · 若 G4(仅 hour18~24) 期望净 ≈ 0 ⇒ 时段本身无独立价值，必须与其他条件组合')
log('  · 若双重留出下真局率仍 >3x ⇒ 该发现跨币、跨时间稳定，值得进一步验证')
log('  · 全程无前视：所有条件在 t 时刻可观测')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
log('')
log(f'报告已写 {OUT}')
log(f'TOTAL {time.time()-t0t:.0f}s')
