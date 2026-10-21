# -*- coding: utf-8 -*-
"""gambler_bidir · 双向对称：跌完之后是不是一定涨？反向预测有没有超额？

用户问题（2026-10-17）：
    「那么跌之外就就一定涨吗？能不能反向预测？
      我认为蝴蝶效应，在预测跌的同时反向预测涨会有更多的意想不到的收获。」

【要测的四格】
                 未来涨>=30%        未来跌>=30%
  过去跌过    A: 跌→涨（均值回归）  B: 跌→跌（动量延续）
  过去涨过    C: 涨→涨（趋势延续）  D: 涨→跌（见顶）

  如果 A 格 lift 高 ⇒ 用「跌」预测「涨」成立（反向预测）
  如果 B 格 lift 高 ⇒ 跌之后继续跌（惯性）
  如果 D 格 lift 高 ⇒ 涨之后必跌（用户核心假设）
  如果所有格 lift≈1 ⇒ 过去涨跌对未来无信息

【口径】
  条件：过去 1h/4h/24h 的涨跌幅（t 时刻可见，无前视）
  目标：未来 48h（576 根 5m）内 最高>=+30% / 最低<=-30%
  分档：±5% 内 / 5~10% / 10~20% / >20%
  训练段（前70%）定档位，测试段（后30%）出结论
  扣费 0.09%（maker 双边 + 滑点）

【额外】做多/做空两个方向的交易期望：
    做多 TP+30/SL-15/48h 超时
    做空 TP+30(价格跌30%)/SL-15(价格涨15%)/48h 超时
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_5m_3y.csv')
HOLD = 576          # 48h
FEE = 0.0009

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_bidir · 双向对称：跌→涨 / 跌→跌 / 涨→涨 / 涨→跌')
log('=' * 116)

# ---------- 1. 读数据 ----------
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
        RAW.append((sym,
                    g['close'].values.astype('float64'),
                    g['high'].values.astype('float64'),
                    g['low'].values.astype('float64')))
    del ch; gc.collect()

log(f'币 {len(RAW)}   读盘 {time.time()-t0:.0f}s')


def fwd_max(a, k):
    """fwd_max[i] = max(a[i : i+k])，末尾不足 k 的置 nan"""
    s = pd.Series(a)
    r = s.rolling(k, min_periods=k).max().shift(-(k - 1)).values
    return r


def fwd_min(a, k):
    s = pd.Series(a)
    r = s.rolling(k, min_periods=k).min().shift(-(k - 1)).values
    return r


# ---------- 2. 逐币算条件 + 目标 ----------
# 分档边界（对称）
BINS = [-np.inf, -0.20, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.20, np.inf]
LAB = ['<-20%', '-20~-10', '-10~-5', '-5~-1', '±1%', '+1~5', '+5~10', '+10~20', '>+20%']

SCALES = [('r1h', 12), ('r4h', 48), ('r24h', 288)]

# 统计容器：每个尺度 × 每个档 → 计数
stats = {name: {lab: {'n': 0, 'up': 0, 'dn': 0} for lab in LAB} for name, _ in SCALES}
tot = {'n': 0, 'up': 0, 'dn': 0}
# 训练/测试分开
stats_tr = {name: {lab: {'n': 0, 'up': 0, 'dn': 0} for lab in LAB} for name, _ in SCALES}
stats_te = {name: {lab: {'n': 0, 'up': 0, 'dn': 0} for lab in LAB} for name, _ in SCALES}
tot_tr = {'n': 0, 'up': 0, 'dn': 0}
tot_te = {'n': 0, 'up': 0, 'dn': 0}

for (sym, c, h, l) in RAW:
    n = len(c)
    if n < 288 * 4 + HOLD + 2:
        continue
    fhi = fwd_max(h, HOLD)
    flo = fwd_min(l, HOLD)
    ok = np.isfinite(fhi) & np.isfinite(flo)
    up30 = ok & (fhi >= c * 1.30)
    dn30 = ok & (flo <= c * 0.70)
    use = ok.copy()

    t_cut = int(n * 0.70)
    idx = np.arange(n)
    keep = use & (idx >= 288)          # 需要 24h 历史

    for name, lag in SCALES:
        with np.errstate(invalid='ignore', divide='ignore'):
            r = c / np.roll(c, lag) - 1.0
        r[~keep] = np.nan
        b = pd.cut(pd.Series(r), bins=BINS, labels=LAB, right=False)
        # 逐档统计
        for lab in LAB:
            m = (b == lab).values
            if not m.any():
                continue
            k = int(m.sum())
            u = int((up30 & m).sum())
            d = int((dn30 & m).sum())
            stats[name][lab]['n'] += k
            stats[name][lab]['up'] += u
            stats[name][lab]['dn'] += d
            # 训练/测试
            mtr = m & (idx < t_cut)
            mte = m & (idx >= t_cut)
            if mtr.any():
                stats_tr[name][lab]['n'] += int(mtr.sum())
                stats_tr[name][lab]['up'] += int((up30 & mtr).sum())
                stats_tr[name][lab]['dn'] += int((dn30 & mtr).sum())
            if mte.any():
                stats_te[name][lab]['n'] += int(mte.sum())
                stats_te[name][lab]['up'] += int((up30 & mte).sum())
                stats_te[name][lab]['dn'] += int((dn30 & mte).sum())
    # 总基准
    k = int(keep.sum())
    tot['n'] += k; tot['up'] += int((up30 & keep).sum()); tot['dn'] += int((dn30 & keep).sum())
    ktr = int((keep & (idx < t_cut)).sum())
    tot_tr['n'] += ktr; tot_tr['up'] += int((up30 & keep & (idx < t_cut)).sum()); tot_tr['dn'] += int((dn30 & keep & (idx < t_cut)).sum())
    kte = int((keep & (idx >= t_cut)).sum())
    tot_te['n'] += kte; tot_te['up'] += int((up30 & keep & (idx >= t_cut)).sum()); tot_te['dn'] += int((dn30 & keep & (idx >= t_cut)).sum())

log(f'统计完成 {time.time()-t0:.0f}s')
log(f'全样本 n={tot["n"]:,}  P(未来48h最高>=+30%)={100*tot["up"]/max(tot["n"],1):.3f}%  P(未来48h最低<=-30%)={100*tot["dn"]/max(tot["n"],1):.3f}%')

base_up = tot['up'] / max(tot['n'], 1)
base_dn = tot['dn'] / max(tot['n'], 1)
base_up_te = tot_te['up'] / max(tot_te['n'], 1)
base_dn_te = tot_te['dn'] / max(tot_te['n'], 1)
log(f'测试段 n={tot_te["n"]:,}  P(涨)={100*base_up_te:.3f}%  P(跌)={100*base_dn_te:.3f}%')

# ---------- 3. 四格 lift ----------
log('')
log('=' * 116)
log('【一】四格条件概率与 lift（测试段）')
log('=' * 116)
log(f'{"尺度":<6} {"过去涨跌":<10} {"样本":>10} {"P(涨)":>8} {"lift涨":>7} {"P(跌)":>8} {"lift跌":>7}   判读')
log('-' * 116)
for name, _ in SCALES:
    for lab in LAB:
        s = stats_te[name][lab]
        if s['n'] < 500:
            continue
        pu = s['up'] / s['n']
        pd_ = s['dn'] / s['n']
        lu = pu / base_up_te if base_up_te > 0 else 0
        ld = pd_ / base_dn_te if base_dn_te > 0 else 0
        # 判读
        tags = []
        if lu >= 2.0 and lu > ld: tags.append('★反向(跌→涨)')
        if ld >= 2.0 and ld > lu: tags.append('★同向(跌→跌)')
        if lu >= 2.0 and lab.startswith('+') and lu > ld: tags.append('★延续(涨→涨)')
        if ld >= 2.0 and lab.startswith('+'): tags.append('★见顶(涨→跌)')
        log(f'{name:<6} {lab:<10} {s["n"]:>10,} {100*pu:>7.2f}% {lu:>6.2f}x {100*pd_:>7.2f}% {ld:>6.2f}x   {",".join(tags)}')

# ---------- 4. 一致性：训练 vs 测试 ----------
log('')
log('=' * 116)
log('【二】训练段 vs 测试段 lift 一致性（真信号两段应同向）')
log('=' * 116)
log(f'{"尺度":<6} {"过去涨跌":<10} {"训lift涨":>9} {"测lift涨":>9} {"训lift跌":>9} {"测lift跌":>9}  稳定')
log('-' * 116)
for name, _ in SCALES:
    for lab in LAB:
        a = stats_tr[name][lab]; b = stats_te[name][lab]
        if a['n'] < 500 or b['n'] < 500:
            continue
        base_up_tr = tot_tr['up'] / max(tot_tr['n'], 1)
        base_dn_tr = tot_tr['dn'] / max(tot_tr['n'], 1)
        lu_tr = (a['up'] / a['n']) / base_up_tr if base_up_tr > 0 else 0
        ld_tr = (a['dn'] / a['n']) / base_dn_tr if base_dn_tr > 0 else 0
        lu_te = (b['up'] / b['n']) / base_up_te if base_up_te > 0 else 0
        ld_te = (b['dn'] / b['n']) / base_dn_te if base_dn_te > 0 else 0
        st = ''
        if lu_te >= 1.5 and lu_tr >= 1.5: st += '涨稳 '
        if ld_te >= 1.5 and ld_tr >= 1.5: st += '跌稳 '
        if st:
            log(f'{name:<6} {lab:<10} {lu_tr:>8.2f}x {lu_te:>8.2f}x {ld_tr:>8.2f}x {ld_te:>8.2f}x  {st}')

# ---------- 5. 落盘 ----------
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bidir_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))

log('')
log(f'报告已写 {os.path.join(OUT, "bidir_summary.txt")}')
log(f'TOTAL {time.time()-t0:.0f}s')
