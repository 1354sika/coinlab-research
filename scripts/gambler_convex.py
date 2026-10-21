# -*- coding: utf-8 -*-
"""gambler_convex.py — 不预测，只参与：靠「不对称赔率」从真实波动里取钱

用户判断（2026-10-17）：
    「确实没有局，但是翻几倍的波动是真的。」

这句话把问题拆成了两半：
    ① 波动是真的        —— 已证实：周期涨幅 p90=78%、p99=769%、最大 15150%（R=10% 口径）
    ② 波动可预测吗      —— 已否证：认定前 AUC 0.524、早期识别 AUC 0.34

⇒ 剩下的路只有一条：**不预测，改赔率**。
   小亏快速截断、大赚放开跑 ⇒ 只要「尾部收益 × 尾部概率 > 小亏 × 大亏概率 + 成本」，
   不预测也能正期望。这在数学上就是**凸性（convexity）**。

【本脚本测四种"不预测"的参与方式，全部无前视、全样本、扣 0.09%】
    P1 固定持有 48h          （对照组：什么退出规则都不用）
    P2 固定止损 -15% / 持有 48h
    P3 移动止损：从峰值回落 20% 离场，最长 30 天
    P4 移动止损：从峰值回落 30% 离场，最长 30 天
    P5 移动止损：从峰值回落 40% 离场，最长 30 天

【入场】每个 5m 时点都当作潜在入场（每币抽样 3000 点），**不做任何筛选**
        —— 这是"不预测"的严格含义：连"什么时候买"都不猜。

【输出】单笔期望、中位、胜率、尾部占比、平均持有期、资金周转效率（年化近似）
"""
import os, gc, time
import numpy as np
import pandas as pd

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(D, 'klines_5m_3y.csv')
FEE = 0.0009
HOLD = int(48 * 12)              # 48h = 576 根
MAXH = int(30 * 288)             # 30 天 = 8640 根
KEEP = 3000

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_convex · 不预测，只参与：四种退出方式的期望（全样本、无筛选）')
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
log(f'读入 {len(RAW)} 币   耗时 {time.time()-t0:.0f}s')

R = {k: [] for k in ('P1', 'P2', 'P3', 'P4', 'P5')}
HOLDN = {k: [] for k in R}
PEAK = []

for (sym, ot, c, h, l) in RAW:
    n = len(c)
    if n < MAXH + 10:
        continue
    idx = np.arange(0, n - MAXH - 1, max(1, (n - MAXH) // KEEP))
    if len(idx) == 0:
        continue
    # 逐点向前的路径（用累积 max 求"首次回落 R%"的位置）
    for k in idx:
        seg_h = h[k:k + MAXH]
        seg_l = l[k:k + MAXH]
        seg_c = c[k:k + MAXH]
        c0 = c[k]
        peak_run = np.maximum.accumulate(seg_h)
        dd = seg_c / np.maximum(peak_run, 1e-12) - 1.0
        # P1 固定 48h
        R['P1'].append(seg_c[HOLD] / c0 - 1.0 - FEE); HOLDN['P1'].append(HOLD)
        # P2 固定止损 -15%，持有 48h
        w48 = seg_c[:HOLD]
        hit = seg_l[:HOLD] <= c0 * 0.85
        v2 = (-0.15 if hit.any() else w48[-1] / c0 - 1.0) - FEE
        R['P2'].append(v2); HOLDN['P2'].append(HOLD)
        # P3~P5 移动止损
        for tag, d in (('P3', 0.20), ('P4', 0.30), ('P5', 0.40)):
            trig = dd <= -d
            if trig.any():
                j = int(np.argmax(trig))
                R[tag].append(seg_c[j] / c0 - 1.0 - FEE); HOLDN[tag].append(j)
            else:
                R[tag].append(seg_c[-1] / c0 - 1.0 - FEE); HOLDN[tag].append(len(seg_c) - 1)
        PEAK.append(seg_h.max() / c0 - 1.0)
del RAW; gc.collect()

PEAK = np.array(PEAK)
log(f'样本 {len(PEAK):,}   耗时 {time.time()-t0:.0f}s')

log('\n' + '=' * 116)
log('【结果】每笔期望（扣 0.09%），入场不做任何筛选')
log('=' * 116)
log(f'  {"规则":<30}{"笔数":>9}{"期望净%":>10}{"中位%":>9}{"胜率%":>8}{"≥+50%":>8}{"≥+100%":>9}'
    f'{"≥+300%":>9}{"平均持有":>10}')
NAME = {'P1': 'P1 固定持有 48h', 'P2': 'P2 止损-15% / 48h',
        'P3': 'P3 移动止损 回落20%', 'P4': 'P4 移动止损 回落30%', 'P5': 'P5 移动止损 回落40%'}
eff = {}
for k in ('P1', 'P2', 'P3', 'P4', 'P5'):
    v = np.array(R[k]); hn = np.array(HOLDN[k])
    mh = hn.mean() * 5 / 60 / 24        # 平均持有天数
    eff[k] = (np.nanmean(v), mh)
    log(f'  {NAME[k]:<30}{len(v):>9,}{100*np.nanmean(v):>+10.2f}{100*np.nanmedian(v):>9.2f}'
        f'{100*np.nanmean(v > 0):>8.1f}{100*np.nanmean(v >= 0.50):>8.1f}'
        f'{100*np.nanmean(v >= 1.00):>9.1f}{100*np.nanmean(v >= 3.00):>9.1f}{mh:>9.1f}天')

log('\n' + '=' * 116)
log('【资金效率】期望 ÷ 平均持有天数 = 每占用一天的资金能赚多少')
log('=' * 116)
log(f'  {"规则":<30}{"期望净%":>10}{"平均持有天":>11}{"每天资金收益%":>14}{"年化(×365)":>12}')
for k in ('P1', 'P2', 'P3', 'P4', 'P5'):
    e, mh = eff[k]
    d = e / max(mh, 1e-9)
    log(f'  {NAME[k]:<30}{100*e:>+10.2f}{mh:>11.2f}{100*d:>+14.4f}{100*d*365:>+12.1f}%')

log('\n' + '=' * 116)
log('【凸性检验】入场后 30 天内"曾经达到"的峰值涨幅分布（这才是翻几倍的机会）')
log('=' * 116)
log(f'  中位 {100*np.median(PEAK):.1f}%   p75 {100*np.percentile(PEAK,75):.1f}%   '
    f'p90 {100*np.percentile(PEAK,90):.1f}%   p99 {100*np.percentile(PEAK,99):.1f}%   '
    f'最大 {100*PEAK.max():.0f}%')
for th in (0.5, 1.0, 2.0, 3.0, 5.0):
    log(f'    30 天内曾涨 ≥{th*100:.0f}%：{100*(PEAK >= th).mean():.2f}%')

log('\n' + '=' * 116)
log('【判读】')
log('=' * 116)
log('  · 若 P3~P5 期望 > P1（固定持有）⇒ 移动止损确实把"小亏"截断、让"大赚"跑，凸性成立')
log('  · 若"每天资金收益"为正且显著 ⇒ 不预测也能赚钱，靠的是赔率结构不是预测')
log('  · 若全部为负 ⇒ 连"不预测+不对称"也不行，5m 数据这条路彻底到头')

out = os.path.join(OUT, 'convex_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
