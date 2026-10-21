# -*- coding: utf-8 -*-
"""
问题：本金 100，多次交易累计赚 50（100 -> 150），现实吗？
- 入场：每 8h 决策点（5m K，96 根），按 波动率(sd of 前96根) 取 Top5
- 出场：TP=1%/2%/3% 先触及即成交（high 触价），否则持满 96 根按 close 出
- 成本：往返 0.08%（单边 0.04%，已是最乐观的 maker 假设）
- 统计：每年笔数、每笔净期望、满仓复利序列、达到 1.5x 的滚动时间、最大回撤
输出 coinlab/evidence/cum50_REAL.txt
"""
import numpy as np, pandas as pd, time, sys, os

t0 = time.time()
CSV = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = 'D:/中信证券/coinlab/evidence/cum50_REAL.txt'
STEP = 96          # 8h
TOP = 5
COST = 0.0008      # 往返 0.08%
TPS = [0.01, 0.02, 0.03]

D = pd.read_csv(CSV, usecols=['open_time','symbol','open','high','low','close','volume'])
print('rows', len(D), 'load %.1fs' % (time.time()-t0), flush=True)
D['open'] = D['open'].astype(np.float32); D['high'] = D['high'].astype(np.float32)
D['low'] = D['low'].astype(np.float32);  D['close'] = D['close'].astype(np.float32)
D['volume'] = D['volume'].astype(np.float32)
D = D.sort_values(['symbol','open_time']).reset_index(drop=True)

# 先只保留决策点及其后 96 根，省内存：按每币每 96 根切块
pieces = []
for sym, g in D.groupby('symbol', sort=False):
    o = g.open.to_numpy(); h = g.high.to_numpy(); c = g.close.to_numpy()
    t = g.open_time.to_numpy(); n = len(o)
    sd = pd.Series(r:=np.diff(np.log(c), prepend=np.log(c[0]))).rolling(STEP, min_periods=STEP).std().to_numpy()
    idx = np.arange(STEP, n-STEP, STEP)          # 决策点：至少看过 96 根、后面还有 96 根
    if len(idx) == 0: continue
    idx = idx[np.isfinite(sd[idx])]
    if len(idx) == 0: continue
    pieces.append(pd.DataFrame({
        't': t[idx], 'sym': sym, 'vol': sd[idx],
        'entry_i': idx, 'gid': np.arange(len(idx)),
    }))
P = pd.concat(pieces, ignore_index=True)
P = P.sort_values(['t','vol'], ascending=[True, False])
P['rank'] = P.groupby('t').cumcount()
P = P[P['rank'] < TOP]
print('决策笔数(候选)', len(P), '%.1fs' % (time.time()-t0), flush=True)

# 回放每一笔
res = []
for sym, g in D.groupby('symbol', sort=False):
    gg = P[P['sym'] == sym]
    if len(gg) == 0: continue
    o = g.open.to_numpy(); h = g.high.to_numpy(); c = g.close.to_numpy()
    for e in gg['entry_i'].to_numpy():
        ep = float(o[e]); hi = h[e:e+STEP]; cl = float(c[e+STEP-1])
        row = {'t': int(gg.loc[gg['entry_i'] == e, 't'].iloc[0]), 'sym': sym, 'entry': ep}
        for tp in TPS:
            target = ep * (1 + tp)
            hit = np.flatnonzero(hi >= target)
            raw = tp if len(hit) else cl / ep - 1
            row['r%g' % tp] = raw - COST
        res.append(row)
R = pd.DataFrame(res).sort_values('t').reset_index(drop=True)
print('成交笔数', len(R), '%.1fs' % (time.time()-t0), flush=True)

L = []
L.append('多次交易累计到 +50%%（100 -> 150）可达性实测  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('入场：每 8h 按波动率 Top%d 选币；出场：TP 先触及即卖，否则持满 96 根收盘卖' % TOP)
L.append('成本：往返 0.08%%（单边 0.04%%）；零滑点、零成交率折扣')
L.append('数据：60 币 / 3 年 / 5m；总笔数 %d' % len(R))
L.append('')
L.append('=== 一、每笔净期望（这就是"多次"里的单次）')
L.append('  TP        笔数     均值/笔      中位/笔    胜率    触发率  未触发均值')
for tp in TPS:
    col = 'r%g' % tp
    arr = R[col].to_numpy()
    trig = arr >= tp - COST - 1e-9
    L.append('  %-6s %6d  %+8.3f%%  %+8.3f%%  %5.1f%%  %5.1f%%  %+8.3f%%' % (
        '%.0f%%' % (tp*100), len(arr), arr.mean()*100, np.median(arr)*100,
        (arr > 0).mean()*100, trig.mean()*100, arr[~trig].mean()*100 if (~trig).any() else 0.0))
L.append('')
L.append('=== 二、满仓复利：从 100 到 150 需要多久')
for tp in TPS:
    col = 'r%g' % tp
    eq = np.cumprod(1 + R[col].to_numpy())
    days = (R.t.max() - R.t.min()) / 1000 / 86400
    yrs = days / 365
    # 达到 1.5x 的首次时间
    hit = np.flatnonzero(eq >= 1.5)
    tgt = '从未达到' if len(hit) == 0 else '第 %d 笔 (%.1f 天)' % (hit[0]+1, (R.t.iloc[hit[0]]-R.t.iloc[0])/1000/86400)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    L.append('  TP=%-4s 年笔数 %6.1f  首达1.5x %-22s 最大回撤 %6.1f%%  3年终值 %.2fx' % (
        '%.0f%%' % (tp*100), len(R)/yrs, tgt, mdd*100, eq[-1]))
L.append('')
L.append('=== 三、对照：只有每笔期望为正、且笔数足够，累计才成立')
L.append('  累计到 1.5x 所需笔数 ≈ ln(1.5)/每笔净期望')
for tp in TPS:
    mu = R['r%g' % tp].mean()
    if mu > 0:
        L.append('  TP=%-4s μ=%+.3f%% => 需约 %d 笔（按均值线性外推，未计波动拖累）' % ('%.0f%%' % (tp*100), mu*100, int(np.log(1.5)/mu+1)))
    else:
        L.append('  TP=%-4s μ=%+.3f%% => 期望为负，累计必然亏' % ('%.0f%%' % (tp*100), mu*100))
L.append('')
L.append('=== 四、"几十笔累计到 +50%%"在历史上多久出现一次（滑窗）')
for tp in TPS:
    col = 'r%g' % tp
    arr = np.clip(R[col].to_numpy(), -0.99, None)
    lg = np.log1p(arr)
    for n in (30, 40, 50):
        sv = np.lib.stride_tricks.sliding_window_view(lg, n)
        tot = np.exp(sv.sum(axis=1))
        hit = tot >= 1.5
        L.append('  TP=%-4s 连做 %2d 笔：窗口 %6d 个，其中累计 >=1.5x 的 %5d 个 (%.2f%%)，该口径下中位累计 %.2fx' % (
            '%.0f%%' % (tp*100), n, len(tot), hit.sum(), hit.mean()*100, np.median(tot)))
L.append('')
L.append('=== 五、纪律版："累计到 +50%% 就停手"，从任一时刻起要多少笔才碰到 1.5x')
for tp in TPS:
    col = 'r%g' % tp
    lg = np.log1p(np.clip(R[col].to_numpy(), -0.99, None))
    cum = np.cumsum(lg)
    n = len(cum)
    # cum 非单调：只在"从右往左的严格最大记录"上二分（任何首次达标点必在其中）
    rec = []
    for j in range(n - 1, -1, -1):
        if not rec or cum[j] > cum[rec[-1]]:
            rec.append(j)
    rec = np.array(rec[::-1])          # 索引升序，cum 值严格降序
    rv = -cum[rec]                     # 升序
    T = -(cum + np.log(1.5))           # 升序搜索
    pos = np.searchsorted(rv, T, side='left')
    ok = pos < len(rec)
    need = np.where(ok, rec[np.minimum(pos, len(rec) - 1)] - np.arange(n), -1)
    hit = need >= 0
    L.append('  TP=%-4s 能碰到的起点 %5d/%d (%.1f%%)，中位 %3d 笔，25%%分位 %3d 笔，75%%分位 %3d 笔' % (
        '%.0f%%' % (tp*100), int(hit.sum()), n, hit.mean()*100,
        int(np.median(need[hit])) if hit.any() else -1,
        int(np.percentile(need[hit], 25)) if hit.any() else -1,
        int(np.percentile(need[hit], 75)) if hit.any() else -1))
L.append('')
L.append('=== 六、40 笔窗口能到 1.5x 的时点分布（TP=2%%，看是不是集中在某几段行情）')
arr2 = np.clip(R['r0.02'].to_numpy(), -0.99, None)
lg2 = np.log1p(arr2)
sv2 = np.lib.stride_tricks.sliding_window_view(lg2, 40)
tot2 = np.exp(sv2.sum(axis=1))
ts = pd.to_datetime(R['t'].to_numpy()[39:], unit='ms')
qq = pd.PeriodIndex(ts, freq='Q').astype(str)
d2 = pd.DataFrame({'q': qq, 'hit': tot2 >= 1.5})
g2 = d2.groupby('q')['hit'].agg(['size', 'sum'])
for idx in g2.index:
    sz = int(g2.loc[idx, 'size']); sm = int(g2.loc[idx, 'sum'])
    L.append('  %s  窗口 %5d  命中 %4d  (%.1f%%)' % (idx, sz, sm, sm / sz * 100))
L.append('')
L.append('=== 七、到了 1.5x 之后：立刻停 vs 回撤 f%% 再停（同一账户从第一笔跑起，模拟"凭感觉"的反面）')
for tp in TPS:
    col = 'r%g' % tp
    arr = np.clip(R[col].to_numpy(), -0.99, None)
    eq = np.cumprod(1 + arr)
    n = len(eq)
    if not (eq >= 1.5).any():
        L.append('  TP=%-4s 3 年内从未到 1.5x（峰值 %.2fx）' % ('%.0f%%' % (tp*100), eq.max()))
        continue
    j0 = int(np.argmax(eq >= 1.5))
    pk = float(eq[j0:].max())
    L.append('  TP=%-4s 第 %d 笔首达 1.5x；此后峰值 %.2fx；3 年终值 %.4fx（无规则=任其回落）' % (
        '%.0f%%' % (tp*100), j0 + 1, pk, eq[-1]))
    for f in (0.02, 0.05, 0.10):
        peak = eq[j0]; val = eq[-1]; stop = n - 1
        for k in range(j0, n):
            if eq[k] > peak: peak = eq[k]
            if eq[k] <= peak * (1 - f):
                val = eq[k]; stop = k; break
        L.append('     回撤 %.0f%% 就停：第 %6d 笔锁定 %.3fx（峰值 %.2fx，从峰值吐回 %.1f%%）' % (
            f * 100, stop + 1, val, peak, (1 - val / peak) * 100))
L.append('')
L.append('=== 八、L3 守护层跨季度稳健性（每季独立起盘：净值 1.0，首达 1.5x 后从峰值回撤 5%% 清仓，TP=2%%）')
arr8 = np.clip(R['r0.02'].to_numpy(), -0.99, None)
qq8 = pd.PeriodIndex(pd.to_datetime(R['t'].to_numpy(), unit='ms'), freq='Q').astype(str)
L.append('  季度     笔数   到过1.5x   守护锁定    当季终值(无规则)')
for qv in sorted(set(qq8)):
    m = qq8 == qv
    if int(m.sum()) < 100:
        continue
    a = arr8[m]
    eq = np.cumprod(1 + a)
    if not (eq >= 1.5).any():
        L.append('  %s  %5d   否        -          %.3fx' % (qv, len(a), eq[-1]))
        continue
    j0 = int(np.argmax(eq >= 1.5))
    peak = eq[j0]; val = eq[-1]; stop = len(eq) - 1
    for k in range(j0, len(eq)):
        peak = max(peak, eq[k])
        if eq[k] <= peak * 0.95:
            val = eq[k]; stop = k; break
    L.append('  %s  %5d   是       %.3fx      %.3fx' % (qv, len(a), val, eq[-1]))
L.append('')
L.append('=== 九、按"日"的口径：一天内所有笔累计 >= +50%% 的天数（达到即停、次日以新本金复利）')
R['day'] = pd.to_datetime(R['t'], unit='ms').dt.strftime('%Y-%m-%d')
for tp in (2,):
    col = 'r0.02'
    arr = np.clip(R[col].to_numpy(), -0.99, None)
    df = pd.DataFrame({'day': R['day'], 'r': arr})
    daily = df.groupby('day')['r'].apply(lambda x: float(np.prod(1 + x.to_numpy()) - 1))
    nd = len(daily)
    hit = daily >= 0.5
    L.append('  TP=2%%  总天数 %d  日内累计 >=+50%% 的 %d 天 (%.2f%%)  折合每年约 %.1f 天' % (
        nd, int(hit.sum()), hit.mean() * 100, hit.sum() / nd * 365))
    L.append('        日收益分位：P10 %+.2f%%  P25 %+.2f%%  中位 %+.2f%%  P75 %+.2f%%  P90 %+.2f%%  最好 %+.1f%%' % (
        np.percentile(daily, 10) * 100, np.percentile(daily, 25) * 100, np.median(daily) * 100,
        np.percentile(daily, 75) * 100, np.percentile(daily, 90) * 100, daily.max() * 100))
    daily_cap = np.minimum(daily.to_numpy(), 0.5)
    eq = np.cumprod(1 + daily_cap)
    L.append('        若"到 +50%% 当日即停、次日新本金复利"：%d 天终值 %.4fx（最差日 %.1f%%）' % (
        nd, eq[-1], daily_cap.min() * 100))
L.append('')
L.append('=== 十、日内双向闸门：日内累计到 +50%% 停、跌到 -X%% 也停，次日以新本金复利（TP=2%%）')
df10 = pd.DataFrame({'day': R['day'], 'r': np.clip(R['r0.02'].to_numpy(), -0.99, None)})
for lo in (0.05, 0.10, 0.20):
    logeq = 0.0; n_hit = 0; n_lo = 0; worst = 0.0
    for d, g in df10.groupby('day'):
        e = 1.0
        for rr in g['r'].to_numpy():
            e *= (1 + rr)
            if e >= 1.5:
                n_hit += 1; break
            if e <= 1 - lo:
                n_lo += 1; break
        worst = min(worst, e - 1)
        logeq += np.log(max(e, 1e-12))
    L.append('  日内上限 +50%% / 下限 -%-3.0f%%：达标 %3d 天  触底熔断 %3d 天  1096 天终值 %.4fx  最差日 %.1f%%' % (
        lo * 100, n_hit, n_lo, np.exp(min(logeq, 700)), worst * 100))
L.append('')
L.append('诚实边界：')
L.append('  1. 全部是回测；TP 用 high 触价假设能以限价成交，实盘挂单可能不成交（成交率未计）。')
L.append('  2. 零滑点；成本只按单边 0.04%% 往返 0.08%%；若为 taker 0.10%% 单边，TP=1%% 档会转负。')
L.append('  3. 满仓复利是理论口径；真实资金曲线受同时持仓数、可用保证金限制，未模拟。')
L.append('  4. 决策点每 8h 取一个、样本按币去重（不同币不重叠），同币相邻决策不重叠。')
L.append('  5. 未做样本外/分段验证之外的稳健性检查；304 币池与 60 币池结论可能不同。')
L.append('')
L.append('TOTAL %.0fs' % (time.time()-t0))
txt = '\n'.join(L)
open(OUT, 'w', encoding='utf-8').write(txt)
print(txt)
