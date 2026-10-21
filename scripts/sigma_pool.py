# -*- coding: utf-8 -*-
"""
sigma_pool.py — 验证「3σ 止盈止损 + 高波动筛选」在真实资金约束下能不能赚

上一轮发现：
  · 止盈 = 3σ、止损 = 3σ 是最优比例（σ = 过去 240 根 1m 波动率）
  · σ 越大越赚：σ 档 5 是档 1 的 6 倍

本轮要验证的是：这两条加起来，放进「单账号、N 仓位并发、满仓跳过」的真实约束里，
还能不能赚。这是前面所有策略失败的地方。

三个变量：
  1. σ 门槛（只做抖动大的币）：0 / 0.15% / 0.20% / 0.25% / 0.30% / 0.40%
  2. 池 N（并发仓位数）：1 / 2 / 5 / 10 / 20 / 50
  3. 时间切分：前 70% / 后 30%
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'sigma_pool.txt')

COST   = 0.0025       # 0.25% 往返
H      = 480          # 持仓上限 480 根 1m = 8h
STEP   = 60           # 入场点每 60 根取一个（1h），保留信号密度
DROP   = 0.02
VOLR   = 3.0
SIG_N  = 240
K      = 3.0          # 止盈止损 = 3σ

SIG_MINS = [0.0, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040]
POOLS    = [1, 2, 5, 10, 20, 50]

L = []
def p(s=''):
    print(s)
    L.append(s)

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    p('=' * 112)
    p('3σ 池模拟  file=%s  TP=SL=%.1fσ  COST=%.2f%%  H=%d根' % (fname, K, COST * 100, H))
    p('=' * 112)

    fp = os.path.join(DATA, fname)
    t0 = time.time()
    d = pd.read_csv(fp, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': 'str', 'open_time': 'int64',
                           'high': 'float32', 'low': 'float32',
                           'close': 'float32', 'volume': 'float32'})
    p('rows %d  %.1fs' % (len(d), time.time() - t0))
    syms = sorted(d.symbol.unique())
    p('币 %d' % len(syms))

    # 收集所有入场点的出场结果（不含池约束）
    recs = []   # (t_exit_ms, symbol, r_net, hold_bars, sigma, t_entry_ms)
    for si, sym in enumerate(syms):
        g = d[d.symbol == sym].sort_values('open_time')
        c = g.close.to_numpy(np.float64)
        h = g.high.to_numpy(np.float64)
        lo = g.low.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
        tt = g.open_time.to_numpy(np.int64)
        n = len(c)
        if n < SIG_N + H + 300:
            continue
        lr = np.zeros(n, np.float64)
        lr[1:] = np.log(c[1:] / c[:-1])
        sig = pd.Series(lr).rolling(SIG_N, min_periods=SIG_N // 2).std().to_numpy()
        r24 = np.zeros(n, np.float64)
        r24[24:] = c[24:] / c[:-24] - 1.0
        vma = pd.Series(v).rolling(96, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)

        cand = np.flatnonzero((r24 <= -DROP) & (volr >= VOLR) & (sig > 0))
        if len(cand) == 0:
            continue
        keep, last = [], -10**9
        for i in cand:
            if i - last >= STEP and i + H < n:
                keep.append(i); last = i

        for i in keep:
            sg = sig[i]
            entry = c[i]
            if not (np.isfinite(sg) and sg > 0 and entry > 0):
                continue
            up = entry * (1.0 + K * sg)
            dn = entry * (1.0 - K * sg)
            seg_h = h[i + 1:i + 1 + H]
            seg_l = lo[i + 1:i + 1 + H]
            seg_c = c[i + 1:i + 1 + H]
            m = len(seg_c)
            if m < 5:
                continue
            ht = seg_h >= up
            hl = seg_l <= dn
            it = int(np.argmax(ht)) if ht.any() else 10**9
            il = int(np.argmax(hl)) if hl.any() else 10**9
            if it < 10**9 and it <= il:
                r = K * sg; hold = it + 1; j = it
            elif il < 10**9:
                r = -K * sg; hold = il + 1; j = il
            else:
                r = seg_c[-1] / entry - 1.0; hold = m; j = m - 1
            recs.append((int(tt[i + 1 + j]), sym, float(r - COST), int(hold), float(sg), int(tt[i])))
        if (si + 1) % 25 == 0:
            p('  [%d/%d] recs=%d  %.0fs' % (si + 1, len(syms), len(recs), time.time() - t0))

    p('done  recs=%d  %.0fs' % (len(recs), time.time() - t0))
    R = pd.DataFrame(recs, columns=['t_exit', 'sym', 'r', 'hold', 'sigma', 't_entry'])
    R = R.sort_values('t_entry').reset_index(drop=True)
    p('信号跨度 %s ~ %s' % (pd.to_datetime(R.t_entry.min(), unit='ms'),
                           pd.to_datetime(R.t_entry.max(), unit='ms')))
    p()

    # ---------- 一、单笔（无池约束，已扣成本）----------
    p('=== 一、单笔（无池约束，已扣 %.2f%%）' % (COST * 100))
    p('  %-10s %8s %10s %10s %10s' % ('σ门槛', '笔数', '均值/笔', '中位/笔', '胜率'))
    p('  ' + '-' * 52)
    for sm in SIG_MINS:
        sub = R[R.sigma >= sm] if sm > 0 else R
        if len(sub) == 0:
            p('  %-10.2f%% %8d %10s' % (sm * 100, 0, '-'))
            continue
        p('  %-10.2f%% %8d %9.4f%% %9.4f%% %9.2f%%'
          % (sm * 100, len(sub), sub.r.mean() * 100, sub.r.median() * 100, (sub.r > 0).mean() * 100))

    # ---------- 二、池模拟（真实资金约束）----------
    def pool_sim(df, N, tmax=None):
        """单账号 N 仓并发：按入场时间先后，满仓跳过。返回 (期末倍数, 最大回撤, 成交笔数)"""
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        slots = []   # 每个槽的 (t_free, r)
        taken = 0
        # 按出场时间排序结算（简化：用 t_exit 结算）
        ev = df.sort_values('t_exit')
        pending = []
        for _, row in ev.iterrows():
            t_in = int(row.t_entry)
            # 释放已经到期的槽
            free = [s for s in pending if s[0] <= t_in]
            pending = [s for s in pending if s[0] > t_in]
            for s in free:
                eq *= (1.0 + s[1] / N)
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1.0)
            if len(pending) < N:
                pending.append((int(row.t_exit), float(row.r)))
                taken += 1
        for s in pending:
            eq *= (1.0 + s[1] / N)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
        return eq, mdd, taken

    p()
    p('=== 二、池模拟（单账号 N 仓并发，满仓跳过）')
    p('  读法：期末倍数 >1 才是赚；<1 是亏。回撤只计已平仓。')
    for sm in SIG_MINS:
        sub = R[R.sigma >= sm] if sm > 0 else R
        if len(sub) < 50:
            continue
        p()
        p('  --- σ >= %.2f%%  （信号 %d 笔）' % (sm * 100, len(sub)))
        p('    %-6s %8s %12s %10s' % ('池N', '成交笔', '期末倍数', '最大回撤'))
        p('    ' + '-' * 42)
        for N in POOLS:
            eq, mdd, taken = pool_sim(sub, N)
            p('    %-6d %8d %11.4fx %9.2f%%' % (N, taken, eq, mdd * 100))

    # ---------- 三、时间切分 ----------
    p()
    p('=== 三、时间切分（前 70% 训练 / 后 30% 验证）')
    tmid = int(np.median(R.t_entry))
    pre = R[R.t_entry <= tmid]
    post = R[R.t_entry > tmid]
    p('  前段 %d 笔（%s ~ %s）' % (len(pre), pd.to_datetime(pre.t_entry.min(), unit='ms'),
                                  pd.to_datetime(pre.t_entry.max(), unit='ms')))
    p('  后段 %d 笔（%s ~ %s）' % (len(post), pd.to_datetime(post.t_entry.min(), unit='ms'),
                                  pd.to_datetime(post.t_entry.max(), unit='ms')))
    p()
    p('  %-10s %-6s %10s %10s %10s %10s' % ('σ门槛', '池N', '前段均值', '后段均值', '前段期末', '后段期末'))
    p('  ' + '-' * 66)
    for sm in [0.0, 0.0020, 0.0030]:
        for N in [1, 5, 20]:
            a = pre[pre.sigma >= sm] if sm > 0 else pre
            b = post[post.sigma >= sm] if sm > 0 else post
            if len(a) < 30 or len(b) < 30:
                continue
            ea, _, _ = pool_sim(a, N)
            eb, _, _ = pool_sim(b, N)
            p('  %-10.2f%% %-6d %9.4f%% %9.4f%% %9.4fx %9.4fx'
              % (sm * 100, N, a.r.mean() * 100, b.r.mean() * 100, ea, eb))

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径；止盈假设能以该价位成交（实盘需挂单，成交率未计）。')
    p('2. σ 用过去 240 根 1m 收益标准差，无前视；K=3.0 是全样本最优（含样本内）。')
    p('3. 同一根内 TP/SL 同触时按 TP 优先（乐观）。')
    p('4. 入场点每 60 根取一个（1h），比之前 240 根的样本重叠更严重。')
    p('5. 池模拟用 t_exit 结算，未模拟逐仓保证金、资金费率、最小下单量。')
    p('6. 已下架归零币不在池内（幸存者偏差）。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
