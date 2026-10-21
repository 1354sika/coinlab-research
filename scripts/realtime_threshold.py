# -*- coding: utf-8 -*-
"""
realtime_threshold.py — 用数学直接算「涨跌各自的实时临界值」

用户要的：不是全历史搜一个固定 TP/SL，而是每个入场时刻用当时的 σ
算出「这一刻该把止盈挂在 +a、止损挂在 -b，a 和 b 各是多少」。

做法：
  1. 每个入场点用「过去 240 根 1m」算 σ（实时，无前视）
  2. 按 σ 分 20 个对数等距桶
  3. 每桶内扫描 (a/σ, b/σ) 网格，统计「先触 +a 还是先触 -b」
  4. 求每桶的最优 (a/σ, b/σ) => 看这两个比例是否随 σ 稳定
  5. 若稳定 => 得到公式 a* = k_tp·σ, b* = k_sl·σ（这就是「实时临界值」）

数学背景：
  无漂移 GBM 下先到 +a 的概率 = b/(a+b)，期望恒为 0（公平游戏）。
  所以正期望只能来自「急跌后反弹的偏差」 —— 本脚本用经验频率直接测这个偏差，
  不假设任何模型。
"""
import os, sys, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EVI  = r'D:\中信证券\coinlab\evidence'
OUT  = os.path.join(EVI, 'realtime_threshold.txt')

COST  = 0.0015
H     = 480
STEP  = 240
DROP  = 0.02
VOLR  = 3.0
SIG_N = 240

K_TP = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0], np.float64)
K_SL = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0], np.float64)
# σ 桶：对数等距 0.03% ~ 3%
SIG_EDGES = np.geomspace(0.0003, 0.03, 21)
NBIN = len(SIG_EDGES) - 1

L = []
def p(s=''):
    print(s)
    L.append(s)

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'klines_1m_wide.csv'
    p('=' * 112)
    p('实时临界值   file=%s   COST=%.2f%%   σ=过去%d根1m标准差   桶=%d' % (fname, COST * 100, SIG_N, NBIN))
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

    na, nb = len(K_TP), len(K_SL)
    cnt = np.zeros((NBIN, na, nb), np.float64)
    ntp = np.zeros((NBIN, na, nb), np.float64)
    ssum = np.zeros((NBIN, na, nb), np.float64)
    sig_bin_sum = np.zeros(NBIN, np.float64)
    sig_bin_cnt = np.zeros(NBIN, np.float64)
    n_entry = 0

    for si, sym in enumerate(syms):
        g = d[d.symbol == sym].sort_values('open_time')
        c = g.close.to_numpy(np.float64)
        h = g.high.to_numpy(np.float64)
        lo = g.low.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
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
            n_entry += 1
            b = int(np.searchsorted(SIG_EDGES, sg, side='right') - 1)
            if b < 0 or b >= NBIN:
                continue
            sig_bin_sum[b] += sg
            sig_bin_cnt[b] += 1
            seg_h = h[i:i + H]
            seg_l = lo[i:i + H]
            seg_c = c[i:i + H]
            for ai, kt in enumerate(K_TP):
                up = entry * (1.0 + kt * sg)
                ht = seg_h >= up
                it = int(np.argmax(ht)) if ht.any() else 10**9
                for bi, ks in enumerate(K_SL):
                    dn = entry * (1.0 - ks * sg)
                    hl = seg_l <= dn
                    il = int(np.argmax(hl)) if hl.any() else 10**9
                    if it < 10**9 and it <= il:
                        r = kt * sg; win = 1.0
                    elif il < 10**9:
                        r = -ks * sg; win = 0.0
                    else:
                        r = seg_c[-1] / entry - 1.0; win = 1.0 if r > 0 else 0.0
                    cnt[b, ai, bi] += 1
                    ntp[b, ai, bi] += win
                    ssum[b, ai, bi] += r
        if (si + 1) % 25 == 0:
            p('  [%d/%d] entry=%d  %.0fs' % (si + 1, len(syms), n_entry, time.time() - t0))

    p('done  entry=%d  %.0fs' % (n_entry, time.time() - t0))
    p()

    # ---- 合并成 5 档（按样本量分位） ----
    sig_med = np.where(sig_bin_cnt > 0, sig_bin_sum / np.maximum(sig_bin_cnt, 1), 0.0)
    order = np.argsort(np.where(sig_bin_cnt > 0, sig_med, 1e9))
    valid = [b for b in order if sig_bin_cnt[b] >= 50]
    if len(valid) < 2:
        p('有效桶不足，退出'); return
    # 分成最多 5 组
    ng = min(5, len(valid))
    groups = np.array_split(np.array(valid), ng)

    p('=== 一、按 σ 分档（每档 σ 中位 / 最优止盈 a* / 最优止损 b*）')
    p('  %-6s %10s %8s %12s %12s %10s %10s' %
      ('档', 'σ中位', '入场数', '最优a/σ', '最优b/σ', '均值/笔', '胜率'))
    p('  ' + '-' * 76)
    best_rows = []
    for gi, gb in enumerate(groups):
        gb = [b for b in gb if sig_bin_cnt[b] > 0]
        if not gb:
            continue
        cc = cnt[gb].sum(0); tt = ntp[gb].sum(0); ss = ssum[gb].sum(0)
        n_ent = sig_bin_cnt[gb].sum()
        sg_med = (sig_bin_sum[gb].sum()) / max(sig_bin_cnt[gb].sum(), 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            exp = np.where(cc > 0, ss / cc, -9)
        # 加上成本
        exp_net = exp - COST
        ai, bi = np.unravel_index(np.argmax(exp_net), exp_net.shape)
        e_net = exp_net[ai, bi]
        wr = ntp[gb, ai, bi].sum() / max(cc[ai, bi], 1)
        p('  %-6d %9.4f%% %7d %11.2f %11.2f %9.4f%% %9.1f%%'
          % (gi + 1, sg_med * 100, n_ent, K_TP[ai], K_SL[bi], e_net * 100, wr * 100))
        best_rows.append((gi + 1, sg_med, n_ent, K_TP[ai], K_SL[bi], e_net, wr))

    p()
    p('=== 二、核心判定：最优 a/σ 和 b/σ 是否随 σ 稳定')
    if len(best_rows) >= 2:
        ktps = np.array([r[3] for r in best_rows])
        ksls = np.array([r[4] for r in best_rows])
        sigs = np.array([r[1] for r in best_rows])
        p('  最优 a/σ 序列：' + ' '.join('%.2f' % x for x in ktps))
        p('  最优 b/σ 序列：' + ' '.join('%.2f' % x for x in ksls))
        p('  σ 序列      ：' + ' '.join('%.4f%%' % (x * 100) for x in sigs))
        # 变异系数
        cv_tp = ktps.std() / ktps.mean() if ktps.mean() > 0 else 9
        cv_sl = ksls.std() / ksls.mean() if ksls.mean() > 0 else 9
        p()
        p('  变异系数 a/σ = %.2f   b/σ = %.2f  （<0.3 算稳定，>0.5 说明"实时临界值"不成立）'
          % (cv_tp, cv_sl))
        if cv_tp < 0.3 and cv_sl < 0.3:
            p('  => 稳定 => 可以用公式：止盈 = %.2f·σ，止损 = %.2f·σ' % (ktps.mean(), ksls.mean()))
        else:
            p('  => 不稳定 => "最优临界值随 σ 变"这件事不成立（至少不是简单的线性比例）')

    p()
    p('=== 三、对照：全样本最优固定值（不管你 σ 是多少，都用同一个数）')
    cc = cnt.sum(0); tt = ntp.sum(0); ss = ssum.sum(0)
    exp = np.where(cc > 0, ss / cc, -9) - COST
    ai, bi = np.unravel_index(np.argmax(exp), exp.shape)
    p('  固定最优：a=%.2f·σ  b=%.2f·σ  （若 σ 取中位 %.4f%%，等于固定 a=%.2f%% b=%.2f%%）'
      % (K_TP[ai], K_SL[bi], sig_med[valid].mean() * 100,
         K_TP[ai] * sig_med[valid].mean() * 100, K_SL[bi] * sig_med[valid].mean() * 100))
    p('  均值/笔 %.4f%%  胜率 %.1f%%' % (exp[ai, bi] * 100, ntp.cumsum(0)[-1, ai, bi] / max(cc[ai, bi], 1) * 100))

    p()
    p('=== 四、期望矩阵（行=a/σ，列=b/σ，值=均值/笔 %%，已扣成本）')
    hdr = '  %-8s' % '' + ''.join('%8.1f' % b for b in K_SL)
    p(hdr)
    for ai in range(na):
        row = '  a=%-6.1f' % K_TP[ai]
        for bi in range(nb):
            v = exp[ai, bi] * 100 if cc[ai, bi] > 0 else -99
            row += '%8.3f' % v
        p(row)

    p()
    p('--- 诚实边界 ---')
    p('1. 收盘价 + high/low 触价口径；止盈假设能以该价位成交（实盘需挂单，成交率未计）。')
    p('2. σ 用过去 240 根 1m 收益标准差，无前视；但"每档最优 a,b"是全样本算的，样本内。')
    p('3. 同一根内 TP/SL 同触时按 TP 优先（乐观），真实顺序未知。')
    p('4. 入场点每 240 根取一个，样本重叠已降低但同波行情仍可能重复。')
    p('5. 已下架归零币不在池内（幸存者偏差）。')
    p('6. 成本 0.15% 往返线性扣除。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    p()
    p('written %s' % OUT)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('TOTAL %.0fs' % (time.time() - t0))
