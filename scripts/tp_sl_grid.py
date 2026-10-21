# -*- coding: utf-8 -*-
"""tp_sl_grid.py — 止损 × 止盈 全组合网格（0.5% 步长，上限 20%）

用户口径：止损、止盈都按 0.5% 一档，从 0.5% 到 20%，两个方向全组合 = 40×40=1600 组。
跑全量历史：入场条件不变（2h 跌 2% 且量比>=3），出场用纯 TP/SL（去掉 θ 跟踪），
最长持 96 根 5m（8h），成本 0.04% 单边（往返 0.08%）。

要点：这不是模型，是穷举。跑完能回答「这两条线的任何组合，有没有正期望」。

用法：
  python -X utf8 tp_sl_grid.py                 # 全部数据集
  python -X utf8 tp_sl_grid.py klines_5m_3y.csv

产物：
  evidence/tp_sl_grid.txt      每数据集的最优组合表
  evidence/tp_sl_heat_<名>.png  40×40 均值热力图
"""
import os
import sys
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
EVI = r'D:/中信证券/coinlab/evidence'
OUT = os.path.join(EVI, 'tp_sl_grid.txt')
H, COST2, W = 96, 0.0008, 48
DIP, VOLR = -0.02, 3.0
GRID = np.round(np.arange(0.005, 0.2001, 0.005), 4)   # 40 档

FILES = [
    ('klines_5m_3y.csv', 300),
    ('klines_5m_502.csv', 300),
    ('klines_1m_1y.csv', 60),
    ('klines_1m_502.csv', 60),
    ('klines_1m_sync.csv', 60),
    ('klines_1m_wide.csv', 60),
    ('klines_1m_long.csv', 60),
    ('klines_1m_top20_7d.csv', 60),
    ('klines_1h_top20_365d.csv', 3600),
    ('klines_1d_full.csv', 86400),
    ('klines_1d_top502.csv', 86400),
]

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def collect(fname, tf):
    """收集所有信号 + 未来 96 根的相对价格。1m 先聚合成 5m。"""
    p = os.path.join(D, fname)
    df = pd.read_csv(p, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
    need_agg = tf < 300
    hs, ls, cs = [], [], []
    nsig = 0
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        if need_agg:
            g = g.assign(bucket=(g.open_time // 300000))
            g = (g.groupby('bucket')
                   .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                        close=('close', 'last'), volume=('volume', 'sum'))
                   .reset_index())
        h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
        c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
        n = len(c)
        if n < 600 + H:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
        idxs = np.where((ret24 <= DIP) & (volr >= VOLR))[0]
        idxs = idxs[(idxs >= W + 300) & (idxs < n - H - 1)]
        nsig += len(idxs)
        for i in idxs:
            e = float(c[i])
            if e <= 0:
                continue
            seg = slice(i + 1, i + 1 + H)
            hs.append(h[seg] / e - 1.0)
            ls.append(l[seg] / e - 1.0)
            cs.append(c[seg] / e - 1.0)
    del df
    if not hs:
        return None, 0
    return (np.stack(hs).astype(np.float32),
            np.stack(ls).astype(np.float32),
            np.stack(cs).astype(np.float32)), nsig


def first_hit(rel, level, above=True):
    m = (rel >= level) if above else (rel <= level)
    idx = m.argmax(axis=1).astype(np.int16)
    none = ~m.any(axis=1)
    idx[none] = 127
    return idx


def stats(r):
    n = len(r)
    if n == 0:
        return None
    eq = 1.0; peak = 1.0; mdd = 0.0; to15 = None
    for i, x in enumerate(r, 1):
        eq *= (1 + x)
        if eq > peak:
            peak = eq
        dd = eq / peak - 1
        if dd < mdd:
            mdd = dd
        if to15 is None and eq >= 1.5:
            to15 = i
    return dict(n=n, mean=float(r.mean()), med=float(np.median(r)), win=float((r > 0).mean()),
                eq=eq, mdd=mdd, to15=to15)


def run(fname, tf):
    t0 = time.time()
    arr, nsig = collect(fname, tf)
    if arr is None:
        log('%-26s  无信号' % fname)
        return None
    hh, ll, cc = arr
    N = len(hh)
    j_tp = np.stack([first_hit(hh, tp, True) for tp in GRID])       # 40×N
    j_sl = np.stack([first_hit(ll, -sl, False) for sl in GRID])     # 40×N
    last = cc[:, -1]

    mat_mean = np.zeros((len(GRID), len(GRID)), dtype=np.float64)   # [tp, sl]
    best = []
    n_pos = 0
    # 顺序复利用：收集所有组合的逐笔 -> 太占内存，只对前若干名算复利
    cand = []
    for a, tp in enumerate(GRID):
        jt = j_tp[a]
        for b, sl in enumerate(GRID):
            js = j_sl[b]
            none = (jt == 127) & (js == 127)
            hit_tp = jt < js
            r = np.where(none, last - COST2,
                         np.where(hit_tp, tp - COST2, -sl - COST2)).astype(np.float64)
            m = r.mean()
            mat_mean[a, b] = m
            if m > 0:
                n_pos += 1
            cand.append((m, a, b, r))

    cand.sort(key=lambda z: -z[0])
    top = cand[:12]
    log('')
    log('=' * 108)
    log('%s  周期 %ds  信号 %d  耗时 %.0fs' % (fname, tf, N, time.time() - t0))
    log('=' * 108)
    log('1600 组里，均值为正的组合：%d 组（%.1f%%）' % (n_pos, n_pos / len(cand) * 100))
    log('')
    log('%-9s %-9s %8s %11s %10s %8s %12s %9s %8s' %
        ('止盈TP', '止损SL', '笔数', '均值/笔', '中位', '胜率', '复利终值', '最大回撤', '到1.5x'))
    log('-' * 100)
    for (m, a, b, r) in top:
        s = stats(r)
        eqtxt = ('%.3fx' % s['eq']) if abs(s['eq']) < 1e6 else ('%.2e' % s['eq'])
        log('%-9s %-9s %8d %+10.4f%% %+9.4f%% %7.1f%% %12s %8.2f%% %8s' %
            ('%.1f%%' % (GRID[a] * 100), '%.1f%%' % (GRID[b] * 100), s['n'],
             m * 100, s['med'] * 100, s['win'] * 100, eqtxt, s['mdd'] * 100,
             ('%d笔' % s['to15']) if s['to15'] else '未达'))
    log('')
    log('最差组合：%+.4f%%（TP=%.1f%% SL=%.1f%%）' %
        (min(z[0] for z in cand) * 100,
         GRID[min(cand, key=lambda z: z[0])[1]] * 100,
         GRID[min(cand, key=lambda z: z[0])[2]] * 100))

    # 热力图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7.6))
        im = ax.imshow(mat_mean * 100, origin='lower', aspect='auto', cmap='viridis',
                       extent=[GRID[0] * 100, GRID[-1] * 100, GRID[0] * 100, GRID[-1] * 100],
                       vmin=np.percentile(mat_mean * 100, 2), vmax=np.percentile(mat_mean * 100, 98))
        cs_ = ax.contour(mat_mean * 100, levels=[0], origin='lower',
                         extent=[GRID[0] * 100, GRID[-1] * 100, GRID[0] * 100, GRID[-1] * 100])
        ax.clabel(cs_, fmt='0%%', fontsize=8)
        ax.set_xlabel('stop-loss SL (%)')
        ax.set_ylabel('take-profit TP (%)')
        ax.set_title('mean net return per trade (%%) - %s  N=%d' % (fname, N))
        fig.colorbar(im, ax=ax, label='mean %/trade')
        name = fname.replace('.csv', '')
        png = os.path.join(EVI, 'tp_sl_heat_%s.png' % name)
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        log('热力图已写 %s' % png)
    except Exception as e:
        log('热力图失败: %s' % e)
    return dict(file=fname, npos=n_pos, best=top[0][0], best_tp=GRID[top[0][1]], best_sl=GRID[top[0][2]])


def main():
    log('=' * 108)
    log('止损 × 止盈 全组合网格 0.5%%~20%%（1600 组）× 全量历史   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('入场：2h 跌 2%% 且量比>=3；出场：纯 TP/SL；最长 8h；成本 0.04%% 单边')
    log('=' * 108)
    targets = sys.argv[1:]
    files = [f for f in FILES if not targets or f[0] in targets]
    summ = []
    for fname, tf in files:
        if not os.path.exists(os.path.join(D, fname)):
            log('%-26s 文件不存在' % fname); continue
        r = run(fname, tf)
        if r:
            summ.append(r)
    log('')
    log('=' * 108)
    log('汇总：每组「最好的 TP/SL」')
    log('=' * 108)
    log('%-26s %10s %12s %10s %10s' % ('数据集', '正期望组数', '最好均值/笔', 'TP', 'SL'))
    log('-' * 80)
    for r in summ:
        log('%-26s %10d %+11.4f%% %9.1f%% %9.1f%%' %
            (r['file'], r['npos'], r['best'] * 100, r['best_tp'] * 100, r['best_sl'] * 100))
    log('')
    log('诚实边界：零滑点；TP 按 high 触价、SL 按 (1-SL) 触价，均假设能成交（实盘需排队）；')
    log('         按币独立结算；含样本内；复利终值只对上表前 12 名逐笔计算。')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('已写 %s' % OUT)


if __name__ == '__main__':
    main()
