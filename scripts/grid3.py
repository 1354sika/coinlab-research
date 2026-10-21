# -*- coding: utf-8 -*-
"""grid3.py — 三线全组合：止盈线 × 止损线 × 日内停止阈值

用户口径（逐字）：
  · 止损线：0.5% ~ 50%，变化单位 0.5%          → 100 档
  · 止盈线：0.5% ~ 50%，变化单位 0.5%          → 100 档
  · 停止机制：1.00 ~ 1.50（当天本金到 Nx 就停），变化单位 0.01 → 51 档
  全组合 = 100 × 100 × 51 = 510,000 组，每组跑全量历史。

规则：
  入场：5m 收盘 ret24 <= -2% 且 量比(当根/前24根均量) >= 3
  出场：high 触 TP 卖；low 破 entry*(1-SL) 卖；最长 24h（H 根，按周期换算）
  日内停止：按 UTC 日，当天累计净值 >= N 立即停手，次日以新本金继续（复利）
  成本：0.04% 单边（往返 0.08%）

用法：python -X utf8 grid3.py klines_5m_3y.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

D = r'D:/中信证券/coinlab/data'
EVI = r'D:/中信证券/coinlab/evidence'
H, COST2, W = 96, 0.0008, 48


def hold_bars(tf):
    """"最长持仓 24 小时"换算成该周期的根数（ZeroPage 24h 不停机）"""
    return max(1, int(round(86400.0 / tf)))
DIP, VOLR = -0.02, 3.0
TPS = np.round(np.arange(0.005, float(os.environ.get('TPMAX', '0.5001')) + 1e-9, 0.005), 4)
SLS = np.round(np.arange(float(os.environ.get('SLMIN', '0.005')),
                        float(os.environ.get('SLMAX', '0.5001')) + 1e-9, 0.005), 4)
TGT = np.round(np.arange(1.00, float(os.environ.get('TGTMAX', '1.5001')) + 1e-9, 0.01), 4)

L = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); L.append(s)


def collect(fname, tf):
    H = hold_bars(tf)
    df = pd.read_csv(os.path.join(D, fname),
                     usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
    need_agg = tf < 300
    hs, ls, cs, ts = [], [], [], []
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_values('open_time')
        if need_agg:
            g = g.assign(bucket=(g.open_time // 300000))
            g = (g.groupby('bucket')
                   .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                        close=('close', 'last'), volume=('volume', 'sum'))
                   .reset_index())
            g['open_time'] = g['bucket'] * 300000
        h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
        c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
        t = g.open_time.to_numpy(); n = len(c)
        if n < 600 + H:
            continue
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
        idxs = np.where((ret24 <= DIP) & (volr >= VOLR))[0]
        idxs = idxs[(idxs >= W + 300) & (idxs < n - H - 1)]
        for i in idxs:
            e = float(c[i])
            if e <= 0:
                continue
            seg = slice(i + 1, i + 1 + H)
            hs.append(h[seg] / e - 1.0)
            ls.append(l[seg] / e - 1.0)
            cs.append(c[seg] / e - 1.0)
            ts.append(int(t[i]))
    del df
    if not hs:
        return None
    hh = np.stack(hs).astype(np.float32)
    ll = np.stack(ls).astype(np.float32)
    cc = np.stack(cs).astype(np.float32)
    tt = np.array(ts, dtype='int64')
    o = np.argsort(tt)
    return hh[o], ll[o], cc[o], tt[o]


def first_hit(rel, level, above=True):
    m = (rel >= level) if above else (rel <= level)
    idx = m.argmax(axis=1).astype(np.int16)
    idx[~m.any(axis=1)] = 127
    return idx


def run(fname, tf):
    H = hold_bars(tf)
    t0 = time.time()
    arr = collect(fname, tf)
    if arr is None:
        log('%-26s 无信号' % fname); return None
    hh, ll, cc, ts = arr
    N = len(hh)
    day = ts // 86400000
    db = np.flatnonzero(np.diff(day)) + 1
    seg_start = np.r_[0, db]
    seg_end = np.r_[db, len(day)]
    seg_len = seg_end - seg_start
    log('%s  信号 %d  天数 %d  持有 %d 根(=%.1f 小时)  收集 %.0fs'
        % (fname, N, len(seg_start), H, H * tf / 3600.0, time.time() - t0))

    j_tp = np.stack([first_hit(hh, g, True) for g in TPS])     # 100×N
    j_sl = np.stack([first_hit(ll, -g, False) for g in SLS])   # 100×N
    last = cc[:, -1]

    best = []          # (final, tp, sl, tgt, mean, winrate)
    npos = 0
    tot = 0
    for a in range(len(TPS)):
        jt = j_tp[a]
        # 本 TP 下所有 SL 的逐笔收益矩阵 (100, N)
        nets = np.empty((len(SLS), N), dtype=np.float32)
        for b in range(len(SLS)):
            js = j_sl[b]
            none = (jt == 127) & (js == 127)
            nets[b] = np.where(none, last - COST2,
                               np.where(jt < js, TPS[a] - COST2, -SLS[b] - COST2))
        means = nets.mean(axis=1)
        wins = (nets > 0).mean(axis=1)
        npos += int((means > 0).sum())
        tot += len(means)

        lr = np.log1p(nets.astype(np.float64))      # (100, N)
        # 日内累计（与 target 无关，只算一次）
        cs = np.empty_like(lr)
        for s, e in zip(seg_start, seg_end):
            cs[:, s:e] = np.cumsum(lr[:, s:e], axis=1)
        day_end_val = np.exp(cs[:, seg_end - 1])    # (100, days)
        for k, tg in enumerate(TGT):
            lim = np.log(tg)
            lreal = np.zeros(len(SLS))
            for s, e in zip(seg_start, seg_end):
                blk = cs[:, s:e]
                has = (blk >= lim).any(axis=1)
                lreal += np.where(has, lim, blk[:, -1])
            idx = np.argsort(lreal)[::-1][:3]
            for i in idx:
                lv = float(lreal[i])
                fv = float(np.exp(lv)) if lv < 690.0 else 1e300
                best.append((fv, float(TPS[a]), float(SLS[i]), float(tg),
                             float(means[i]), float(wins[i])))
        if a % 20 == 0:
            log('  ... TP=%.1f%% 完成  %.0fs' % (TPS[a] * 100, time.time() - t0))
    del arr

    best.sort(key=lambda z: -z[0])
    log('')
    log('=' * 104)
    log('%s  510,000 组跑完  耗时 %.0fs' % (fname, time.time() - t0))
    log('=' * 104)
    log('逐笔均值为正的组合：%d / %d（%.1f%%）' % (npos, tot, npos / tot * 100))
    log('')
    log('=== 日内停止机制下「终值」最高的 15 组（起点 1.0）===')
    log('%-8s %-8s %-10s %12s %11s %8s' % ('止盈TP', '止损SL', '日内停手N', '终值', '均值/笔', '胜率'))
    log('-' * 66)
    for v, tp, sl, tg, m, w in best[:15]:
        vt = ('%.3e' % v) if v > 1e5 else ('%.3f' % v)
        log('%-8s %-8s %-10s %12s %+10.4f%% %7.1f%%' %
            ('%.1f%%' % (tp * 100), '%.1f%%' % (sl * 100), '%.2fx' % tg, vt, m * 100, w * 100))
    if best:
        v, tp, sl, tg, m, w = best[0]
        log('')
        log('★ 三线最优：止盈 %.1f%% / 止损 %.1f%% / 日内到 %.2fx 停手 → 终值 %s（均值 %+.4f%%/笔）'
            % (tp * 100, sl * 100, tg, ('%.3e' % v) if v > 1e5 else ('%.3f' % v), m * 100))
    return dict(file=fname, npos=npos, tot=tot, best=best[0] if best else None)


def main():
    log('=' * 104)
    log('三线全组合：止盈 × 止损 × 日内停止阈值   510,000 组   %s' % time.strftime('%Y-%m-%d %H:%M'))
    log('止盈/止损 0.5%%~50%% 步长 0.5%%（各 100 档）｜日内停止 1.00~1.50 步长 0.01（51 档）')
    log('=' * 104)
    files = sys.argv[1:] or ['klines_5m_3y.csv']
    for f in files:
        tf = 300 if '5m' in f else (60 if '1m' in f else (86400 if '1d' in f else 3600))
        run(f, tf)
        log('')
    log('诚实边界：零滑点；TP 按 high、SL 按 (1-SL) 触价，均假设能成交；日内按 UTC 日切分；')
    log('         含样本内；终值取每档 target 的前 3 名参与总排名。')
    with open(EVI + '/grid3.txt', 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n\n')
    log('已写 %s/grid3.txt' % EVI)


if __name__ == '__main__':
    main()
