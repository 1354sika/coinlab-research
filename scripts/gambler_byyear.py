# -*- coding: utf-8 -*-
"""
gambler_byyear · 分年最优模式：每年是不是有自己的最佳打法？模式随年变吗？

数据：E:\binance\klines_1m\1m\*.csv（240 币 1m）→ 聚合成日线
思路：同族候选模式 = (回看 k 天, 动量阈值 thr, 持有 h 天)，全样本枚举，
      然后按年切，看每年各自的最优是否不同、且能不能跨界用。

判据：
  · 各年最优相同                                ⇒ 有通用模式，不需要自适应
  · 各年最优不同 + 非对角线仍 > 0               ⇒ 真漂移，必须自适应
  · 各年最优不同 + 非对角线 < 0（一跨界就崩）    ⇒ 过拟合，自适应是在追噪声
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = r'E:\binance\klines_1m\1m'
OUT = r'D:\中信证券\coinlab\evidence'
FEE = 0.0004                      # 往返 maker 2bp × 2
KS = [1, 3, 5, 10, 20]
THRS = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
HS = [1, 3, 5, 10, 20]


def say(*a):
    print(*a, flush=True)


def load_daily(maxf=0):
    files = sorted(glob.glob(os.path.join(SRC, '*.csv')))
    if maxf:
        files = files[:maxf]
    say(f'  文件 {len(files)} 个（{SRC}）')
    frames = []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c')
        except Exception as e:
            say(f'    跳过 {os.path.basename(f)}: {e}')
            continue
        if len(d) < 1000:
            continue
        d['day'] = (d.open_time // 86400000).astype(np.int64)
        g = d.groupby('day').agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
        g['sym'] = os.path.basename(f).replace('.csv', '')
        frames.append(g.reset_index())
        if (i + 1) % 40 == 0:
            say(f'    … {i+1}/{len(files)}  [{time.time()-t0:.0f}s]')
    if not frames:
        raise SystemExit('没有读到任何文件')
    df = pd.concat(frames, ignore_index=True)
    say(f'  日线 {len(df):,} 行 / {df.sym.nunique()} 币  [{time.time()-t0:.0f}s]')
    return df


def build_panel(df):
    M = df.pivot(index='day', columns='sym', values='c').sort_index()
    M = M.ffill()
    M = M.loc[:, M.notna().sum() > 300]          # 至少 300 天上市
    return M.values.astype(np.float64), M.index.values


def evaluate(C, days):
    T, N = C.shape
    t0 = time.time()
    R = np.diff(np.log(C), axis=0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(R, axis=0)])   # cs[t] = 前 t 天累计日志收益

    # 年号：用起始日对齐到公历年（open_time 是 ms，天序号 * 86400000）
    year = pd.to_datetime(days * 86400000, unit='ms').year.values

    out = {}
    for k in KS:
        mom = np.full((T, N), np.nan)
        mom[k:] = C[k:] / C[:-k] - 1.0
        for thr in THRS:
            sig = (mom < thr) if thr < 0 else (mom > thr)
            for h in HS:
                fut = np.full((T, N), np.nan)
                fut[:T - h] = cs[h:] - cs[:T - h]
                pnl = sig * fut - FEE
                per = {}
                for y in np.unique(year):
                    m = (year == y)[:, None] & ~np.isnan(fut) & (sig | True)
                    # 只在"有信号且收益可知"的位置统计（不做空 ⇒ 信号外的根收益为 0，单独说明）
                    m2 = m & (sig)
                    v = pnl[m2]
                    if v.size >= 300:
                        per[int(y)] = [float(v.mean()), int(v.size)]
                if per:
                    out[f'{k}|{thr}|{h}'] = per
    say(f'  {len(out)} 组候选 × {len(np.unique(year))} 年  [{time.time()-t0:.0f}s]')
    return out, year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maxf', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    say('=' * 110)
    say('gambler_byyear · 每年是不是有自己的最佳打法？模式随年变吗？')
    say('=' * 110)

    df = load_daily(a.maxf)
    C, days = build_panel(df)
    say(f'  矩阵 {C.shape[0]:,} 天 × {C.shape[1]} 币  [{time.time()-t0:.0f}s]')

    res, year = evaluate(C, days)
    uy = sorted(int(y) for y in np.unique(year))
    say(f'  年份 {uy[0]}..{uy[-1]}')

    # 【一】每年最优
    say()
    say('=' * 110)
    say('【一】每年的最佳模式  (k=回看天数, thr=动量阈值, h=持有天数；只统计该年有信号的根)')
    say('=' * 110)
    say(f'  {"年":>6} {"最优 k|thr|h":>16} {"单笔均值":>11} {"样本根":>10} {"正收益候选占比":>16}')

    best = {}
    for y in uy:
        cand = [(per[y][0], per[y][1], key) for key, per in res.items() if y in per]
        if not cand:
            continue
        cand.sort(reverse=True)
        m, n, key = cand[0]
        pos = sum(1 for c in cand if c[0] > 0) / len(cand)
        best[y] = key
        say(f'  {y:>6} {key:>16} {m*100:+10.4f}% {n:>10,} {pos*100:>14.1f}%')

    # 【二】跨年矩阵
    say()
    say('=' * 110)
    say('【二】跨年稳定性：第 i 年的最优参数，搬到第 j 年还灵不灵？（单笔均值 %）')
    say('=' * 110)
    say('  ' + ' '.join(f'{y:>9}' for y in uy) + '    ← 当年最优')
    for yi in uy:
        if yi not in best:
            continue
        per = res[best[yi]]
        row = [f'{per[yj][0]*100:+9.4f}' if yj in per else f'{"—":>9}' for yj in uy]
        say(f'{yi:>4} ' + ' '.join(row) + f'   ← {best[yi]}')

    # 【三】判据
    say()
    say('=' * 110)
    say('【三】判据')
    say('=' * 110)
    same = len(set(best.values())) == 1
    say(f'  各年最优是否全部相同：{"是（存在通用模式）" if same else "否（每年不同）"}')
    diag, off = [], []
    for yi in uy:
        if yi not in best:
            continue
        per = res[best[yi]]
        for yj in uy:
            if yj not in per:
                continue
            (diag if yj == yi else off).append(per[yj][0])
    if diag:
        dm, om = np.mean(diag), np.mean(off) if off else float('nan')
        say(f'  对角线（当年最优跑当年）  均值 {dm*100:+.4f}%   n={len(diag)}')
        say(f'  非对角线（搬到别年）      均值 {om*100:+.4f}%   n={len(off)}')
        if om > 0:
            say('  ⇒ 搬到别年仍为正：模式在漂移，但底层规律可迁移 —— 该自适应（换参数不换道理）')
        else:
            say('  ⇒ 一跨界就崩：每年的"最优"是过拟合噪声 —— 自适应只是在追噪声')

    path = os.path.join(OUT, 'byyear.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'best': {str(k): v for k, v in best.items()}, 'res': res}, f, ensure_ascii=False)
    say(f'  写入 {path}')
    say(f'用时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
