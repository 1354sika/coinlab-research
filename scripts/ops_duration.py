# -*- coding: utf-8 -*-
"""时间结构特征算子族 D1-D6（用户补充：持续性时间本身是重要参数）

用户原话：
  「什么时候涨跌，持续了多久？某一时刻的变化率，某段时间涨跌了多少？
    突然变这样还是持续的变这样，各个变化时间的持续时间的构成。」

对应算子：
  D1 段长      streak_len     —— 当前连涨/连跌了几根（涨为正、跌为负）
  D2 段速率    seg_speed      —— |累计涨跌| / 段长（快涨还是慢涨）
  D3 段幅度    seg_amp        —— 当前这段累计涨跌了多少
  D4 突变标记  jump_flag      —— |单根收益| > kσ（突然变）
  D5 漂移标记  drift_flag     —— 单根小、但同向连续（持续变）
  D6 段构成    seg_composition—— 窗口内：变号次数、平均段长（时间构成）

用法：
  from ops_duration import d1_streak_len, d4_jump_flag, all_duration_features
"""
import numpy as np
import pandas as pd


def d1_streak_len(ret1: pd.DataFrame) -> pd.DataFrame:
    """D1 连涨/连跌段长（涨段为正、跌段为负），O(n) 向量化"""
    r = ret1.values
    s = np.sign(r).astype('float32')          # -1 / 0 / +1
    T, N = s.shape
    out = np.zeros((T, N), dtype='float32')
    for j in range(N):
        col = s[:, j]
        newseg = np.r_[True, col[1:] != col[:-1]]
        newseg |= (col == 0)                  # 零变动处强制开新段（修：原来把 ret==0 归入跌，造出 35 万根的假段）
        seg_id = np.cumsum(newseg) - 1
        start = np.where(newseg)[0][seg_id]
        run = (np.arange(T) - start + 1).astype('float32')
        out[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
    return pd.DataFrame(out, index=ret1.index, columns=ret1.columns)


def d2_seg_speed(close: pd.DataFrame, streak: pd.DataFrame) -> pd.DataFrame:
    """D2 段速率：当前段的累计涨跌幅 / 段长（正=上涨段，单位 %/根）—— 向量化"""
    L = streak.abs().values.astype('int64').clip(1, None)
    T, N = L.shape
    cv = close.values
    rows = np.arange(T)[:, None]
    start_row = np.clip(rows - (L - 1), 0, T - 1)
    start_px = np.take_along_axis(cv, start_row, axis=0)
    out = ((cv / start_px - 1.0) / L * 100.0).astype('float32')
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def d3_seg_amp(close: pd.DataFrame, streak: pd.DataFrame) -> pd.DataFrame:
    """D3 段幅度：当前段累计涨跌幅（%）—— 向量化"""
    L = streak.abs().values.astype('int64').clip(1, None)
    T, N = L.shape
    cv = close.values
    rows = np.arange(T)[:, None]
    start_row = np.clip(rows - (L - 1), 0, T - 1)
    start_px = np.take_along_axis(cv, start_row, axis=0)
    out = ((cv / start_px - 1.0) * 100.0).astype('float32')
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def d4_jump_flag(ret1: pd.DataFrame, win: int = 240, k: float = 3.0) -> pd.DataFrame:
    """D4 突变标记：|单根收益| > k × 滚动标准差"""
    sd = ret1.rolling(win, min_periods=max(20, win // 4)).std()
    return (ret1.abs() > k * sd).astype('float32')


def d5_drift_flag(ret1: pd.DataFrame, win: int = 240, frac: float = 0.5) -> pd.DataFrame:
    """D5 漂移标记：单根幅度 <= frac×σ（缓慢移动，配合 D1 的段长构成"持续变"）"""
    sd = ret1.rolling(win, min_periods=max(20, win // 4)).std()
    return (ret1.abs() <= frac * sd).astype('float32')


def d6_seg_composition(ret1: pd.DataFrame, win: int = 240):
    """D6 段构成：窗口内变号次数 + 平均段长（衡量"震荡碎"还是"趋势整"）"""
    s = np.sign(ret1.values)
    chg = (s[1:] != s[:-1]).astype('float32')
    df = pd.DataFrame(chg, index=ret1.index[1:], columns=ret1.columns)
    seg_count = df.rolling(win, min_periods=max(20, win // 4)).sum()
    avg_seg_len = win / (seg_count + 1.0)
    return seg_count, avg_seg_len


def d7_frozen_len(ret1: pd.DataFrame) -> pd.DataFrame:
    """D7 冻结段长：连续「价格完全不变」已持续几根
    （用户：长期不动的币也是重要观察，不是噪声）
    """
    f = (ret1.values == 0)
    T, N = f.shape
    out = np.zeros((T, N), dtype='float32')
    idx = np.arange(T)
    for j in range(N):
        col = f[:, j]
        if not col.any():
            continue
        last_false = np.where(~col, idx, -1)
        last_false = np.maximum.accumulate(last_false)
        out[:, j] = np.where(col, idx - last_false, 0).astype('float32')
    return pd.DataFrame(out, index=ret1.index, columns=ret1.columns)


def d8_market_frozen_ratio(ret1: pd.DataFrame, win: int = 60):
    """D8 全市场冻结率：每个时刻处于「不动」状态的币占比
    （用户：所有币是一体的 —— 这是个市场整体冷热指标）
    返回 (瞬时值 Series, 滚动均值 Series)
    """
    f = (ret1.values == 0).astype('float32')
    ratio = f.mean(axis=1)
    s = pd.Series(ratio, index=ret1.index)
    return s, s.rolling(win, min_periods=max(5, win // 4)).mean()


def all_duration_features(close: pd.DataFrame, ret1: pd.DataFrame, win: int = 240) -> dict:
    """一次算出全部时间结构特征"""
    streak = d1_streak_len(ret1)
    feats = {
        'D1_streak_len': streak,
        'D4_jump_flag':  d4_jump_flag(ret1, win),
        'D5_drift_flag': d5_drift_flag(ret1, win),
    }
    feats['D2_seg_speed'] = d2_seg_speed(close, streak)
    feats['D3_seg_amp'] = d3_seg_amp(close, streak)
    sc, asl = d6_seg_composition(ret1, win)
    feats['D6_seg_count'] = sc
    feats['D7_frozen_len'] = d7_frozen_len(ret1)
    _, d8r = d8_market_frozen_ratio(ret1, 60)
    feats['D8_market_frozen'] = pd.DataFrame(
        np.repeat(d8r.values[:, None], ret1.shape[1], axis=1),
        index=ret1.index, columns=ret1.columns)
    feats['D6_avg_seg_len'] = asl
    return feats


# ------------------------------------------------------------------ 自测 + 实测验证
if __name__ == '__main__':
    import os
    DATA = r'D:\中信证券\coinlab\data'

    def say(*a): print(*a, flush=True)

    say('=' * 100)
    say('时间结构特征 D1-D6：算子自测 + 暴涨前对照')
    say('=' * 100)

    p = os.path.join(DATA, 'klines_1m_1y.csv')
    df = pd.read_csv(p, usecols=['symbol', 'open_time', 'close'])
    df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
    df = df.sort_values(['symbol', 'open_time'])
    close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=5)
    T, N = close.shape
    say('matrix %d x %d' % (T, N))
    del df

    ret1 = close.pct_change().astype('float32')

    say('\n【算子自测（全样本统计，确认没算错）】')
    streak = d1_streak_len(ret1)
    sv = streak.values[np.isfinite(streak.values)]
    say('  D1 段长：均值 %.2f  P50 %.0f  P95 %.0f  最长 %d'
        % (sv.mean(), np.percentile(np.abs(sv), 50), np.percentile(np.abs(sv), 95), int(np.abs(sv).max())))

    jf = d4_jump_flag(ret1, 240, 3.0).values
    say('  D4 突变率：|ret| > 3σ 的占比 = %.4f%%' % (np.nanmean(jf) * 100))

    df5 = d5_drift_flag(ret1, 240, 0.5).values
    say('  D5 漂移率：|ret| <= 0.5σ 的占比 = %.2f%%' % (np.nanmean(df5) * 100))

    sc, asl = d6_seg_composition(ret1, 240)
    say('  D6 段数(240根内变号次数)：均值 %.1f -> 平均段长 %.1f 根' % (np.nanmean(sc.values), np.nanmean(asl.values)))

    d2 = d2_seg_speed(close, streak).values
    d3 = d3_seg_amp(close, streak).values
    say('  D2 段速率：均值 %+.4f %%/根    D3 段幅度：均值 %+.4f %%'
        % (np.nanmean(d2), np.nanmean(d3)))

    # ---------------- 暴涨前对照 ----------------
    say('\n【对照】暴涨前 vs 平常：时间结构特征有什么不同（4h 内 >= +50%）')
    WIN = 240
    rev = close.iloc[::-1]
    wmax = rev.rolling(WIN, min_periods=1).max().iloc[::-1]
    fwd_max = (wmax.shift(-1) / close - 1.0).values
    del rev, wmax

    pump = fwd_max >= 0.5
    rows, cols = np.where(pump)
    sel, last_c, last_r = [], -1, -10 ** 9
    for r, c in zip(rows, cols):
        if c != last_c or (r - last_r) >= WIN:
            sel.append((r, c))
        last_c, last_r = c, r
    say('  去重后暴涨事件 = %d 个' % len(sel))

    # 取每个事件起点前 60 根的 D 特征，与全样本对照
    keys = ['D1_abs_streak', 'D2_seg_speed', 'D3_seg_amp', 'D4_jump', 'D5_drift', 'D6_seg_count']
    pre = {k: [] for k in keys}
    for r, c in sel:
        lo = max(0, r - 60)
        if r - lo < 10: continue
        pre['D1_abs_streak'].append(np.nanmean(np.abs(streak.values[lo:r, c])))
        pre['D2_seg_speed'].append(np.nanmean(np.abs(d2[lo:r, c])))
        pre['D3_seg_amp'].append(np.nanmean(np.abs(d3[lo:r, c])))
        pre['D4_jump'].append(np.nanmean(jf[lo:r, c]))
        pre['D5_drift'].append(np.nanmean(df5[lo:r, c]))
        pre['D6_seg_count'].append(np.nanmean(sc.values[lo:r, c]))

    allv = {
        'D1_abs_streak': np.nanmean(np.abs(streak.values), axis=0),
        'D2_seg_speed':  np.nanmean(np.abs(d2), axis=0),
        'D3_seg_amp':    np.nanmean(np.abs(d3), axis=0),
        'D4_jump':       np.nanmean(jf, axis=0),
        'D5_drift':      np.nanmean(df5, axis=0),
        'D6_seg_count':  np.nanmean(sc.values, axis=0),
    }

    say('\n  %-16s %14s %14s %10s' % ('特征', '暴涨前60根均值', '全样本均值', '倍数'))
    for k in keys:
        a = np.nanmean(pre[k]) if pre[k] else float('nan')
        b = np.nanmean(allv[k])
        say('  %-16s %14.4f %14.4f %9.2fx' % (k, a, b, (a / b) if b else float('nan')))

    # ---------------- 长期不动（D7）之后会发生什么 ----------------
    say('\n【观察】长期不动之后：价格会怎样（用户：不动的币也是重要观察）')
    FEE_M = 0.0002
    FWD60 = (close.shift(-60) / close - 1.0).values.astype('float32')
    frozen = d7_frozen_len(ret1)
    fv = frozen.values
    base = FWD60[np.isfinite(FWD60)]
    say('  基准（全样本）：未来60m 均值 %+.4f%%  P(涨) %.2f%%'
        % (base.mean() * 100, (base > 0).mean() * 100))
    for th in [30, 120, 720, 2880]:
        m = (fv >= th) & np.isfinite(fv) & np.isfinite(FWD60)
        n = int(m.sum())
        if n < 500:
            say('  冻结 >= %5d 根：样本 %s（不足）' % (th, format(n, ',')))
            continue
        r = FWD60[m]
        say('  冻结 >= %5d 根  n=%10s  未来60m 均值 %+.4f%%  P(涨) %.2f%%  做多净maker %+.4f%%'
            % (th, format(n, ','), r.mean() * 100, (r > 0).mean() * 100,
               r.mean() * 100 - 2 * FEE_M * 100))

    say('\n【观察】全市场冻结率 D8（所有币是一体：市场冷热指标）')
    d8s, d8r = d8_market_frozen_ratio(ret1, 60)
    qs = [10, 50, 90, 99]
    say('  不动币占比：均值 %.2f%%  ' % (np.nanmean(d8r.values) * 100)
        + '  '.join('P%d %.2f%%' % (q, np.nanpercentile(d8r.values, q) * 100) for q in qs))
    hi = d8r.values >= np.nanpercentile(d8r.values, 90)
    lo = d8r.values <= np.nanpercentile(d8r.values, 10)
    mkt = ret1.median(axis=1).values
    say('  冻结率最高10%%时：全市场收益中位 %+.5f%%  最低10%%时：%+.5f%%'
        % (np.nanmean(mkt[hi]) * 100, np.nanmean(mkt[lo]) * 100))

    say('\nTOTAL done')
