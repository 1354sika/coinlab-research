# -*- coding: utf-8 -*-
"""操盘手指纹算子库 F1-F6（对手建模的算子层落地）

设计原则（见 evidence/操盘手对手建模_设计_2026-10-15.md）：
  不做「价格回归」，做「对手状态推断」。
  每个算子输出一个可观测量，供上层做「阶段后验 / 转移概率」。

可得性：
  F1 主动买占比      aggTrades（m 字段）    —— 需逐笔，已启动录制
  F2 大单净流入      aggTrades（q 分位）    —— 需逐笔，已启动录制
  F3 撤单率          depth 快照序列         —— 需订单簿，已启动录制
  F4 盘口失衡        depth 快照             —— 需订单簿，已启动录制
  F5 秒级突刺率      1s K 线               —— 需秒级，已启动录制
  F6 量价背离度      1m/5m K 线             —— 【现在就能算，已有数据】

诚实边界：
  · 本文件只实现「算子」本身，**未做任何有效性验证**（除 F6 已单独验证：独立做空 ≈ 0，分月 4/13）
  · F1~F5 的实现代码已写好，但**输入数据（录制）目前只有几分钟**，无法给出任何结论
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- F6（唯一现在可算）
def f6_divergence(close: pd.DataFrame, qv: pd.DataFrame,
                  win_hi: int = 60, win_vs: int = 5, win_vl: int = 60):
    """F6 量价背离度：价格接近 N 根新高 + 量能萎缩

    参数
      close : T x N 收盘价宽表
      qv    : T x N 成交额宽表
    返回
      near_high  : close / 过去 win_hi 根最高价（越接近 1 越靠新高）
      volr       : 短期均量 / 长期均量（<1 = 萎缩）
      divergence : near_high * (1 - clip(volr,0,2))，越大越"背离"
    """
    hi = close.rolling(win_hi, min_periods=max(2, win_hi // 2)).max()
    near_high = (close / hi).astype('float32')
    v_s = qv.rolling(win_vs, min_periods=max(2, win_vs // 2)).mean()
    v_l = qv.rolling(win_vl, min_periods=max(2, win_vl // 2)).mean()
    volr = (v_s / v_l).astype('float32')
    divergence = (near_high * (1.0 - volr.clip(0, 2))).astype('float32')
    return near_high, volr, divergence


# ---------------------------------------------------------------- F1 / F2（逐笔）
def f1_active_buy_ratio(trades: pd.DataFrame, window_ms: int = 60_000) -> float:
    """F1 主动买占比
    trades 需含列：time(ms), price, qty, is_buyer_maker
    主动买 = is_buyer_maker == False（买方主动吃掉卖单）
    """
    if trades is None or len(trades) == 0:
        return float('nan')
    t = trades[trades['time'] >= (trades['time'].max() - window_ms)]
    if len(t) == 0:
        return float('nan')
    amt = t['price'].astype(float) * t['qty'].astype(float)
    buy = amt[~t['is_buyer_maker'].astype(bool)]
    tot = amt.sum()
    return float(buy.sum() / tot) if tot > 0 else float('nan')


def f2_large_order_netflow(trades: pd.DataFrame, q: float = 0.95,
                           window_ms: int = 60_000) -> float:
    """F2 大单净流入（正=主动买占优）
    大单 = 单笔成交额 > 该窗口内 q 分位
    """
    if trades is None or len(trades) == 0:
        return float('nan')
    t = trades[trades['time'] >= (trades['time'].max() - window_ms)]
    if len(t) == 0:
        return float('nan')
    amt = (t['price'].astype(float) * t['qty'].astype(float))
    thr = amt.quantile(q)
    big = t[amt >= thr]
    if len(big) == 0:
        return float('nan')
    b_amt = big['price'].astype(float) * big['qty'].astype(float)
    is_buy = ~big['is_buyer_maker'].astype(bool)
    return float(b_amt[is_buy].sum() - b_amt[~is_buy].sum())


# ---------------------------------------------------------------- F3 / F4（订单簿）
def f4_book_imbalance(snapshot: pd.DataFrame) -> float:
    """F4 盘口失衡：(bid 总量 - ask 总量) / 总量
    snapshot 需含列：side('bid'/'ask'), qty
    """
    if snapshot is None or len(snapshot) == 0:
        return float('nan')
    b = snapshot.loc[snapshot['side'] == 'bid', 'qty'].astype(float).sum()
    a = snapshot.loc[snapshot['side'] == 'ask', 'qty'].astype(float).sum()
    return float((b - a) / (b + a)) if (b + a) > 0 else float('nan')


def f3_cancel_ratio(prev_snap: pd.DataFrame, cur_snap: pd.DataFrame,
                    trades: pd.DataFrame | None = None) -> float:
    """F3 撤单率：上一快照存在、这一快照消失的挂单量 / 上一快照总量
    （严格版还应扣除「被成交」的部分；trades 传入时做粗略扣减）

    prev_snap / cur_snap 需含列：side, price, qty
    """
    if prev_snap is None or len(prev_snap) == 0:
        return float('nan')
    key = lambda d: d['side'].astype(str) + '@' + d['price'].astype(str)
    p = prev_snap.assign(k=key(prev_snap)).groupby('k')['qty'].apply(lambda s: float(s.astype(float).sum()))
    c = cur_snap.assign(k=key(cur_snap)).groupby('k')['qty'].apply(lambda s: float(s.astype(float).sum()))
    vanished_k = p.index.difference(c.index)
    vanished = float(p.loc[vanished_k].sum())
    total = float(p.sum())
    if total <= 0:
        return float('nan')
    # 粗略扣除「这一快照周期内成交的量」（用 trades 近 10 秒总额近似）
    if trades is not None and len(trades):
        traded = float((trades['price'].astype(float) * trades['qty'].astype(float)).sum())
        vanished = max(0.0, vanished - traded)
    return float(vanished / total)


# ---------------------------------------------------------------- F5（秒级）
def f5_spike_rate(k1s: pd.DataFrame, k_sigma: float = 3.0, window: int = 300) -> float:
    """F5 秒级突刺率：最近 window 秒中 |收益| > k_sigma * σ 的秒数占比
    k1s 需含列：close
    """
    if k1s is None or len(k1s) < 30:
        return float('nan')
    r = k1s['close'].astype(float).pct_change().tail(window).dropna()
    if len(r) < 20 or r.std() == 0:
        return float('nan')
    z = (r - r.mean()).abs() / r.std()
    return float((z > k_sigma).mean())


# ---------------------------------------------------------------- 汇总
def all_fingerprints(kline_kwargs=None, trades=None, prev_snap=None,
                     cur_snap=None, k1s=None) -> dict:
    """把当前能算的指纹都算出来（缺数据的返回 nan）"""
    out = {}
    if kline_kwargs:
        _, volr, div = f6_divergence(kline_kwargs['close'], kline_kwargs['qv'])
        out['F6_divergence'] = float(np.nanmean(div.values)) if div is not None else float('nan')
        out['F6_volr'] = float(np.nanmean(volr.values)) if volr is not None else float('nan')
    out['F1_active_buy_ratio'] = f1_active_buy_ratio(trades) if trades is not None else float('nan')
    out['F2_large_netflow'] = f2_large_order_netflow(trades) if trades is not None else float('nan')
    out['F3_cancel_ratio'] = f3_cancel_ratio(prev_snap, cur_snap, trades) if prev_snap is not None else float('nan')
    out['F4_book_imbalance'] = f4_book_imbalance(cur_snap) if cur_snap is not None else float('nan')
    out['F5_spike_rate'] = f5_spike_rate(k1s) if k1s is not None else float('nan')
    return out


if __name__ == '__main__':
    print('ops_fingerprint：F1-F6 算子库')
    print('  F6 立即可用（K 线）')

    # 自测：用已有数据算 F6
    import os
    p = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
    if os.path.exists(p):
        df = pd.read_csv(p, usecols=['symbol', 'open_time', 'close', 'quote_volume'])
        df['dt'] = pd.to_datetime(df.open_time, unit='ms', utc=True)
        df = df.sort_values(['symbol', 'open_time'])
        close = df.pivot(index='dt', columns='symbol', values='close').astype('float32').ffill(limit=5)
        qv = df.pivot(index='dt', columns='symbol', values='quote_volume').astype('float32')
        nh, vr, dv = f6_divergence(close, qv)
        print('  F6 自测：near_high 均值 %.4f  volr 均值 %.4f  divergence 均值 %.4f'
              % (np.nanmean(nh.values), np.nanmean(vr.values), np.nanmean(dv.values)))
    print('  F1-F5 需录制数据（live_recorder.py 正在写 data/live/<日期>/）')
