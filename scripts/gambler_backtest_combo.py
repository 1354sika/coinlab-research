# -*- coding: utf-8 -*-
"""
gambler_backtest_combo.py · 把「三张牌合成信号」从统计做成真交易回测

信号（已验证）：
  score = rank(disp240) − rank(breadth60) − rank(dow_cos)
  在 1 年 60 币、1 小时持有期上：top 20% 单笔 +0.074%，t=+2.64，4 段全正，打乱对照 −0.010%

这个脚本把它变成资金曲线：
  · 每 60 分钟决策一次；score ≥ 训练段 80 分位 ⇒ 开多（全市场 60 币等权）
  · 持有 60 分钟平仓（DEC = H，不重叠）
  · 扣往返手续费（maker 0.02%×2 / taker 0.10%×2 各跑一遍）
  · 输出：资金曲线、总收益、笔数、胜率、最大回撤、年化
  · 对照组：打乱序列做完全相同的交易；以及"随机开仓"基线

诚实边界：
  · 这是"大盘择时"（同时开 60 币），不是选币；实际成交需 60 笔 maker 单
  · 未计滑点、未计挂单成交率
  · 1 年数据，样本外仅约 4 个月

用法：python -X utf8 gambler_backtest_combo.py
输出：evidence/gambler_backtest_combo.txt / _curve.csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gambler_gate_cross import shuffle_panel
from gambler_combo import build_score

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
NSYM = int(os.environ.get('CROSS_N', '60'))
H = 60
DEC = 60
TRAIN = 0.70
FEE_MAKER = 0.0004      # 往返
FEE_TAKER = 0.0020      # 往返
TXT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def run(score, mkt_ret, cut, T, q80, fee, label):
    """score 每 DEC 步看一次；≥ q80 开多，持有 H 步平仓"""
    eq = 1.0
    curve = []
    trades = []
    t = cut
    while t + 1 + H < T:
        if score[t] >= q80:
            r = mkt_ret[t + 1:t + 1 + H]
            r = np.nansum(r)
            net = r - fee
            eq *= (1 + net)
            trades.append(net)
            curve.append((t, eq))
            t += H                      # 持有期结束再找下一次
        else:
            t += DEC
    a = np.array(trades)
    if len(a) == 0:
        return None
    peak = np.maximum.accumulate(np.array([1.0] + [c[1] for c in curve]))
    dd = (np.array([1.0] + [c[1] for c in curve]) / peak - 1).min()
    # 年化：样本外时长（分钟）
    minutes = (T - cut)
    years = minutes / (1440 * 365)
    ann = eq ** (1 / years) - 1 if years > 0 and eq > 0 else np.nan
    say(f'  {label:<22}{len(a):>7}{a.mean() * 100:>+9.3f}%{(a > 0).mean() * 100:>7.1f}%'
        f'{eq:>9.3f}{dd * 100:>+8.1f}%{ann * 100:>+9.1f}%')
    return dict(label=label, n=len(a), mean=a.mean() * 100, win=(a > 0).mean() * 100,
                equity=eq, mdd=dd * 100, ann=ann * 100)


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_backtest_combo · 合成信号真交易回测')
    say(f'  H={H} 分 | 决策间隔 {DEC} 分 | 币 {NSYM} | 训练定分位 {TRAIN:.0%}')
    say('=' * 100)

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'])
    syms = sorted(df.symbol.unique())[:NSYM]
    df = df[df.symbol.isin(syms)]
    df['t'] = pd.to_datetime(df.open_time, unit='ms')
    piv = df.pivot_table(index='t', columns='symbol', values='close').sort_index()[syms]
    C = piv.values.astype(np.float64)
    C[~np.isfinite(C) | (C <= 0)] = np.nan
    T, S = C.shape
    dow = piv.index.dayofweek.values
    say(f'  矩阵 {T:,} × {S} 币  {time.time() - t0:.0f}s')

    # 全市场等权组合的每分钟收益（对数）
    R = np.zeros((T, S))
    R[1:] = np.log(C[1:] / C[:-1])
    mkt_ret = np.nanmean(np.where(np.isfinite(R), R, np.nan), axis=1)
    mkt_ret = np.nan_to_num(mkt_ret, nan=0.0)
    say(f'  市场收益序列完成  {time.time() - t0:.0f}s')

    score = build_score(C, dow)
    cut = int(T * TRAIN)
    q80 = np.nanquantile(score[:cut], 0.80)
    q50 = np.nanquantile(score[:cut], 0.50)
    say(f'  训练段分位： q80={q80:.4f}  q50={q50:.4f}')

    rng = np.random.default_rng(20261018)
    C2 = shuffle_panel(C, rng)
    score2 = build_score(C2, dow)
    R2 = np.zeros((T, S))
    R2[1:] = np.log(C2[1:] / C2[:-1])
    mkt_ret2 = np.nan_to_num(np.nanmean(np.where(np.isfinite(R2), R2, np.nan), axis=1), nan=0.0)
    say(f'  打乱对照完成  {time.time() - t0:.0f}s')

    say('')
    say(f'{"策略":<22}{"笔数":>7}{"单笔均值":>9}{"胜率":>7}{"资金":>9}{"回撤":>8}{"年化":>9}')
    ok = score >= q80
    rows = []
    for fee, tag in ((FEE_MAKER, 'maker'), (FEE_TAKER, 'taker')):
        r = run(score, mkt_ret, cut, T, q80, fee, f'♠ 真实序列（{tag}）')
        if r:
            rows.append(r)
    r = run(score2, mkt_ret2, cut, T, q80, FEE_MAKER, '♣ 打乱对照（maker）')
    if r:
        rows.append(r)
    r = run(score, mkt_ret, cut, T, q50, FEE_MAKER, '♦ 一半时间在场（maker）')
    if r:
        rows.append(r)
    # 随机开仓基线
    rnd = np.random.default_rng(7)
    score_r = rnd.uniform(size=T)
    r = run(score_r, mkt_ret, cut, T, 0.80, FEE_MAKER, '♥ 随机信号（maker）')
    if r:
        rows.append(r)

    say('')
    say('=' * 100)
    say('【判据】')
    real = [x for x in rows if x['label'].startswith('♠')]
    best = max(real, key=lambda x: x['equity']) if real else None
    if best:
        say(f'  · 真实序列最好 ({best["label"]})：单笔 {best["mean"]:+.3f}%  资金 {best["equity"]:.3f}  '
            f'回撤 {best["mdd"]:+.1f}%  年化 {best["ann"]:+.1f}%')
        say(f'  · 采样期约 {(T - cut) / (1440 * 365):.2f} 年（样本外），年化数字波动很大，仅供参考')
    shuf = [x for x in rows if x['label'].startswith('♣')]
    if shuf:
        say(f'  · 打乱对照：单笔 {shuf[0]["mean"]:+.3f}%  资金 {shuf[0]["equity"]:.3f}'
            f'   （真实序列应明显更好才算数）')
    say('')
    say('  未计：滑点、挂单成交率（maker 假设全成交，实际可能部分不成交）')
    say('  全市场 60 币同时开仓 ⇒ 实际是押"大盘方向"，币间高度相关')
    say('')
    say(f'TOTAL {time.time() - t0:.0f}s')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'gambler_backtest_combo.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gambler_backtest_combo.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
