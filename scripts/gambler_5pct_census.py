# -*- coding: utf-8 -*-
"""
gambler_5pct_census.py · 普查：扣除手续费+滑点后，单笔往返能到 +5% 的机会有多少次？

用户问题（2026-10-18）：
  "扣去滑点和手续费，能赚的一进一出达到本金的 5% 的次数是多少？我们要的不是只要，而是都要，但也不能亏。"

口径（严格区分两件事，报告里必须分开写）：
  A 摆动供给：|未来 H 根收益| >= 5% + 成本 的次数
     → 这是"市场给了多少次 5% 的摆动"，不代表你抓得住（方向未知）
  B 完美上界：假设每次方向都判对，按去重后的摆动次数复利 → 这是不可能达到的上界
  C 可执行次数：非重叠采样（每 H 根最多一次进出），扣成本后 >= 5%

成本：手续费 taker 0.10%×2 + 滑点 0.10%×2 = 往返 0.40%（同时给 0.20% / 0.60% 对照）

用法：python -X utf8 gambler_5pct_census.py
输出：evidence/gambler_5pct_census.txt / .json
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT = r'D:\中信证券\coinlab\evidence'
H_LIST = [12, 48, 288, 864]          # 5m 根：1h / 4h / 1d / 3d
H_NAME = {12: '1小时', 48: '4小时', 288: '1天', 864: '3天'}
COSTS = [0.002, 0.004, 0.006]        # 往返成本：乐观 / 基准 / 悲观
TH = 0.05
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)


def merge_runs(mask):
    """把连续 True 段合并成事件，返回起点索引"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.array([], dtype=int)
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    if m[0]:
        starts = np.concatenate([[0], starts])
    return starts


def main():
    t0 = time.time()
    say('=' * 100)
    say('gambler_5pct_census · 扣费扣滑点后，单笔往返 +5% 的机会次数')
    say(f'  门槛 {TH:.0%}  持有 {[H_NAME[h] for h in H_LIST]}  成本对照 {[f"{c:.1%}" for c in COSTS]}')
    say('=' * 100)

    dt = {'symbol': 'category', 'open_time': 'int64', 'close': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())
    say(f'  数据 {len(df):,} 行 / {len(syms)} 币  {time.time()-t0:.0f}s')

    rows = []
    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        c = g.close.values.astype(np.float64)
        t = g.open_time.values.astype(np.int64)
        n = len(c)
        if n < 5000:
            continue
        for H in H_LIST:
            if n <= H + 100:
                continue
            fwd = np.full(n, np.nan)
            fwd[:n - H] = c[H:] / np.maximum(c[:n - H], 1e-12) - 1
            ok = np.isfinite(fwd)
            span_days = (t[-1] - t[0]) / 86400e3
            for cost in COSTS:
                net_abs = np.abs(fwd) - cost
                hit = ok & (net_abs >= TH)
                starts = merge_runs(hit)
                # 非重叠可执行：固定网格每 H 根最多一进一出（可执行上界）
                exec_hits = int(np.nansum(hit[::H]))
                rows.append(dict(sym=str(sym), H=H, cost=cost, n=n, days=span_days,
                                 events=int(len(starts)), exec_hits=int(exec_hits),
                                 mean_abs=float(np.nanmean(np.abs(fwd))),
                                 p90_abs=float(np.nanpercentile(np.abs(fwd), 90))))
        if (si + 1) % 20 == 0:
            say(f'    … {si+1}/{len(syms)} 币  {time.time()-t0:.0f}s')

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'gambler_5pct_census.csv'), index=False, encoding='utf-8-sig')

    say('\n' + '=' * 100)
    say('【一】摆动供给：|H 根收益| ≥ 5%+成本 的次数（去重成事件；重叠严重，仅供看供给量）')
    say(f'  {"持有":<6}{"成本":<7}{"事件总数":>9}{"每币每天":>10}{"每币每周":>10}{"每笔平均绝对波动":>16}')
    for H in H_LIST:
        for cost in COSTS:
            s = R[(R.H == H) & (R.cost == cost)]
            if not len(s):
                continue
            ev = s.events.sum()
            per_day = (s.events / s.days).mean()
            say(f'  {H_NAME[H]:<6}{cost:<7.2%}{ev:>9,}{per_day:>10.2f}{per_day*7:>10.2f}{s.mean_abs.mean():>16.2%}')

    say('\n【二】可执行次数：非重叠（每 H 根最多一进一出），扣成本后 ≥ 5%')
    say(f'  {"持有":<6}{"成本":<7}{"笔数":>9}{"每币每天":>10}{"每币每周":>10}')
    for H in H_LIST:
        for cost in COSTS:
            s = R[(R.H == H) & (R.cost == cost)]
            if not len(s):
                continue
            ex = s.exec_hits.sum()
            per_day = (s.exec_hits / s.days).mean()
            say(f'  {H_NAME[H]:<6}{cost:<7.2%}{ex:>9,}{per_day:>10.3f}{per_day*7:>10.3f}')

    say('\n【三】如果"都要"且每次都 5%：按可执行笔数复利（数学上界，不是可实现收益）')
    for H in [48, 288, 864]:
        s = R[(R.H == H) & (R.cost == 0.004)]
        ex = int(s.exec_hits.sum())
        if ex:
            say(f'  {H_NAME[H]:<4} 全市场 {ex:,} 笔 → 每笔 +5% 复利 = '
                f'{1.05**min(ex, 200):.3e} 倍（只算 200 笔已溢出；说明"每次都赚 5%"不可能是目标，'
                f'方向正确率才是全部难点）')

    say('\n【四】结论分开写')
    say('  · 市场供给：5% 的摆动是有的（见【一】），所以"机会次数"不是瓶颈')
    say('  · 真正瓶颈：方向。上面的次数都是 |收益| ≥ 5%，方向判错就是 -5%')
    say('  · 所以"都要且不能亏"= 要求方向几乎全对，这是模型要证明的事，不是数据能直接给的')
    say(f'\nTOTAL {time.time()-t0:.0f}s')
    with open(os.path.join(OUT, 'gambler_5pct_census.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    with open(os.path.join(OUT, 'gambler_5pct_census.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(h=H_LIST, costs=COSTS, th=TH,
                       summary=R.groupby(['H', 'cost'])[['events', 'exec_hits', 'days']].sum().reset_index().to_dict('records')),
                  f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
