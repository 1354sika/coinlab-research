# -*- coding: utf-8 -*-
"""
gambler_sl_sweep · 纯算法（不用模型）：TP=+5%，止损三档 -2% / -3% / -4%
  触发：240min 低点 ×1.03，当根收盘价买入
  持有上限 1440 分钟（24h）
  输出：命中率 / 止损率 / 超时率 / 净期望 / 平衡线 / 前后两半 / 按月
"""
import os, time, numpy as np, pandas as pd

ROOT = r'D:\中信证券\coinlab'
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
OUT = os.path.join(ROOT, 'evidence', 'cycle')

LOW_WIN = 240
CONFIRM = 0.03
HOLD = 1440
TP = 0.05
FEE = 0.005
CAP = 1500
SLS = [-0.02, -0.03, -0.04, -0.06]   # -6% 作对照

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_sl_sweep · 纯算法：TP=+5%，止损 -2% / -3% / -4%（-6% 对照）')
    log('=' * 100)
    log(f'  触发：240min低点×{1+CONFIRM:.2f}  当根收盘价买入  CAP={CAP}/币')
    log(f'  持有上限 {HOLD} 分钟  成本 {FEE:.2%}/笔（双边 taker+滑点）')

    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'open', 'high', 'low', 'close'])
    syms = sorted(df['symbol'].unique())
    log(f'  币 {len(syms)}  行 {len(df):,}')

    # 每个 SL 档：收集 (ret, exit_kind, t_enter)
    rets = {sl: [] for sl in SLS}
    kinds = {sl: [] for sl in SLS}
    tenter = {sl: [] for sl in SLS}

    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        if len(g) < LOW_WIN + HOLD + 10:
            continue
        ot = g['open_time'].values.astype(np.int64)
        h = g['high'].values.astype(np.float64)
        l = g['low'].values.astype(np.float64)
        c = g['close'].values.astype(np.float64)
        op = g['open'].values.astype(np.float64)
        n = len(c)

        lo_win = pd.Series(l).rolling(LOW_WIN, min_periods=60).min().values
        entry_line = lo_win * (1 + CONFIRM)
        m = (c >= entry_line) & (op <= entry_line * 1.03)
        m[0] = False
        m = m & np.isfinite(lo_win)
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > CAP:
            idxs = idxs[::max(1, len(idxs) // CAP)][:CAP]

        px = c[idxs]
        tp_p = px * (1 + TP)

        for sl in SLS:
            sl_p = px * (1 + sl)   # sl 是负数
            for k, idx in enumerate(idxs):
                end = min(idx + HOLD, n - 1)
                if end <= idx + 1:
                    continue
                sh = h[idx + 1:end + 1]
                sl_ = l[idx + 1:end + 1]
                ht = np.where(sh >= tp_p[k])[0]
                hs = np.where(sl_ <= sl_p[k])[0]
                it = int(ht[0]) if len(ht) else 10 ** 9
                isl = int(hs[0]) if len(hs) else 10 ** 9
                if it < 10 ** 9 and it < isl:
                    r = TP - FEE
                    kind = 'TP'
                elif isl < 10 ** 9 and isl < it:
                    r = sl + FEE * -1   # 止损：亏 sl，再扣成本
                    r = sl - FEE
                    kind = 'SL'
                elif isl < 10 ** 9 and it < 10 ** 9 and it == isl:
                    r = sl - FEE    # 同一根双触，保守算止损
                    kind = 'DUAL'
                else:
                    r = c[end] / px[k] - 1.0 - FEE
                    kind = 'TO'
                rets[sl].append(r)
                kinds[sl].append(kind)
                tenter[sl].append(ot[idx])

        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    log('\n' + '=' * 100)
    log('【一】三档止损（+ -6% 对照）：全部触发点都买，不用模型')
    log('=' * 100)
    log(f'  {"SL":>6} {"平衡线":>7} {"笔数":>8} {"命中率":>7} {"止损率":>7} {"超时率":>7} '
        f'{"均值":>8} {"中位":>8} {"胜率":>7} {"判定":>6}')
    for sl in SLS:
        r = np.array(rets[sl])
        kd = np.array(kinds[sl])
        if len(r) == 0:
            continue
        p_star = (abs(sl) + FEE) / (TP + abs(sl))
        hit = float((kd == 'TP').mean())
        los = float(((kd == 'SL') | (kd == 'DUAL')).mean())
        to = float((kd == 'TO').mean())
        win = float((r > 0).mean())
        mark = '✓赚' if r.mean() > 0 else '✗亏'
        log(f'  {sl:>6.1%} {p_star:>7.1%} {len(r):>8,} {hit:>6.1%} {los:>6.1%} {to:>6.1%} '
            f'{r.mean():>+7.2%} {np.median(r):>+7.2%} {win:>6.1%} {mark:>6}')

    # 前后两半 + 按月
    log('\n' + '=' * 100)
    log('【二】稳定性：时间前后两半 + 按月')
    log('=' * 100)
    for sl in SLS:
        r = np.array(rets[sl]); t = np.array(tenter[sl])
        if len(r) == 0:
            continue
        order = np.argsort(t)
        r = r[order]; t = t[order]
        half = len(r) // 2
        log(f'\n  --- SL {sl:.0%} ---')
        log(f'    前段 n={half:,}  均值 {r[:half].mean():+.2%}  胜率 {(r[:half]>0).mean():.1%}')
        log(f'    后段 n={len(r)-half:,}  均值 {r[half:].mean():+.2%}  胜率 {(r[half:]>0).mean():.1%}')
        # 按月
        mth = pd.to_datetime(t, unit='ms', utc=True).strftime('%Y-%m').values
        uq = sorted(set(mth))
        pos = 0
        for m0 in uq:
            mm = mth == m0
            n_m = int(mm.sum())
            if n_m < 20:
                continue
            mean_m = float(r[mm].mean())
            if mean_m > 0:
                pos += 1
            log(f'      {m0}  n={n_m:>6,}  均值 {mean_m:>+7.2%}  胜率 {(r[mm]>0).mean():>5.1%}')
        log(f'    正收益月 {pos}/{len([m0 for m0 in uq if (mth==m0).sum()>=20])}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('=' * 100)
    log('  · 不用模型，只要「均值 > 0」就说明纯规则可交易')
    log('  · 止损越紧：命中率越高（平衡线越低）但被扫次数也越多')
    log('  · 前后两半同号为正 ⇒ 不是单段运气')
    log('  · 若三档全负 ⇒ 纯规则不可行，必须靠模型')

    out = os.path.join(OUT, 'sl_sweep_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
