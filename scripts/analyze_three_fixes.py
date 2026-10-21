# -*- coding: utf-8 -*-
"""analyze_three_fixes.py — 三个改进方向一次跑完

1) SL 扫描：止损 1%/1.5%/2%/2.5%/3%/4%/5%，TP 固定 3%
2) 入场确认：跌 5% 后等 1~2 根"不创新低"再买 / 等一根阳线再买
3) 按盘口流动性分档：用 depth_impact 实测的滑点给币分档，看收益差

口径固定：收盘价确认触线、扣 0.25% 往返成本、MAXBARS=2880（48h）
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'three_fixes.txt')

LOOKBACK = 15
DROP = 0.05
TP = 0.03
MAXBARS = 2880
COST = 0.0025
NSYM = 60
SLS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]

# depth_impact.txt 实测的 1000 USDT 卖出中位滑点（bp）
LIQ = {
    'ADAUSDT': 0.00, 'ALGOUSDT': 0.00, 'AVAXUSDT': 0.00, 'ARBUSDT': 0.00,
    'BNBUSDT': 0.00, 'DOTUSDT': 0.00, 'DOGEUSDT': 0.00, 'BTCUSDT': 0.00,
    'ETHUSDT': 0.00, 'ETCUSDT': 0.00, 'SOLUSDT': 0.00, 'PEPEUSDT': 0.00,
    'OPUSDT': 0.00, 'NEARUSDT': 0.00, 'LTCUSDT': 0.00, 'LINKUSDT': 0.00,
    'XRPUSDT': 0.00, 'UNIUSDT': 0.00, 'SUIUSDT': 0.00, 'TRXUSDT': 0.00,
    'HBARUSDT': 0.31, 'ATOMUSDT': 0.45, 'AAVEUSDT': 0.71, 'FILUSDT': 1.38,
    'INJUSDT': 1.78, 'TIAUSDT': 1.81, 'APTUSDT': 1.91, 'SEIUSDT': 2.01,
    'WIFUSDT': 3.33,
}
LIQ_GOOD = {k for k, v in LIQ.items() if v <= 0.31}
LIQ_BAD = {k for k, v in LIQ.items() if v >= 1.78}

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


def run_path(rr, i_sl, i_tp):
    """给定 rr 路径和触发的索引，返回 (exit_r, what)"""
    if i_sl < i_tp and i_sl < MAXBARS:
        return rr[i_sl], -1
    if i_tp < MAXBARS:
        return rr[i_tp], 1
    return rr[-1], 0


t0 = time.time()
files = sorted(x for x in os.listdir(DATA) if x.endswith('.csv'))[:NSYM]

# 结果容器
res_sl = {s: [] for s in SLS}      # SL 扫描（base 入场）
res_mode = {m: [] for m in ['base', 'wait1', 'wait2', 'green1']}
res_liq = {'good': [], 'mid': [], 'bad': [], 'unknown': []}

n_sig = 0
n_skip = 0
for fi in files:
    sym = fi[:-4]
    try:
        df = pd.read_csv(os.path.join(DATA, fi), usecols=['open', 'high', 'low', 'close'])
    except Exception:
        continue
    if len(df) < 50000:
        continue
    o = df['open'].to_numpy(np.float32)
    c = df['close'].to_numpy(np.float32)
    n = len(c)

    ret = c / np.roll(c, LOOKBACK) - 1.0
    ret[:LOOKBACK] = 0.0
    idx = np.where(ret <= -DROP)[0]
    idx = idx[(idx >= LOOKBACK) & (idx + MAXBARS + 4 < n)]
    n_sig += len(idx)

    sl_arr = np.array(SLS, dtype=np.float64)[:, None]

    for i in idx:
        # --- base: i+1 开盘买 ---
        def build(iv):
            if iv >= n - 2:
                return None
            ep = o[iv]
            if not np.isfinite(ep) or ep <= 0:
                return None
            seg = c[iv:iv + MAXBARS]
            return seg / ep - 1.0

        rr = build(i + 1)
        if rr is None:
            n_skip += 1
            continue

        # 1) SL 扫描（base）
        slm = rr[None, :] <= -sl_arr
        has = slm.any(axis=1)
        i_sl = np.where(has, slm.argmax(axis=1), 10 ** 9)
        tpm = rr >= TP
        i_tp = int(np.argmax(tpm)) if tpm.any() else 10 ** 9
        for k, s in enumerate(SLS):
            er, w = run_path(rr, int(i_sl[k]), i_tp)
            res_sl[s].append(er - COST)

        # 2) 入场确认（用 SL=2% 作代表，后面也能对照）
        SL_REP = 0.02
        def eval_mode(rr2):
            m = rr2 <= -SL_REP
            js = int(np.argmax(m)) if m.any() else 10 ** 9
            jt = int(np.argmax(rr2 >= TP)) if (rr2 >= TP).any() else 10 ** 9
            er, w = run_path(rr2, js, jt)
            return er - COST

        res_mode['base'].append(eval_mode(rr))

        # wait1: 要求 c[i+1] > c[i]（不创新低），入场 i+2
        if i + 1 < n and c[i + 1] > c[i]:
            rr2 = build(i + 2)
            if rr2 is not None:
                res_mode['wait1'].append(eval_mode(rr2))
        # wait2: 要求连续两根都高于信号根收盘
        if i + 2 < n and c[i + 1] > c[i] and c[i + 2] > c[i]:
            rr2 = build(i + 3)
            if rr2 is not None:
                res_mode['wait2'].append(eval_mode(rr2))
        # green1: 要求 i+1 是阳线，入场 i+2
        if i + 1 < n and c[i + 1] > o[i + 1]:
            rr2 = build(i + 2)
            if rr2 is not None:
                res_mode['green1'].append(eval_mode(rr2))

        # 3) 流动性分档（用 SL=2% 的 base 结果）
        er = res_mode['base'][-1]
        if sym in LIQ_GOOD:
            res_liq['good'].append(er)
        elif sym in LIQ_BAD:
            res_liq['bad'].append(er)
        elif sym in LIQ:
            res_liq['mid'].append(er)
        else:
            res_liq['unknown'].append(er)

p('')
p('=' * 104)
p('三个改进方向实测  15m跌5%% 信号  |  %d 币 / %d 信号 / 跳过 %d / %.0fs'
  % (NSYM, n_sig, n_skip, time.time() - t0))
p('=' * 104)


def summarize(arr):
    a = np.array(arr, dtype=np.float64)
    if not len(a):
        return None
    return dict(n=len(a), mean=a.mean(), med=np.median(a), win=(a > 0).mean())


def comp(arr):
    a = np.array(arr, dtype=np.float64)
    if not len(a):
        return np.nan
    v = 1000.0
    for x in a:
        v *= (1.0 + x)
        if v <= 0:
            return 0.0
    return v


p('')
p('=== 一、SL 扫描（TP 固定 3%%，base 入场，扣 0.25%% 成本）')
p('  %-8s %8s %8s %8s %10s %10s %14s' %
  ('SL', '笔数', '胜率', '中位', '均值/笔', '复利终值', '平均出场(亏损笔)'))
p('  ' + '-' * 78)
best_sl, best_v = None, -1
for s in SLS:
    a = np.array(res_sl[s], dtype=np.float64)
    if not len(a):
        continue
    wins = a[a > 0]
    loss = a[a <= 0]
    v = comp(res_sl[s])
    p('  %-8.1f%% %8d %7.1f%% %7.2f%% %9.3f%% %11.1fx %12.2f%%'
      % (s * 100, len(a), 100 * (a > 0).mean(), 100 * np.median(a),
         100 * a.mean(), v / 1000.0, 100 * (loss.mean() if len(loss) else 0)))
    if v > best_v:
        best_v, best_sl = v, s
p('  => 复利终值最高：SL=%.1f%%（%.1fx）' % (best_sl * 100, best_v / 1000.0))

p('')
p('=== 二、入场确认（SL=2%%，TP=3%%，扣 0.25%% 成本）')
p('  %-14s %8s %8s %7s %9s %10s %14s' %
  ('入场方式', '笔数', '占base', '胜率', '中位', '均值/笔', '复利终值'))
p('  ' + '-' * 78)
base_n = len(res_mode['base'])
names = {'base': '跌5%次根开盘买', 'wait1': '等1根不新低',
         'wait2': '等2根不新低', 'green1': '等1根阳线'}
for m in ['base', 'wait1', 'wait2', 'green1']:
    a = np.array(res_mode[m], dtype=np.float64)
    if not len(a):
        p('  %-14s %8s' % (names[m], '-'))
        continue
    p('  %-14s %8d %7.1f%% %7.1f%% %8.3f%% %10.1fx'
      % (names[m], len(a), 100.0 * len(a) / base_n,
         100 * (a > 0).mean(), 100 * np.median(a), comp(res_mode[m]) / 1000.0))

p('')
p('=== 三、按盘口流动性分档（SL=2%%，TP=3%%）')
p('  好档 = 1000USDT 卖出滑点 <= 0.31bp（20 币）；差档 = >= 1.78bp（5 币）')
p('  %-10s %8s %8s %8s %9s %10s' % ('档位', '笔数', '胜率', '中位', '均值/笔', '复利终值'))
p('  ' + '-' * 60)
for k, nm in [('good', '好(20币)'), ('mid', '中(4币)'), ('bad', '差(5币)'), ('unknown', '无盘口数据')]:
    a = np.array(res_liq[k], dtype=np.float64)
    if not len(a):
        p('  %-10s %8s' % (nm, '-'))
        continue
    p('  %-10s %8d %7.1f%% %7.2f%% %8.3f%% %10.1fx'
      % (nm, len(a), 100 * (a > 0).mean(), 100 * np.median(a), 100 * a.mean(),
         comp(res_liq[k]) / 1000.0))

p('')
p('--- 诚实边界 ---')
p('1. 收盘价确认触线，不含执行延迟与真实滑点；成本 0.25%% 是线性近似。')
p('2. SL 扫描的"复利终值"是逐笔满仓顺序复利，未模拟多币并发。')
p('3. 入场确认的笔数会变少（wait2 尤其），笔数不同时终值不可直接比大小。')
p('4. 流动性分档只覆盖 depth_impact 实测的 29 币，其余 31 币归"无盘口数据"。')
p('5. 样本含样本内参数，未做时间切分外推。')
p('6. 已下架归零币不在池内。')

os.makedirs(EVI, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
