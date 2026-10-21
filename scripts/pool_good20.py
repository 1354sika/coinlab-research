# -*- coding: utf-8 -*-
"""pool_good20.py — 只做流动性好的币，24h 盯盘（池 N 分仓），本金 1000 的年化

口径：
  · 信号 15m 跌 5%（LOOKBACK=15, DROP=5%）
  · TP 3% / SL 2%（three_fixes 里 SL 档差别不大，取 2% 作代表）
  · 只做 depth_impact 实测滑点 <= 0.31bp 的币（"好档"）
  · 池 N：每个信号占 equity/N，满仓时新信号丢弃（单账号真实约束）
  · 成本 0.25% 往返
"""
import os, time
import numpy as np
import pandas as pd

DATA = r'E:\binance\klines_1m\1m'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'pool_good.txt')

LOOKBACK = 15
DROP = 0.05
TP = 0.03
SL = 0.02
MAXBARS = 2880
COST = 0.0025
POOLS = [1, 2, 3, 5, 10, 20]

LIQ_GOOD = {
    'ADAUSDT', 'ALGOUSDT', 'AVAXUSDT', 'ARBUSDT', 'BNBUSDT', 'DOTUSDT',
    'DOGEUSDT', 'BTCUSDT', 'ETHUSDT', 'ETCUSDT', 'SOLUSDT', 'PEPEUSDT',
    'OPUSDT', 'NEARUSDT', 'LTCUSDT', 'LINKUSDT', 'XRPUSDT', 'UNIUSDT',
    'SUIUSDT', 'TRXUSDT', 'HBARUSDT',
}
LIQ_MID = {'ATOMUSDT', 'AAVEUSDT', 'FILUSDT'}      # 0.45~1.38 bp
LIQ_BAD = {'INJUSDT', 'TIAUSDT', 'APTUSDT', 'SEIUSDT', 'WIFUSDT'}

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(x for x in os.listdir(DATA) if x.endswith('.csv'))
p('=' * 100)
p('只做流动性好的币 + 24h 盯盘（池份仓）  总文件 %d' % len(files))
p('=' * 100)

# 先收集：每个 group 的信号 (symbol, entry_ms, exit_ms, ret)
def collect(group, label):
    sig = []
    syms_used = set()
    tmin = tmax = None
    for fi in files:
        sym = fi[:-4]
        if sym not in group:
            continue
        try:
            df = pd.read_csv(os.path.join(DATA, fi), usecols=['open_time', 'open', 'close'])
        except Exception:
            continue
        if len(df) < 50000:
            continue
        t = df['open_time'].to_numpy(np.int64)
        o = df['open'].to_numpy(np.float32)
        c = df['close'].to_numpy(np.float32)
        n = len(c)
        ret = c / np.roll(c, LOOKBACK) - 1.0
        ret[:LOOKBACK] = 0.0
        idx = np.where(ret <= -DROP)[0]
        idx = idx[(idx >= LOOKBACK) & (idx + MAXBARS + 2 < n)]
        if not len(idx):
            continue
        syms_used.add(sym)
        if tmin is None:
            tmin, tmax = int(t[idx[0]]), int(t[-1])
        else:
            tmin = min(tmin, int(t[idx[0]]))
            tmax = max(tmax, int(t[-1]))

        for i in idx:
            iv = i + 1
            ep = o[iv]
            if not np.isfinite(ep) or ep <= 0:
                continue
            seg = c[iv:iv + MAXBARS] / ep - 1.0
            m = seg <= -SL
            js = int(np.argmax(m)) if m.any() else 10 ** 9
            jt = int(np.argmax(seg >= TP)) if (seg >= TP).any() else 10 ** 9
            if js < jt and js < MAXBARS:
                k, r = js, seg[js]
            elif jt < MAXBARS:
                k, r = jt, seg[jt]
            else:
                k, r = MAXBARS - 1, seg[-1]
            sig.append((int(t[iv]), int(t[iv + k]), float(r - COST)))
    sig.sort()
    span_days = (tmax - tmin) / 86400000.0 if tmin else 0
    p('')
    p('【%s】币 %d 个 / 信号 %d 笔 / 时间跨度 %.1f 年（%.0f 天）'
      % (label, len(syms_used), len(sig), span_days / 365.25, span_days))
    p('  币：%s' % ', '.join(sorted(syms_used)))
    return sig, span_days


def pool_sim(sig, pool, span_days):
    """满仓跳过 + 等权 1/pool，equity 只计已平仓盈亏"""
    eq = 1000.0
    peak = eq
    mdd = 0.0
    open_until = []          # 每个仓位的平仓时间
    taken = 0
    for (t_in, t_out, r) in sig:
        open_until = [x for x in open_until if x > t_in]
        if len(open_until) >= pool:
            continue
        stake = eq / pool
        eq += stake * r
        open_until.append(t_out)
        taken += 1
    years = span_days / 365.25 if span_days > 0 else 1
    ann = (eq / 1000.0) ** (1.0 / years) - 1 if eq > 0 else -1.0
    return dict(taken=taken, eq=eq, ann=ann, years=years)


res = {}
for grp, label in [(LIQ_GOOD, '流动性好 21 币'), (LIQ_GOOD | LIQ_MID, '好+中 24 币'),
                   (LIQ_BAD, '流动性差 5 币')]:
    sig, span = collect(grp, label)
    res[label] = []
    for N in POOLS:
        r = pool_sim(sig, N, span)
        res[label].append((N, r))
        p('    池%-3d  成交 %6d 笔  终值 %12.2f 元  年化 %8.2f%%'
          % (N, r['taken'], r['eq'], 100 * r['ann']))

p('')
p('=== 汇总：本金 1000，24h 盯盘，年化 ===')
p('  %-16s' % '币池' + ''.join('%8s' % ('池%d' % N) for N in POOLS))
p('  ' + '-' * 92)
for label in res:
    line = '  %-16s' % label
    for (N, r) in res[label]:
        line += '%8.1f%%' % (100 * r['ann'])
    p(line)

p('')
p('--- 诚实边界 ---')
p('1. 收盘价确认触线；成本 0.25%% 线性近似；不含执行延迟。')
p('2. 池 N 满仓跳过 = 单账号真实约束；equity 只计已平仓盈亏，未计浮亏（真实回撤更深）。')
p('3. 好档 21 币里有一部分不在字母序前 60 的样本中（如 BTC/ETH/SOL/PEPE/OP），实际参与更少。')
p('4. 已下架归零币不在池内（幸存者偏差）。')
p('5. 样本含样本内参数（TP3/SL2），未做时间切分外推。')
p('6. 年化是把整段跨度复利折算，早期年份参与币少，不代表每年都能这样。')

os.makedirs(EVI, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
