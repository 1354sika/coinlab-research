# -*- coding: utf-8 -*-
"""analyze_1s_latency.py — 秒级策略加上实测网络延迟后还剩多少

实测延迟（measure_latency.py）：
  live_recorder 轮询 2.00s + HTTP RTT 0.31s + 数据滞后 0.24s = 2.54 秒

所以本脚本把"信号出现 -> 成交"的延迟设为 DELAY 秒，扫 0/3/5/10 秒，
看每一档的均值衰减。0 秒是"骗自己"，2.54 秒是现实。
"""
import os, glob, time
import numpy as np
import pandas as pd
from collections import defaultdict

LIVE = r'D:\中信证券\coinlab\data\live'
EVI = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(EVI, 'latency_impact.txt')

DROPS = [0.005, 0.008, 0.010]
TP = 0.020
SL = 0.010
DELAYS = [0, 3, 5, 10]      # 秒
MAXSEC = 3600
COST = 0.0025

L = []


def p(s=''):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
files = sorted(glob.glob(os.path.join(LIVE, '*', 'klines_1s.csv')))
p('=' * 100)
p('秒级策略 + 网络延迟实测影响  |  延迟档 %s 秒' % DELAYS)
p('=' * 100)

frames = []
for f in files:
    day = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f, usecols=['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        p('  %s FAIL %s' % (day, str(e)[:60]))
        continue
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df = df.dropna(subset=['open_time'])
    df = df[df['symbol'].astype(str).str.endswith('USDT')]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'close'])
    p('  %s rows=%d syms=%d' % (day, len(df), df['symbol'].nunique()))
    frames.append(df)

if not frames:
    p('没有任何文件读入成功'); raise SystemExit
D = pd.concat(frames, ignore_index=True)
p('concat rows=%d  dtypes: open=%s close=%s' % (len(D), D['open'].dtype, D['close'].dtype))
D = D[D['close'] > 0]
p('close>0 rows=%d' % len(D))
D = D.drop_duplicates(subset=['symbol', 'open_time'])
D = D.sort_values(['symbol', 'open_time'])
p('去重后 rows=%d  币=%d' % (len(D), D['symbol'].nunique()))

acc = defaultdict(list)      # (drop, delay) -> [net_r]
n_cand = defaultdict(int)

for sym, g in D.groupby('symbol', sort=False):
    t = g['open_time'].to_numpy(np.int64)
    c = g['close'].to_numpy(np.float64)
    n = len(c)
    if n < 200:
        continue
    gap = np.diff(t) / 1000.0
    cont = np.concatenate([[True], gap <= 5.0])

    LB = 60
    ret = np.zeros(n)
    ret[LB:] = c[LB:] / c[:-LB] - 1.0
    ok = np.zeros(n, dtype=bool)
    ok[LB:] = cont[LB:]

    for drop in DROPS:
        idx = np.where(ok & (ret <= -drop))[0]
        # 只保留后面还有足够空间的
        idx = idx[(idx + 60 < n)]
        n_cand[drop] += len(idx)

        for i in idx:
            for delay in DELAYS:
                j = i + delay
                if j >= n:
                    continue
                # 成交价：延迟后那一秒的 close（近似"你看到并下单"时的价）
                ep = c[j]
                if not np.isfinite(ep) or ep <= 0:
                    continue
                hi = j + 1
                lo = j + 1
                seg = c[hi:hi + MAXSEC] / ep - 1.0
                if len(seg) < 10:
                    continue
                m_sl = seg <= -SL
                js = int(np.argmax(m_sl)) if m_sl.any() else 10 ** 9
                tpm = seg >= TP
                jt = int(np.argmax(tpm)) if tpm.any() else 10 ** 9
                if js < jt and js < MAXSEC:
                    r = seg[js]
                elif jt < MAXSEC:
                    r = seg[jt]
                else:
                    r = seg[-1]
                acc[(drop, delay)].append(r - COST)

p('')
p('=== 结果（TP %.0f%% / SL %.0f%% / 成本 %.2f%%）' % (TP * 100, SL * 100, COST * 100))
p('  %-10s %8s %10s %10s %10s %10s' %
  ('跌幅', '信号数', '0秒(骗自己)', '3秒', '5秒', '10秒'))
p('  ' + '-' * 66)
for drop in DROPS:
    line = '  %-9.1f%% %8d' % (drop * 100, n_cand[drop])
    for delay in DELAYS:
        a = np.array(acc[(drop, delay)], dtype=np.float64)
        line += '%9.3f%%' % (100 * a.mean()) if len(a) >= 30 else '%10s' % '-'
    p(line)

p('')
p('=== 每一档的胜率')
p('  %-10s %10s %10s %10s %10s' % ('跌幅', '0秒', '3秒', '5秒', '10秒'))
p('  ' + '-' * 54)
for drop in DROPS:
    line = '  %-9.1f%%' % (drop * 100)
    for delay in DELAYS:
        a = np.array(acc[(drop, delay)], dtype=np.float64)
        line += '%9.1f%%' % (100 * (a > 0).mean()) if len(a) >= 30 else '%10s' % '-'
    p(line)

p('')
p('=== 延迟把收益吃掉多少（以 1 秒跌 1% 为基准）')
base = np.array(acc[(0.010, 0)], dtype=np.float64)
for delay in DELAYS:
    a = np.array(acc[(0.010, delay)], dtype=np.float64)
    if len(a) < 30 or not len(base):
        continue
    p('  延迟 %-3d 秒：均值 %+7.3f%%（相比 0 秒 %+7.3f%%，差 %+6.3f pp，剩 %.0f%%）'
      % (delay, 100 * a.mean(), 100 * base.mean(),
         100 * (a.mean() - base.mean()),
         100.0 * a.mean() / base.mean() if base.mean() else 0))

p('')
p('--- 诚实边界 ---')
p('1. 只有 4 天、30 币、约 65%% 的秒有数据；样本做不了统计。')
p('2. "延迟后成交"用第 i+delay 秒的 close 近似，真实是市价单吃盘口，会更差。')
p('3. 65%% 覆盖时 i+delay 那一秒可能不存在，本脚本只能取到存在的行。')
p('4. 未模拟成交率：急跌时市价单可能部分成交。')
p('5. 2.54 秒是实测中位，P90 会到 3 秒以上；10 秒档是压力测试。')
p('6. 已下架归零币不在池内。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written %s  TOTAL %.0fs' % (OUT, time.time() - t0))
