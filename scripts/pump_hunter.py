# -*- coding: utf-8 -*-
"""操盘手识别 v1：短期暴涨事件普查 + 生命周期时间结构 + 前后特征

用户口径（2026-10-16）：
  · 操盘手 = 短期（4~5 小时）内价格翻好几倍的币种 —— 这就是"识别操盘手"的可观测定义
  · 高期望就搞到就走，不逗留
  · 长期不动（D7/D8）与各种持续状态的时间结构是重要参数
  · 暴涨必暴跌 → 暴跌前的信号也要找

数据：klines_5m_502.csv（304 币 × 3 年 5m）—— 覆盖面最广
事件：W=48 根(=4h) 内 未来最高价 / 当前 close − 1 ≥ 门槛
      门槛 ∈ {50%, 100%, 200%, 300%}
去重：同一币相邻事件间隔 < W 根视为同一波，只取首个
输出：
  1. 事件普查表（各门槛：事件数、涨幅中位、启动→峰值耗时中位）
  2. 生命周期：峰值 → 最大回撤的耗时、回撤深度（修 event_pairs 的间隔 bug：从峰值后起算）
  3. 暴涨启动前 60 根 vs 全样本：量比 / 收益 / 波动 / D1 段长 / D4 突变率 / D7 冻结
  4. 暴跌前 12 根 vs 全样本：量比 / 收益
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
W5 = 48           # 4h @5m
H72 = 864         # 72h @5m
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('操盘手识别 v1：短期暴涨事件普查 + 生命周期 + 前后特征   数据=klines_5m_502.csv')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
say('读入 %s 行  %.0fs' % (format(len(df), ','), time.time() - t0))
df = df.sort_values(['symbol', 'open_time'])
def _piv(col):
    m = df.pivot(index='open_time', columns='symbol', values=col).astype('float32')
    m = m.where(m > 0)
    return m.ffill(limit=3)
close = _piv('close')
high = _piv('high')
low = _piv('low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
qv = qv.where(qv >= 0)
del df; gc.collect()
T, N = close.shape
syms = list(close.columns)
tm = close.index.values
say('matrix %d x %d  (%.1f 年)  %.0fs' % (T, N, (tm[-1] - tm[0]) / 31536000000.0, time.time() - t0))

# 未来 W5 根内的最高价（含当前）
fwd_high = high.iloc[::-1].rolling(W5, min_periods=1).max().iloc[::-1]
gain = (fwd_high / close - 1.0).values.astype('float32')
CV, HV, LV = close.values, high.values, low.values

# ---- 全样本基准特征 ----
ret1 = np.full((T, N), np.nan, np.float32); ret1[1:] = CV[1:] / CV[:-1] - 1.0
qv60 = pd.DataFrame(qv).rolling(60, min_periods=20).median().values
qv60 = np.where(qv60 > 0, qv60, np.nan)          # 除零保护（原来是 inf 污染均值）
volr = qv.values / qv60
ret60 = np.full((T, N), np.nan, np.float32); ret60[60:] = CV[60:] / CV[:-60] - 1.0
ret240 = np.full((T, N), np.nan, np.float32); ret240[240:] = CV[240:] / CV[:-240] - 1.0
vol = pd.DataFrame(ret1).rolling(240, min_periods=40).std().values
# D1 段长（涨为正/跌为负）
sgn = np.sign(np.nan_to_num(ret1))
d1 = np.zeros((T, N), np.float32)
for j in range(N):
    col = sgn[:, j]
    newseg = np.r_[True, col[1:] != col[:-1]] | (col == 0)
    seg_id = np.cumsum(newseg) - 1
    start = np.where(newseg)[0][seg_id]
    run = (np.arange(T) - start + 1).astype(np.float32)
    d1[:, j] = np.where(col == 0, 0.0, np.where(col > 0, run, -run))
# D4 突变率（|ret|>3σ）
d4 = (np.abs(ret1) > 3.0 * vol).astype(np.float32)
# D7 冻结
d7 = np.zeros((T, N), np.float32)
for j in range(N):
    fz = (CV[:, j] == np.roll(CV[:, j], 1))
    idx = np.arange(T); last_false = np.where(~fz, idx, -1)
    last_false = np.maximum.accumulate(last_false)
    d7[:, j] = np.where(fz, idx - last_false, 0)
say('基准特征算完  %.0fs' % (time.time() - t0))

FE = {'volr': volr, 'ret60': ret60, 'ret240': ret240, 'vol240': vol, 'D1': d1, 'D4': d4, 'D7': d7}
FEAT_NAMES = list(FE.keys())


def summarize(tag, idx_rc):
    """idx_rc: list of (row, col) 取该处特征均值"""
    if not idx_rc:
        say('  %s : 无样本' % tag); return
    rr = np.array([a for a, b in idx_rc]); cc = np.array([b for a, b in idx_rc])
    out = []
    for k in FEAT_NAMES:
        v = FE[k][rr, cc]
        out.append('%s=%+.4f' % (k, float(np.nanmean(v))))
    say('  %-22s n=%6d  %s' % (tag, len(idx_rc), '  '.join(out)))


say('\n' + '=' * 100)
say('【一】短期暴涨事件普查（4h 窗口内未来最高价相对当前价）')
say('=' * 100)
say('  %-8s %9s %10s %12s %12s %12s' % ('门槛', '事件数', '币数', '涨幅中位', '到峰值中位', '峰值→谷底中位'))
all_events = {}
for THR in (0.5, 1.0, 2.0, 3.0):
    P = gain >= THR
    ev = []      # (col, start_row, peak_row, gain_at_peak)
    for j in range(N):
        col = P[:, j]
        if not col.any():
            continue
        starts = np.where(col & ~np.r_[False, col[:-1]])[0]
        last = -10 ** 9
        for i in starts:
            if i - last < W5:
                continue
            seg = HV[i:i + W5, j]
            pk = i + int(np.nanargmax(seg))
            ev.append((j, int(i), int(pk), float(CV[pk, j] / CV[i, j] - 1.0)))
            last = int(i)
    all_events[THR] = ev
    if not ev:
        say('  %-8s %9d' % ('+%d%%' % (THR * 100), 0)); continue
    gains = np.array([e[3] for e in ev])
    tot = np.array([(e[2] - e[1]) * 5 for e in ev])       # 分钟
    # 峰值后到最低
    dw = []
    for (j, i, pk, g) in ev:
        end = min(T, pk + H72)
        if end - pk < 10:
            continue
        seg = LV[pk + 1:end, j]      # 从峰值之后起算（含峰值那根会把"暴涨前的地板"误当成回撤）
        if len(seg) == 0:
            continue
        k = int(np.nanargmin(seg))
        dw.append((k * 5, float(seg[k] / HV[pk, j] - 1.0)))
    dw_t = np.array([d[0] for d in dw]); dw_d = np.array([d[1] for d in dw])
    say('  %-8s %9d %10d %11.1f%% %11.1f 分 %11.1f 分'
        % ('+%d%%' % (THR * 100), len(ev), len(set(e[0] for e in ev)),
           100 * np.median(gains), np.median(tot), np.median(dw_t)))
    say('        到峰值耗时 P25/50/75 = %.0f / %.0f / %.0f 分；窗口起点→峰值 位置分布'
        % tuple(np.percentile(tot, [25, 50, 75])))
    say('        峰值→谷底耗时 P25/50/75 = %.0f / %.0f / %.0f 分（72h 内最低）'
        % tuple(np.percentile(dw_t, [25, 50, 75])))
    say('         回撤深度：中位 %.1f%%  P25 %.1f%%  超70%%占比 %.1f%%（n=%d）'
        % (100 * np.median(dw_d), 100 * np.percentile(dw_d, 25), 100 * (dw_d <= -0.7).mean(), len(dw_d)))

# ---- 用 +100% 档做特征对照 ----
ev = all_events[1.0]
say('\n' + '=' * 100)
say('【二】特征对照：暴涨启动前 vs 全样本（+100%% 档，n=%d 事件）' % len(ev))
say('=' * 100)
rand = np.random.default_rng(0)
sample = [(int(rand.integers(0, T)), int(rand.integers(0, N))) for _ in range(max(len(ev) * 120, 2000))]
summarize('全样本（随机）', sample)
pre5 = [(max(0, i - 60), j) for (j, i, pk, g) in ev]
summarize('暴涨前 60 根(5h)', pre5)
pre1 = [(max(0, i - 12), j) for (j, i, pk, g) in ev]
summarize('暴涨前 12 根(1h)', pre1)
pk_rows = [(pk, j) for (j, i, pk, g) in ev]
summarize('峰值处', pk_rows)

say('\n' + '=' * 100)
say('【三】暴跌前信号：峰值前 12 根 / 峰值前 3 根 vs 全样本')
say('=' * 100)
summarize('峰值前 12 根(1h)', [(max(0, pk - 12), j) for (j, i, pk, g) in ev])
summarize('峰值前 3 根(15m)', [(max(0, pk - 3), j) for (j, i, pk, g) in ev])

# 存事件表
rows = []
for THR, evs in all_events.items():
    for (j, i, pk, g) in evs:
        rows.append(dict(thr=THR, symbol=syms[j],
                         start_time=pd.to_datetime(int(tm[i]), unit='ms', utc=True),
                         peak_time=pd.to_datetime(int(tm[pk]), unit='ms', utc=True),
                         start_close=float(CV[i, j]), peak_high=float(HV[pk, j]),
                         gain=g, min_to_peak=(pk - i) * 5))
pd.DataFrame(rows).to_csv(os.path.join(EV, 'pump_hunt_events.csv'), index=False, encoding='utf-8')
say('\nsaved: %s  （%d 条事件）' % (os.path.join(EV, 'pump_hunt_events.csv'), len(rows)))

open(os.path.join(EV, 'pump_hunt.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('TOTAL %.0fs' % (time.time() - t0))
