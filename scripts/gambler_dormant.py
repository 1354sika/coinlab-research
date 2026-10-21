# -*- coding: utf-8 -*-
"""gambler · 早期识别之二 —— "吸筹"的 K 线可观测代理能否预测启动？

用户指出（2026-10-16）："确认崩只在局内有用的，问题是还没做到**早期识别操盘手**"
用户早就提示过："**长期不动**的币也是重要观察"

设计文档《操盘手对手建模》里，四阶段的第一阶段"吸筹"的可观测指纹是
    "放量但价格不动"（depth/逐笔层面的假买盘）；
本实验只用 K 线做它的**代理**，检验能不能提前认出来：

  条件 A 长期不动 + 突然放量   ：最近 288 根(24h)振幅 <3%，且当前量比 ≥5
  条件 B 放量滞涨（吸筹痕迹）  ：量比 ≥5 但近 1h 涨幅 |ret12| ≤1%
  条件 C 极长冻结 + 巨量       ：连续冻结段长 ≥288 根(24h)，且量比 ≥10

评估：命中后 **未来 24h / 72h** 的最大涨幅、涨≥20% / ≥50% 的概率、期望收益，
      与**全市场基准**对照（去重：同币 288 根内只取一次）
"""
import os, gc, time
import numpy as np
import pandas as pd

DATA = r'D:\中信证券\coinlab\data'
EV = r'D:\中信证券\coinlab\evidence'
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
say('=' * 100)
say('gambler 早期识别之二：长期不动 + 放量滞涨（吸筹的 K 线代理）→ 能否预测启动')
say('=' * 100)

df = pd.read_csv(os.path.join(DATA, 'klines_5m_502.csv'),
                 usecols=['symbol', 'open_time', 'close', 'high', 'low', 'quote_volume'],
                 dtype={'symbol': 'category', 'open_time': 'int64'})
df = df.sort_values(['symbol', 'open_time'])
def _piv(c):
    m = df.pivot(index='open_time', columns='symbol', values=c).astype('float32')
    return m.where(m > 0).ffill(limit=3)
close, high, low = _piv('close'), _piv('high'), _piv('low')
qv = df.pivot(index='open_time', columns='symbol', values='quote_volume').astype('float32')
qv = qv.where(qv >= 0)
del df; gc.collect()
T, N = close.shape
tm = close.index.values
syms = list(close.columns)
CV, HV, LV, QV = close.values, high.values, low.values, qv.values
say('matrix %d x %d  %.0fs' % (T, N, time.time() - t0))

ret12 = np.full((T, N), np.nan, np.float32); ret12[12:] = CV[12:] / CV[:-12] - 1.0
W = 288
hi288 = pd.DataFrame(HV).rolling(W, min_periods=W // 4).max().values
lo288 = pd.DataFrame(LV).rolling(W, min_periods=W // 4).min().values
amp288 = hi288 / lo288 - 1.0                                    # 24h 振幅
med60 = pd.DataFrame(QV).rolling(60, min_periods=20).median().values
volr = QV / np.where(med60 > 0, med60, np.nan)
# 连续冻结段长（价格完全不变）
d7 = np.zeros((T, N), np.float32)
idxr = np.arange(T)
for j in range(N):
    fz = np.r_[False, CV[1:, j] == CV[:-1, j]]
    lf = np.maximum.accumulate(np.where(~fz, idxr, -1))
    d7[:, j] = np.where(fz, idxr - lf, 0)
# 未来 24h / 72h 的最高价
fmax24 = pd.DataFrame(HV).iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1).values
fmax72 = pd.DataFrame(HV).iloc[::-1].rolling(72, min_periods=1).max().iloc[::-1].shift(-1).values
say('指标算完  %.0fs' % (time.time() - t0))

conds = {
    'A 长期不动(24h振幅<3%) + 放量(量比≥5)': (amp288 < 0.03) & (volr >= 5),
    'B 放量滞涨(量比≥5 且 |1h涨幅|≤1%)': (volr >= 5) & (np.abs(ret12) <= 0.01),
    'C 冻结≥24h + 巨量(量比≥10)': (d7 >= 288) & (volr >= 10),
    'D 冻结≥4h + 放量(量比≥3)': (d7 >= 48) & (volr >= 3),
}
rng = np.random.default_rng(0)
base_idx = [(int(rng.integers(300, T - 73)), int(rng.integers(0, N))) for _ in range(30000)]


def stats(pairs, tag):
    if len(pairs) < 20:
        say('  %-40s 样本不足(%d)' % (tag, len(pairs))); return None
    rr = np.array([p[0] for p in pairs]); cc = np.array([p[1] for p in pairs])
    c0 = CV[rr, cc]
    g24 = fmax24[rr, cc] / c0 - 1.0
    g72 = fmax72[rr, cc] / c0 - 1.0
    r24 = np.where(np.isfinite(CV[np.minimum(rr + 24, T - 1), cc]),
                   CV[np.minimum(rr + 24, T - 1), cc] / c0 - 1.0, np.nan)
    say('  %-40s n=%6d  未来24h最高涨幅中位 %+6.2f%%  P(≥20%%)=%5.2f%%  P(≥50%%)=%5.2f%%  未来24h实际收益中位 %+6.2f%%'
        % (tag, len(pairs), 100 * np.nanmedian(g24), 100 * np.nanmean(g24 >= 0.20),
           100 * np.nanmean(g24 >= 0.50), 100 * np.nanmedian(r24)))
    return dict(n=len(pairs), g24=np.nanmedian(g24), p20=np.nanmean(g24 >= 0.20),
                p50=np.nanmean(g24 >= 0.50), r24=np.nanmedian(r24))


say('\n' + '=' * 100)
say('【一】吸筹代理条件 → 未来涨幅（去重：同币 288 根内只取一次）')
say('=' * 100)
stats(base_idx, '全市场随机基准')
RES = {}
for tag, m in conds.items():
    pairs = []
    for j in range(N):
        idx = np.where(m[:, j])[0]
        idx = idx[idx < T - 73]
        if len(idx) == 0:
            continue
        last = -10 ** 9
        for t in idx:
            if t - last < W:
                continue
            pairs.append((int(t), j)); last = int(t)
    RES[tag] = stats(pairs, tag)

say('\n' + '=' * 100)
say('【二】条件是否真的偏"启动前"？—— 条件命中点相对"暴涨事件起点"的时间分布')
say('=' * 100)
fmax4h = pd.DataFrame(HV).iloc[::-1].rolling(48, min_periods=1).max().iloc[::-1].shift(-1).values
gainW = fmax4h / CV - 1.0
Pp = gainW >= 0.5
ev = []
for j in range(N):
    col = Pp[:, j]
    if not col.any():
        continue
    starts = np.where(col & ~np.r_[False, col[:-1]])[0]
    last = -10 ** 9
    for i in starts:
        if i - last < 48:
            continue
        ev.append((j, int(i))); last = int(i)
say('  暴涨事件（4h≥+50%）= %d 个' % len(ev))
for tag, m in conds.items():
    hits = 0; near = 0; tot = 0
    for (j, i) in ev:
        lo = max(0, i - 288)
        tot += 1
        if m[i, j]:
            hits += 1
        if m[lo:i + 1, j].any():
            near += 1
    say('  %-40s 命中起点本身 %5.1f%% ；起点前 24h 内至少命中一次 %5.1f%%（共 %d 个事件）'
        % (tag, 100 * hits / max(tot, 1), 100 * near / max(tot, 1), tot))

open(os.path.join(EV, 'gambler_dormant.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
