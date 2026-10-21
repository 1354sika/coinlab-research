# -*- coding: utf-8 -*-
"""滤波算子实验：噪声来自 bid-ask bounce，那就别再用收盘价

背景（已实测）：分钟收益 AC(1) 中位 -0.067，VR(30)=0.803 -> 典型的买卖价差跳动噪声。
本脚本上三种滤波算子，判据两条：
  (1) 噪声指标是否改善：AC(1) 是否趋近 0、VR 是否趋近 1
  (2) 信号是否变强：方向性传导 lift / 波动率放大倍数 是否变大
滤波算子：
  F0 基线   ：close-to-close（1 分钟）
  F1 中间价 ：(high + low) / 2 作为价格（对 bid-ask bounce 最对症）
  F2 三分钟 ：3 分钟聚合收益
  F3 高额分钟：只用 quote_volume 位于前 30% 的分钟
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口）
输出：evidence/filter_ops.txt / .png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
Q, MIN_EV = 0.99, 60
KS = [1, 3, 5]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


cols = ["symbol", "open_time", "high", "low", "close", "quote_volume"]
df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=cols)
def pv(c):
    return df.pivot(index="open_time", columns="symbol", values=c).sort_index()

C = pv("close")
H = pv("high")
L = pv("low")
QV = pv("quote_volume")
syms = list(C.columns)
say("矩阵 %d 分钟 x %d 币" % C.shape)


def to_ex(price):
    r = price.pct_change().iloc[1:]
    return r.sub(r.mean(axis=1), axis=0).to_numpy(dtype=np.float32)


def noise_stats(V, name):
    T, N = V.shape
    o1 = np.nanvar(V, axis=0)
    vr = {}
    for k in (2, 5, 30):
        acc = np.zeros_like(V)
        for i in range(k):
            acc[:T - i if i else T] += V[i:]
        vr[k] = np.nanmedian(np.nanvar(acc, axis=0) / (k * o1))
    ac = []
    for j in range(N):
        c = V[:, j]
        v = np.isfinite(c[:-1]) & np.isfinite(c[1:])
        if v.sum() > 100:
            ac.append(np.corrcoef(c[:-1][v], c[1:][v])[0, 1])
    say("  %-14s AC(1)中位 %+.4f   VR2 %.3f  VR5 %.3f  VR30 %.3f"
        % (name, np.nanmedian(ac), vr[2], vr[5], vr[30]))
    return float(np.nanmedian(ac)), vr


say("")
say("=== 1) 噪声指标：滤波后是否改善（AC(1) 趋 0 越好；VR 趋 1 越好）===")
filters = {}
filters["F0 收盘价"], _ = None, None
stats = {}
for name, price in [("F0 收盘价", C), ("F1 (高+低)/2", (H + L) / 2.0)]:
    V = to_ex(price)
    filters[name] = V
    stats[name] = noise_stats(V, name)

# F2 三分钟
r3 = C.pct_change(3).iloc[3:]
V3 = r3.sub(r3.mean(axis=1), axis=0).to_numpy(dtype=np.float32)
filters["F2 三分钟"] = V3
stats["F2 三分钟"] = noise_stats(V3, "F2 三分钟")

# F3 高额分钟
V0 = filters["F0 收盘价"]
Vqh = (H.to_numpy() + L.to_numpy()) / 2.0
qv = QV.to_numpy()
totv = np.nansum(qv, axis=1)
mask_hi = totv > np.nanquantile(totv, 0.7)
say("  （F3 高额分钟：占 %.0f%% 的分钟）" % (100 * mask_hi.mean()))


def transmission(V, mask=None, k=1):
    """返回 lift 分布 + 波动放大倍数中位（同一次冲击定义）"""
    T, N = V.shape
    idx_pool = np.where(mask)[0] if mask is not None else np.arange(T)
    idx_pool = idx_pool[idx_pool < T - k]
    if len(idx_pool) < MIN_EV * 2:
        return None, None
    A = np.abs(V)
    baseA = np.nanmean(A, axis=0)
    baseV = np.nanmean(V, axis=0)
    lifts, mags = [], []
    sub = V[idx_pool]
    absub = A[idx_pool]
    for j in range(N):
        col = absub[:, j]
        fin = np.isfinite(col)
        if fin.sum() < len(idx_pool) * 0.8:
            continue
        thr = np.quantile(col[fin], Q)
        m = fin & (col > thr)
        if m.sum() < MIN_EV:
            continue
        rows = idx_pool[m] + k
        lifts.append(np.nanmean(V[rows, :], axis=0) - baseV)
        mags.append((np.nanmean(A[rows, :], axis=0)) / np.where(baseA > 0, baseA, np.nan))
    if not lifts:
        return None, None
    return (np.array(lifts), np.array(mags))


say("")
say("=== 2) 信号：各滤波下的方向性 lift 与波动放大倍数 ===")
say("%-14s %5s %10s %10s %10s %12s %10s"
    % ("滤波", "滞后", "lift中位%", "lift均值%", "P99%", "波动放大中位", "放大>1.1对%"))
out = {}
for name, V in filters.items():
    for k in KS:
        lf, mg = transmission(V, None, k)
        if lf is None:
            continue
        s = lf[np.isfinite(lf)]
        m = mg[np.isfinite(mg)]
        out[(name, k)] = (s, m)
        say("%-14s %4dm %10.4f %10.4f %10.4f %12.3f %9.0f%%"
            % (name, k, np.median(s) * 100, np.mean(s) * 100, np.percentile(s, 99) * 100,
               np.median(m), 100 * (m > 1.1).mean()))
    # F3 高额分钟（只对 F0 与 F1 价格做）
    if name.startswith("F3"):
        continue
    lf, mg = transmission(V, mask_hi, 1)
    if lf is not None:
        s = lf[np.isfinite(lf)]
        m = mg[np.isfinite(mg)]
        out[(name + " + 高额分钟", 1)] = (s, m)
        say("%-14s %4dm %10.4f %10.4f %10.4f %12.3f %9.0f%%"
            % (name + " +高额", 1, np.median(s) * 100, np.mean(s) * 100,
               np.percentile(s, 99) * 100, np.median(m), 100 * (m > 1.1).mean()))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for name in ["F0 收盘价", "F1 (高+低)/2", "F2 三分钟"]:
    if (name, 1) in out:
        s, _ = out[(name, 1)]
        ax.hist(s * 100, bins=120, histtype="step", label="%s +1m" % name)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("方向性 lift 分布（各滤波）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
bins = [k for k in out if k[1] == 1]
lbl = [k[0] for k in bins]
med = [np.median(out[k][0]) * 100 for k in bins]
p99 = [np.percentile(out[k][0], 99) * 100 for k in bins]
x = np.arange(len(lbl))
ax.bar(x - 0.2, med, 0.4, label="中位", color="0.65")
ax.bar(x + 0.2, p99, 0.4, label="P99", color="0.3")
ax.set_xticks(x)
ax.set_xticklabels(lbl, rotation=25, fontsize=8, ha="right")
ax.axhline(0.2, color="0.5", ls=":", label="手续费 0.2%")
ax.set_title("各滤波的 lift（滞后1分钟）vs 手续费", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
p1 = os.path.join(EV, "filter_ops.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "filter_ops.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
