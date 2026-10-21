# -*- coding: utf-8 -*-
"""分钟尺度「这一分钟 -> 之后几分钟」的第二问：波动率传导（而不是收益率传导）

为什么补：收益率传导已测（lift 中位 ~0，最强对 0.077% < 手续费 0.2%，方向性上不可交易）。
          但日线层发现的「下尾共现比 2.09」提示：真正同步的是**波动**不是**方向**。
          所以本版问：这一分钟某币剧烈波动，之后 1/2/3/5/10 分钟，
          其他币的**波动幅度**是否上升？
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口）
输出：evidence/cascade_1m_vol.txt / .png
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
KS = [1, 2, 3, 5, 10]
Q = 0.99
MIN_EV = 100
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=["symbol", "open_time", "close"])
px = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change().iloc[1:]
syms = list(px.columns)
V = ret.sub(ret.mean(axis=1), axis=0).to_numpy(dtype=np.float32)
A = np.abs(V)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))
say("|分钟超额收益| 中位 %.4f%%；P99 %.4f%%" % (np.nanmedian(A) * 100, np.nanpercentile(A, 99) * 100))

baseA = np.nanmean(A, axis=0)
baseV = np.nanmean(V, axis=0)

say("")
say("=== 波动率传导：|收益| 冲击 -> 之后 k 分钟的 |收益| ===")
say("%-6s %10s %10s %10s %10s" % ("滞后", "lift中位%", "lift均值%", "P90%", "P99%"))
vol_res = {}
for k in KS:
    lifts = np.full((N, N), np.nan, dtype=np.float32)
    for j in range(N):
        col = A[:, j]
        thr = np.nanquantile(col, Q)
        idx = np.where(col > thr)[0]
        idx = idx[idx < T - k]
        if len(idx) < MIN_EV:
            continue
        lifts[j, :] = np.nanmean(A[idx + k, :], axis=0) - baseA
    vol_res[k] = lifts
    sub = lifts[np.isfinite(lifts)]
    say("%5dm %10.4f %10.4f %10.4f %10.4f"
        % (k, np.median(sub) * 100, np.mean(sub) * 100,
           np.percentile(sub, 90) * 100, np.percentile(sub, 99) * 100))

say("")
say("=== 对照：方向性传导（同数据、同冲击定义、看的是收益而不是|收益|）===")
say("%-6s %10s %10s %10s %10s" % ("滞后", "lift中位%", "lift均值%", "P90%", "P99%"))
dir_res = {}
for k in KS:
    lifts = np.full((N, N), np.nan, dtype=np.float32)
    for j in range(N):
        col = A[:, j]
        thr = np.nanquantile(col, Q)
        idx = np.where(col > thr)[0]
        idx = idx[idx < T - k]
        if len(idx) < MIN_EV:
            continue
        lifts[j, :] = np.nanmean(V[idx + k, :], axis=0) - baseV
    dir_res[k] = lifts
    sub = lifts[np.isfinite(lifts)]
    say("%5dm %10.4f %10.4f %10.4f %10.4f"
        % (k, np.median(sub) * 100, np.mean(sub) * 100,
           np.percentile(sub, 90) * 100, np.percentile(sub, 99) * 100))

# 波动率放大倍数（更有意义的口径）：条件|收益| / 无条件|收益|
say("")
say("=== 波动放大倍数 = 条件下 |收益| / 无条件 |收益|（1.0 = 无传导）===")
say("%-6s %12s %12s %12s" % ("滞后", "中位倍数", "均值倍数", "有多少对>1.1"))
for k in KS:
    lifts = vol_res[k]
    ratios = (lifts + baseA) / np.where(baseA > 0, baseA, np.nan)
    r = ratios[np.isfinite(ratios)]
    say("%5dm %12.3f %12.3f %12d" % (k, np.median(r), np.mean(r), int(np.nansum(r > 1.1))))

# 最强对
say("")
say("=== 最强的波动率传导对（滞后 1 分钟）===")
l1 = vol_res[1]
flat = []
for j in range(N):
    for i2 in range(N):
        if i2 != j and np.isfinite(l1[j, i2]):
            flat.append((abs(l1[j, i2]), j, i2))
flat.sort(reverse=True)
for v, j, i2 in flat[:12]:
    say("  %-13s -> %-13s  |收益| 提升 %+.4f 百分点（相对 %.2f 倍）"
        % (syms[j], syms[i2], l1[j, i2] * 100, (l1[j, i2] + baseA[i2]) / baseA[i2]))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
for k in KS:
    sub = vol_res[k][np.isfinite(vol_res[k])]
    ax.hist(sub * 100, bins=100, histtype="step", label="%dm" % k)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("波动率传导 lift 分布（百分点）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for k in KS:
    sub = dir_res[k][np.isfinite(dir_res[k])]
    ax.hist(sub * 100, bins=100, histtype="step", label="%dm" % k)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("方向性传导 lift 分布（百分点）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[2]
med = [np.median(((vol_res[k] + baseA) / baseA)[np.isfinite(vol_res[k])]) for k in KS]
ax.plot(KS, med, "o-", color="0.25")
ax.axhline(1.0, color="0.7", ls="--")
ax.set_xlabel("滞后（分钟）")
ax.set_ylabel("波动放大倍数")
ax.set_title("波动放大倍数 vs 滞后", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "cascade_1m_vol.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cascade_1m_vol.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
