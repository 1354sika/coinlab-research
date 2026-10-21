# -*- coding: utf-8 -*-
"""把风控发现融入预测：用「波动状态」做门控，分区检验预测能力

用户口径：不是把风控单独做模块，而是**把风控发现融进预测**。
已知发现：传导窗口（高波动）覆盖 20.7% 时间，其内波动 2.48 倍、尾部 16 倍。
要回答：预测能力（时序动量 / 横截面传导）是否**依赖状态**？若是，就用状态做门控算子。

做法（全部无前视）：
  1) 用**过去 30 分钟**的市场滚动波动，分成 低/中/高 三个状态
  2) 在每个状态内分别测：
     a) 时序动量 AC(1)（滤波版：用 (高+低)/2）
     b) 动量策略：过去 k 分钟强 -> 未来 k 分钟，扣往返费 0.2%
     c) 横截面传导 lift（谁带谁）
  3) 找出「只在某状态有效」的预测 —— 那才是能落地的门控算子
输出：evidence/regime_gating.txt / .png
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
FEE = 0.002
REG = ["低波动", "中波动", "高波动"]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


cols = ["symbol", "open_time", "high", "low", "close"]
df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=cols)


def pv(c):
    return df.pivot(index="open_time", columns="symbol", values=c).sort_index()


C = pv("close")
MID = (pv("high") + pv("low")) / 2.0
r = C.pct_change().iloc[1:]
rm = MID.pct_change().iloc[1:]
syms = list(C.columns)
V = r.to_numpy(dtype=np.float32)
VM = rm.to_numpy(dtype=np.float32)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

# --- 状态：过去 30 分钟的市场滚动波动（无前视）---
mkt = np.nanmean(V, axis=1)
vol30 = pd.Series(mkt).rolling(30).std().to_numpy()
q33, q67 = np.nanquantile(vol30, [0.33, 0.67])
reg = np.where(vol30 < q33, 0, np.where(vol30 < q67, 1, 2))
say("波动状态阈值：低 < %.4f%% <= 中 < %.4f%% <= 高" % (q33 * 100, q67 * 100))
for k in range(3):
    say("  %s: %d 分钟（%.0f%%）" % (REG[k], int((reg == k).sum()), 100 * (reg == k).mean()))

# --- a) 时序动量 AC(1) 分状态 ---
say("")
say("=== a) 时序动量 AC(1)（(高+低)/2 口径）分状态 ===")
for k in range(3):
    idx = np.where(reg == k)[0]
    idx = idx[idx < T - 1]
    ac = []
    for j in range(N):
        a = VM[idx, j]
        b = VM[idx + 1, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 200:
            ac.append(np.corrcoef(a[m], b[m])[0, 1])
    ac = np.array(ac)
    say("  %s: AC(1) 中位 %+.4f  均值 %+.4f  >0 的币数 %d/%d"
        % (REG[k], np.nanmedian(ac), np.nanmean(ac), int((ac > 0).sum()), len(ac)))

# --- b) 动量策略分状态（扣费）---
say("")
say("=== b) 横截面动量策略（过去 k 强 -> 持有 k 分钟），扣往返 %.1f%% ===" % (FEE * 100))
say("%-8s %5s %10s %12s %12s %10s" % ("状态", "k", "净收益%", "毛收益%", "换手次数", "每笔均值%"))
for kidx, kk in enumerate([1, 3, 5]):
    for k in range(3):
        idx = np.where(reg == k)[0]
        idx = idx[idx < T - kk - 1]
        if len(idx) < 100:
            continue
        nav, ntr, gross_sum = 1.0, 0, 0.0
        prev = np.zeros(N, dtype=np.float32)
        for t in idx:
            past = np.nanmean(V[t - kk + 1:t + 1], axis=0)
            order = np.argsort(-np.nan_to_num(past))
            top = order[:20]
            w = np.zeros(N, dtype=np.float32)
            w[top] = 1.0 / len(top)
            turn = np.nansum(np.abs(w - prev))
            gross = np.nansum(w * np.nan_to_num(V[t + 1]))
            nav *= (1.0 + gross - turn * FEE / 2)
            gross_sum += gross
            ntr += 1
            prev = w
        say("%-8s %5d %9.3f%% %11.3f%% %12d %10.4f%%"
            % (REG[k], kk, (nav - 1) * 100, gross_sum * 100 / max(ntr, 1),
               ntr, (nav - 1) * 100 / max(ntr, 1)))

# --- c) 截面传导 lift 分状态 ---
say("")
say("=== c) 截面传导 lift（某币强 -> 其他币下一分钟）分状态 ===")
say("%-8s %10s %10s %10s" % ("状态", "lift中位%", "P99%", "有效源"))
Q = 0.99
for k in range(3):
    idx = np.where(reg == k)[0]
    idx = idx[idx < T - 1]
    if len(idx) < 500:
        continue
    base = np.nanmean(V, axis=0)
    lifts, nsrc = [], 0
    for j in range(N):
        col = V[idx, j]
        fin = np.isfinite(col)
        if fin.sum() < len(idx) * 0.8:
            continue
        thr = np.quantile(col[fin], Q)
        m = fin & (col > thr)
        if m.sum() < 30:
            continue
        rows = idx[m] + 1
        lifts.append(np.nanmean(V[rows, :], axis=0) - base)
        nsrc += 1
    if not lifts:
        continue
    L = np.array(lifts)
    s = L[np.isfinite(L)]
    say("%-8s %10.4f %10.4f %10d" % (REG[k], np.median(s) * 100, np.percentile(s, 99) * 100, nsrc))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
ax.plot(np.arange(T), vol30 * 100, lw=0.5, color="0.5")
ax.axhline(q33 * 100, color="0.3", ls="--", lw=1)
ax.axhline(q67 * 100, color="0.3", ls=":", lw=1)
ax.set_title("市场滚动波动（30分钟）与状态阈值", fontsize=10)
ax.set_ylabel("std %")
ax.grid(alpha=0.3)
ax = axes[1]
w = 0.35
xs = np.arange(3)
meds = []
for k in range(3):
    idx = np.where(reg == k)[0]
    idx = idx[idx < T - 1]
    ac = []
    for j in range(min(N, 100)):
        a = VM[idx, j]
        b = VM[idx + 1, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 200:
            ac.append(np.corrcoef(a[m], b[m])[0, 1])
    meds.append(np.nanmedian(ac))
ax.bar(xs, meds, 0.5, color="0.4")
ax.set_xticks(xs)
ax.set_xticklabels(REG)
ax.axhline(0, color="0.2", lw=1)
ax.set_title("时序动量 AC(1) 分状态", fontsize=10)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
p1 = os.path.join(EV, "regime_gating.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "regime_gating.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
