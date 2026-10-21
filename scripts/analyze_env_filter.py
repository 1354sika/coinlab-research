# -*- coding: utf-8 -*-
"""降噪 + 环境条件化：把「噪声」拆开，看剩下的关系在哪

用户口径：噪声太大，那就分析环境 + 加算子/滤波，把更深层的关系拔出来。

本脚本三步：
  1) 噪声诊断：方差比 VR(k)（<1 均值回归=噪声主导，>1 趋势=信号主导）+ 一阶自相关
  2) 环境划分：交易时段（亚洲/欧洲/美国，按 UTC）+ 全市场流动性（高/低）
  3) 条件化传导：在每种环境下重测「这一分钟 -> 之后 k 分钟」的方向性 lift
     （同一冲击定义：|超额收益| > 99 分位）
输出：evidence/env_filter.txt / .png
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
KS = [1, 3, 5]
Q = 0.99
MIN_EV = 60
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "close", "quote_volume"])
px = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
qv = df.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
ret = px.pct_change().iloc[1:]
qv = qv.iloc[1:]
syms = list(px.columns)
V = ret.sub(ret.mean(axis=1), axis=0).to_numpy(dtype=np.float32)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

# ---------- 1) 噪声诊断 ----------
say("")
say("=== 1) 噪声诊断：方差比 VR(k) = Var(k分钟) / [k x Var(1分钟)] ===")
say("     VR<1 = 均值回归（价差跳动/噪声主导）；VR>1 = 趋势（信号主导）；VR=1 = 随机游走")
o1 = np.nanvar(V, axis=0)
vr_rows = []
for k in (2, 5, 10, 30):
    acc = np.zeros_like(V)
    for i in range(k):
        acc[:T - i if i else T] += V[i:]
    vr = np.nanvar(acc, axis=0) / (k * o1)
    vr_rows.append(vr)
    say("  VR(%2d): 中位 %.3f  均值 %.3f  >1 的币数 %d/%d"
        % (k, np.nanmedian(vr), np.nanmean(vr), int(np.nansum(vr > 1)), N))

say("")
say("=== 1b) 一阶自相关（应≈0 或负；显著负 = 买卖价差跳动噪声）===")
a1 = []
for j in range(N):
    c = V[:, j]
    a1.append(np.corrcoef(c[:-1], c[1:])[0, 1])
a1 = np.array(a1)
say("  AC(1): 中位 %+.4f  均值 %+.4f  显著为负(< -0.02)的币数 %d/%d"
    % (np.nanmedian(a1), np.nanmean(a1), int((a1 < -0.02).sum()), N))

# ---------- 2) 环境划分 ----------
hour = pd.to_datetime(px.index, unit="ms").hour.to_numpy()
totv = np.nansum(qv.to_numpy(), axis=1)
liq_hi = totv > np.nanquantile(totv, 0.7)

envs = {
    "亚洲时段(UTC0-8)": (hour >= 0) & (hour < 8),
    "欧洲时段(UTC8-16)": (hour >= 8) & (hour < 16),
    "美国时段(UTC16-24)": (hour >= 16) & (hour < 24),
    "高流动性(市场成交额>70%)": liq_hi,
    "低流动性(<=70%)": ~liq_hi,
}
say("")
say("=== 2) 环境划分 ===")
for k, m in envs.items():
    say("  %-24s 分钟数 %d (%.0f%%)" % (k, int(m.sum()), 100 * m.mean()))


# ---------- 3) 条件化传导 ----------
def scan_env(mask, k):
    """返回 (lifts 矩阵 N x N, 各源有效冲击数)"""
    lifts = np.full((N, N), np.nan, dtype=np.float32)
    nev = np.zeros(N, dtype=int)
    idx_all = np.where(mask)[0]
    idx_all = idx_all[idx_all < T - k]
    if len(idx_all) < MIN_EV * 2:
        return lifts, nev
    sub = V[idx_all]
    for j in range(N):
        col = sub[:, j]
        fin = np.isfinite(col)
        if fin.sum() < len(idx_all) * 0.8:
            continue
        thr = np.quantile(col[fin], Q)
        m = fin & (col > thr)
        if m.sum() < MIN_EV:
            continue
        # 响应：原矩阵上 idx_all[m]+k 行
        resp_rows = idx_all[m] + k
        lifts[j, :] = np.nanmean(V[resp_rows, :], axis=0) - np.nanmean(V, axis=0)
        nev[j] = int(m.sum())
    return lifts, nev


say("")
say("=== 3) 条件化传导：各环境下「这一分钟 -> 之后 k 分钟」方向性 lift ===")
say("%-24s %5s %10s %10s %10s %8s" % ("环境", "滞后", "lift中位%", "lift均值%", "P99%", "有效源"))
store = {}
for ename, mask in envs.items():
    for k in KS:
        lifts, nev = scan_env(mask, k)
        ok = nev >= MIN_EV
        if ok.sum() == 0:
            say("%-24s %5dm   (有效源不足)" % (ename, k))
            continue
        sub = lifts[ok, :]
        s = sub[np.isfinite(sub)]
        store[(ename, k)] = (sub, s)
        say("%-24s %5dm %10.4f %10.4f %10.4f %8d"
            % (ename, k, np.median(s) * 100, np.mean(s) * 100,
               np.percentile(s, 99) * 100, int(ok.sum())))

# 每个环境的最强对
say("")
say("=== 各环境下最强的 5 对（滞后 1 分钟）===")
for ename, mask in envs.items():
    key = (ename, 1)
    if key not in store:
        continue
    sub, _ = store[key]
    flat = []
    for j in range(sub.shape[0]):
        for i2 in range(sub.shape[1]):
            if i2 != j and np.isfinite(sub[j, i2]):
                flat.append((abs(sub[j, i2]), j, i2, sub[j, i2]))
    flat.sort(reverse=True)
    say("  [%s]" % ename)
    for v, j, i2, obs in flat[:5]:
        say("    %-13s -> %-13s %+8.4f%%" % (syms[j], syms[i2], obs * 100))

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
vv = [np.nanmedian(r) for r in vr_rows]
ax.plot([2, 5, 10, 30], vv, "o-", color="0.25")
ax.axhline(1.0, color="0.7", ls="--")
ax.set_xscale("log")
ax.set_xlabel("k（分钟）")
ax.set_ylabel("VR(k) 中位")
ax.set_title("方差比：<1 表示噪声/均值回归主导", fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
labels, meds, p99s = [], [], []
for ename in envs:
    key = (ename, 1)
    if key not in store:
        continue
    _, s = store[key]
    labels.append(ename.split("(")[0])
    meds.append(np.median(s) * 100)
    p99s.append(np.percentile(s, 99) * 100)
x = np.arange(len(labels))
ax.bar(x - 0.2, meds, 0.4, label="中位", color="0.6")
ax.bar(x + 0.2, p99s, 0.4, label="P99", color="0.3")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, fontsize=8)
ax.axhline(0.2, color="0.5", ls=":", label="往返手续费 0.2%")
ax.set_ylabel("%")
ax.set_title("各环境 lift（滞后1分钟）vs 手续费", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.hist(a1, bins=60, color="0.55")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("一阶自相关分布（个币）", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "env_filter.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "env_filter.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
