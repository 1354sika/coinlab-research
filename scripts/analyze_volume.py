# -*- coding: utf-8 -*-
"""「更深的关系」下一个候选：成交量传导（量在价先？）

已有的价格类检验全部为负（线性/秩/条件/尾部/多滞后/时变/路径/环境/滤波/门控）。
本脚本换一个**输入量**：不再看价格带动价格，而看**成交量**是否领先价格——
成交量的变化通常先于价格（谁在买、买多少，比价格本身更早暴露意图）。
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口，含 quote_volume）
检验：
  A) 自量传导：某币放量 -> 它自己之后 k 分钟的价格收益/波动
  B) 跨币量传导：某币放量 -> 其他币之后 k 分钟的价格收益/波动
  C) 量-价领先：某币放量但价格未动 -> 之后价格是否跟随
  D) 对照：某币价格剧烈波动（原口径）-> 之后价格
输出：evidence/volume_lead.txt / .png
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
KS = [1, 3, 5, 15]
LBN = 60
Q = 0.99
MIN_EV = 100
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "close", "quote_volume"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
Qv = df.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
V = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
QV = Qv.iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

# 相对放量：当前分钟成交额 / 过去 60 分钟中位
med = pd.DataFrame(QV).rolling(LBN).median().to_numpy(dtype=np.float32)
RVOL = QV / np.where(np.isfinite(med) & (med > 0), med, np.nan)
say("相对放量 RVOL 定义：本分钟成交额 / 过去 %d 分钟中位" % LBN)
say("  RVOL 中位 %.2f；P99 %.2f；最大 %.1f" % (np.nanmedian(RVOL), np.nanpercentile(RVOL, 99), np.nanmax(RVOL)))

base_v = np.nanmean(V, axis=0)
base_a = np.nanmean(np.abs(V), axis=0)


def scan(sig_mat, k, sel="hi"):
    """sig_mat: (T,N) 触发矩阵；返回 (lift收益, lift波动, 有效源数)"""
    res, volres, nsrc = [], [], 0
    for j in range(N):
        col = sig_mat[:, j]
        fin = np.isfinite(col)
        if fin.sum() < T * 0.6:
            continue
        thr = np.nanquantile(col[fin], Q)
        m = fin & (col > thr)
        idx = np.where(m)[0]
        idx = idx[idx < T - k]
        if len(idx) < MIN_EV:
            continue
        rows = idx + k
        res.append(np.nanmean(V[rows, :], axis=0) - base_v)
        volres.append(np.nanmean(np.abs(V[rows, :]), axis=0) - base_a)
        nsrc += 1
    if not res:
        return None, None, 0
    return np.array(res), np.array(volres), nsrc


say("")
say("=== A/B) 放量传导：某币 RVOL 突增（>99分位）-> 之后 k 分钟 ===")
say("%-6s %10s %12s %12s %12s" % ("滞后", "有效源", "价格lift中位%", "价格P99%", "波动lift中位%"))
for k in KS:
    r, vr, n = scan(RVOL, k)
    if r is None:
        continue
    say("%4dm %10d %11.4f%% %11.4f%% %11.4f%%"
        % (k, n, np.nanmedian(r) * 100, np.nanpercentile(r, 99) * 100, np.nanmedian(vr) * 100))

say("")
say("=== C) 量-价背离：放量但价格几乎没动（涨幅<25分位）-> 之后价格是否跟随 ===")
absV = np.abs(V)
q25 = np.nanquantile(absV, 0.25, axis=0)
quiet_but_volume = np.where(absV < q25, RVOL, np.nan).astype(np.float32)
say("%-6s %10s %12s %12s" % ("滞后", "有效源", "价格lift中位%", "价格P99%"))
for k in KS:
    r, _, n = scan(quiet_but_volume, k)
    if r is None:
        continue
    say("%4dm %10d %11.4f%% %11.4f%%" % (k, n, np.nanmedian(r) * 100, np.nanpercentile(r, 99) * 100))

say("")
say("=== D) 对照：价格剧烈波动（原口径）-> 之后价格 ===")
say("%-6s %12s %12s" % ("滞后", "价格lift中位%", "价格P99%"))
for k in KS:
    r, _, n = scan(absV, k)
    if r is None:
        continue
    say("%4dm %11.4f%% %11.4f%%" % (k, np.nanmedian(r) * 100, np.nanpercentile(r, 99) * 100))

# 最强放量->价格 对
r1, _, n1 = scan(RVOL, 1)
if r1 is not None:
    say("")
    say("=== 放量后 1 分钟：价格反应最强/最弱的币对（top 10）===")
    flat = [(abs(r1[j, i]), j, i, r1[j, i]) for j in range(r1.shape[0]) for i in range(N) if i != j]
    flat.sort(reverse=True)
    for v, j, i, obs in flat[:10]:
        say("  %-13s 放量 -> %-13s %+8.4f%%" % (syms[j], syms[i], obs * 100))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.hist(np.clip(RVOL[np.isfinite(RVOL)], 0, 10), bins=100, color="0.5")
ax.set_title("相对放量 RVOL 分布（截断到10）", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[1]
for k in KS:
    r, _, _ = scan(RVOL, k)
    if r is not None:
        ax.hist(r[np.isfinite(r)] * 100, bins=120, histtype="step", label="放量+%dm" % k)
    r2, _, _ = scan(absV, k)
    if r2 is not None and k == 1:
        ax.hist(r2[np.isfinite(r2)] * 100, bins=120, histtype="step", label="价格波动+%dm" % k)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("放量后的价格 lift 分布 vs 价格波动后", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axes[2]
meds = []
for k in KS:
    r, _, _ = scan(RVOL, k)
    meds.append(np.nanmedian(r) * 100 if r is not None else np.nan)
ax.plot(KS, meds, "o-", color="0.25")
ax.axhline(0, color="0.8", lw=1)
ax.set_xlabel("滞后（分钟）")
ax.set_ylabel("价格 lift 中位 %")
ax.set_title("放量传导随滞后", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "volume_lead.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "volume_lead.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
