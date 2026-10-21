# -*- coding: utf-8 -*-
"""分钟尺度传导（按用户口径）：这一分钟 X 动，之后 1/2/3/5/10 分钟谁跟着动

与之前那版的区别：
  之前：日线「冲击 -> 次日」（错的口径）
  本版：分钟「这一分钟 -> 之后 k 分钟」，且**上涨冲击与下跌冲击分开测**
        （日线已发现下跌共振远强于上涨）
数据：klines_1m_sync.csv（100 币 x 60 天，统一窗口，864 万根）
方法：
  1) 超额收益 = 当分钟收益 - 当分钟全市场均值（剔共同因子）
  2) 冲击：某币某分钟超额收益 > 99 分位（涨）或 < 1 分位（跌）
  3) 响应：之后 k 分钟各币的超额收益
  4) lift = 条件均值 - 无条件均值；对 top 对做置换检验（随机抽同样数量分钟）
输出：evidence/cascade_1m.txt / .png
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
Q_HI, Q_LO = 0.99, 0.01
MIN_EV = 100
NPERM = 300
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=["symbol", "open_time", "close"])
say("载入 %d 行 / %d 币" % (len(df), df["symbol"].nunique()))
px = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()
ret = ret.iloc[1:]
say("宽表: %d 分钟 x %d 币" % ret.shape)
V = ret.sub(ret.mean(axis=1), axis=0).to_numpy(dtype=np.float32)
syms = list(px.columns)
T, N = V.shape
say("超额收益矩阵: %d x %d（内存 %.0f MB）" % (T, N, V.nbytes / 1048576))
say("单币分钟超额收益标准差中位 %.4f%%" % (np.nanstd(V, axis=0).mean() * 100))

base = np.nanmean(V, axis=0)
rng = np.random.default_rng(20261015)


def run_shock(mask_col, k):
    """返回该冲击下、滞后 k 分钟的 lift 行向量（对各响应币）"""
    idx = np.where(mask_col)[0]
    idx = idx[idx < T - k]
    if len(idx) < MIN_EV:
        return None, None, 0
    cond = np.nanmean(V[idx + k, :], axis=0)
    return cond - base, idx, len(idx)


results = {}
for kind, q, cmp_op in [("涨", Q_HI, "hi"), ("跌", Q_LO, "lo")]:
    for k in KS:
        lifts = np.full((N, N), np.nan, dtype=np.float32)
        nev = np.zeros(N, dtype=int)
        for j in range(N):
            col = V[:, j]
            valid = np.isfinite(col)
            if valid.sum() < T * 0.9:
                continue
            thr = np.nanquantile(col, q)
            m = (col > thr) if cmp_op == "hi" else (col < thr)
            m = m & valid
            lift, idx, n = run_shock(m, k)
            if lift is None:
                continue
            lifts[j, :] = lift.astype(np.float32)
            nev[j] = n
        results[(kind, k)] = (lifts, nev)
        ok = nev >= MIN_EV
        if ok.sum() == 0:
            continue
        sub = lifts[ok, :]
        say("")
        say("[%s 冲击 -> %d 分钟后] 有效冲击源 %d 个；lift 中位 %+.4f%%，均值 %+.4f%%，|lift|>0.05%% 的币对 %d/%d"
            % (kind, k, int(ok.sum()), np.nanmedian(sub) * 100, np.nanmean(sub) * 100,
               int(np.nansum(np.abs(sub) > 0.0005)), int(np.isfinite(sub).sum())))

# 汇总对比表
say("")
say("=== 汇总：不同滞后下的 lift 分布（百分点）===")
say("%-6s %-6s %10s %10s %10s %10s" % ("冲击", "滞后", "中位", "均值", "P90", "P99"))
for kind in ["涨", "跌"]:
    for k in KS:
        lifts, nev = results[(kind, k)]
        ok = nev >= MIN_EV
        sub = lifts[ok, :]
        sub = sub[np.isfinite(sub)]
        say("%-6s %5dm %10.4f %10.4f %10.4f %10.4f"
            % (kind, k, np.median(sub) * 100, np.mean(sub) * 100,
               np.percentile(sub, 90) * 100, np.percentile(sub, 99) * 100))

# 置换检验：找最强的对
say("")
say("=== 置换检验（%d 次）：最强的「这一分钟 -> k 分钟后」币对 ===" % NPERM)
say("%-6s %5s %-13s %-13s %10s %9s %8s" % ("冲击", "滞后", "冲击币", "响应币", "lift%", "p值", "冲击数"))
best_rows = []
for kind in ["涨", "跌"]:
    q, cmp_op = (Q_HI, "hi") if kind == "涨" else (Q_LO, "lo")
    for k in KS:
        lifts, nev = results[(kind, k)]
        flat = []
        for j in np.where(nev >= MIN_EV)[0]:
            for ii in range(N):
                if ii == j or not np.isfinite(lifts[j, ii]):
                    continue
                flat.append((abs(lifts[j, ii]), j, ii, lifts[j, ii]))
        flat.sort(reverse=True)
        for _, j, ii, obs in flat[:3]:
            col = V[:, j]
            thr = np.nanquantile(col, q)
            m = (col > thr) if cmp_op == "hi" else (col < thr)
            idx = np.where(m & np.isfinite(col))[0]
            idx = idx[idx < T - k]
            resp = V[:, ii]
            cnt, tot = 0, 0
            for _ in range(NPERM):
                pick = rng.choice(np.arange(T - k), size=len(idx), replace=False)
                v = np.nanmean(resp[pick])
                if not np.isfinite(v):
                    continue
                tot += 1
                if abs(v) >= abs(obs):
                    cnt += 1
            p = (cnt + 1) / (tot + 1.0) if tot else float("nan")
            best_rows.append(dict(kind=kind, k=k, src=syms[j], dst=syms[ii], lift=obs, p=p, n=len(idx)))
            say("%-6s %4dm %-13s %-13s %9.4f %9.4f %8d"
                % (kind, k, syms[j], syms[ii], obs * 100, p, len(idx)))

B = pd.DataFrame(best_rows)
B.to_csv(os.path.join(EV, "cascade_1m_pairs.csv"), index=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
ax = axes[0][0]
for kind in ["涨", "跌"]:
    for k in KS:
        lifts, nev = results[(kind, k)]
        sub = lifts[nev >= MIN_EV, :]
        sub = sub[np.isfinite(sub)]
        ax.hist(sub * 100, bins=120, histtype="step", label="%s冲击 + %dm" % (kind, k))
ax.axvline(0, color="0.2", lw=1)
ax.set_title("lift 分布：涨/跌冲击 x 各滞后", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[0][1]
for kind in ["涨", "跌"]:
    med = [np.median(results[(kind, k)][0][results[(kind, k)][1] >= MIN_EV, :]) * 100 for k in KS]
    p90 = [np.percentile(results[(kind, k)][0][results[(kind, k)][1] >= MIN_EV, :], 90) * 100 for k in KS]
    ax.plot(KS, med, "o-", label="%s 中位" % kind)
    ax.plot(KS, p90, "s--", label="%s P90" % kind)
ax.axhline(0, color="0.8", lw=1)
ax.set_xlabel("滞后（分钟）")
ax.set_ylabel("lift (%)")
ax.set_title("lift 随滞后变化", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1][0]
ax.plot(KS, [0.1056] * len(KS), "k-", label="分钟波动 0.1056%")
ax.plot(KS, [0.2] * len(KS), "k--", label="往返手续费 0.2%")
for kind in ["涨", "跌"]:
    p99 = [np.percentile(results[(kind, k)][0][results[(kind, k)][1] >= MIN_EV, :], 99) * 100 for k in KS]
    ax.plot(KS, p99, "o-", label="%s P99" % kind)
ax.set_xlabel("滞后（分钟）")
ax.set_ylabel("%")
ax.set_title("P99 lift vs 波动/手续费", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1][1]
if len(B):
    ax.scatter(B["k"], B["p"], c="0.4")
    ax.axhline(0.01, color="0.7", ls="--")
ax.set_xlabel("滞后（分钟）")
ax.set_ylabel("p 值")
ax.set_title("最强对的 p 值 vs 滞后", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "cascade_1m.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cascade_1m.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
