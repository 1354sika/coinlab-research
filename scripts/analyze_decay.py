# -*- coding: utf-8 -*-
"""「同时开始」我测过了，「同时结束」没测 —— 这是用户提的新角度

已知：最强相关出现在 lag=0（各币同时开始动）。
未测：一次共同冲击之后，各币的**衰减速度**是否相同？若不同，结束时差就是可套利的东西。
做法：
  1) 定义共同冲击：全市场平均 |超额收益| > 95 分位的分钟
  2) 对每次冲击，记录各币之后 1..30 分钟的「沿冲击方向的累计收益」
  3) 度量持续度 persistence[j]：冲击后 k 分钟收益 / 冲击当时收益（衰减比）
  4) 检验：不同币的衰减速度差异有多大；能否用「衰减慢 vs 衰减快」分组做多空
     （分组用滚动历史估计，无前视），扣费后是否为正
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口）
输出：evidence/decay_asym.txt / .png
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
KS = [1, 2, 3, 5, 10, 30]
Q = 0.95
MIN_EV = 100
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=["symbol", "open_time", "close"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
V = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

mkt = np.nanmean(V, axis=1)
mkt_abs = np.nanmean(np.abs(V), axis=0)
market_abs = np.nanmean(np.abs(V), axis=1)
q95 = np.nanquantile(market_abs, Q)
shock_t = np.where(market_abs > q95)[0]
shock_t = shock_t[shock_t < T - max(KS) - 1]
say("共同冲击（全市场平均|收益| > %.4f%%）: %d 次" % (q95 * 100, len(shock_t)))

# ---- 1) 衰减曲线：冲击方向上的后续收益 / 冲击当时收益 ----
say("")
say("=== 1) 衰减比 = 冲击后 k 分钟收益 / 冲击当时收益（1.0=完全持续，0=立刻结束）===")
say("%-8s %10s %10s %10s %10s" % ("滞后", "中位", "均值", "P10", "P90"))
decay = {}
for k in KS:
    ratios = []
    for t0 in shock_t:
        if t0 + k >= T:
            continue
        now = V[t0]
        fut = np.nanmean(V[t0 + 1:t0 + 1 + k], axis=0) * k   # 之后的累计收益
        sgn = np.sign(now)
        denom = np.abs(now)
        m = (denom > 1e-6) & np.isfinite(fut) & np.isfinite(now)
        if m.sum() < 10:
            continue
        ratios.append((fut[m] * sgn[m]) / denom[m])
    r = np.concatenate(ratios)
    r = r[np.isfinite(r)]
    decay[k] = r
    say("%6dm %10.3f %10.3f %10.3f %10.3f"
        % (k, np.median(r), np.mean(r), np.percentile(r, 10), np.percentile(r, 90)))

# ---- 2) 每币的持续度（谁衰减慢、谁衰减快）----
say("")
say("=== 2) 各币的持续度（冲击后 5 分钟内沿方向的累计收益 / 冲击当时）===")
k0 = 5
persist = {}
for j in range(N):
    vals = []
    for t0 in shock_t:
        if t0 + k0 >= T:
            continue
        now = V[t0, j]
        if not np.isfinite(now) or abs(now) < 1e-6:
            continue
        fut = np.nanmean(V[t0 + 1:t0 + 1 + k0, j]) * k0
        if not np.isfinite(fut):
            continue
        vals.append((fut * np.sign(now)) / abs(now))
    if len(vals) >= MIN_EV:
        persist[syms[j]] = float(np.median(vals))
P = pd.Series(persist).sort_values()
say("  可估币数 %d；持续度 中位 %.3f  最小 %.3f（%s）  最大 %.3f（%s）"
    % (len(P), P.median(), P.min(), P.idxmin(), P.max(), P.idxmax()))
say("  持续度最慢的 10 个币: %s" % ", ".join(P.head(10).index))
say("  持续度最快的 10 个币: %s" % ", ".join(P.tail(10).index))

# ---- 3) 可套利检验：做多「衰减慢」/ 做空「衰减快」，用滚动历史分组（无前视）----
say("")
say("=== 3) 可套利检验：按历史持续度分组，做多慢组 / 做空快组（无前视，扣费）===")
say("%-14s %8s %12s %12s %10s" % ("费率", "交易数", "毛收益/次%", "净收益/次%", "总净收益%"))
for fee_label, fee in [("taker 0.10%", 0.0010), ("maker 0.02%", 0.0002)]:
    nav, ntr, gross_sum = 1.0, 0, 0.0
    prev = np.zeros(N, dtype=np.float32)
    step = 60
    order_hist = None
    start = 2000
    for t in range(start, T - 5, step):
        # 用「过去 step 分钟」重估持续度（滚动、无前视）
        vals = []
        for j in range(N):
            v = []
            for t0 in range(max(0, t - 1440), t - 6, 7):
                now = V[t0, j]
                if not np.isfinite(now) or abs(now) < 1e-6:
                    continue
                fut = np.nanmean(V[t0 + 1:t0 + 6, j]) * 5
                if not np.isfinite(fut):
                    continue
                v.append((fut * np.sign(now)) / abs(now))
            vals.append(np.median(v) if len(v) >= 10 else np.nan)
        vals = np.array(vals, dtype=np.float32)
        ok = np.isfinite(vals)
        if ok.sum() < 20:
            continue
        idx = np.argsort(vals)
        idx = idx[ok[idx]]
        slow = idx[-10:]
        fast = idx[:10]
        w = np.zeros(N, dtype=np.float32)
        w[slow] = 0.5 / len(slow)
        w[fast] = -0.5 / len(fast)
        turn = np.nansum(np.abs(w - prev))
        fwd = np.nanmean(V[t:t + step], axis=0)
        gross = np.nansum(w * np.nan_to_num(fwd))
        nav *= (1.0 + gross - turn * fee)
        gross_sum += gross
        ntr += 1
        prev = w
    say("%-14s %8d %11.4f%% %11.4f%% %10.2f%%"
        % (fee_label, ntr, gross_sum * 100 / max(ntr, 1),
           (gross_sum - 0) * 100 / max(ntr, 1) - 0, (nav - 1) * 100))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.plot(KS, [np.median(decay[k]) for k in KS], "o-", color="0.25")
ax.axhline(0, color="0.8", lw=1)
ax.set_xlabel("冲击后分钟")
ax.set_ylabel("衰减比中位")
ax.set_title("共同冲击后的衰减曲线", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[1]
ax.hist(P.values, bins=40, color="0.5")
ax.set_title("各币持续度分布（n=%d）" % len(P), fontsize=10)
ax.grid(alpha=0.3)
ax = axes[2]
ax.hist(np.clip(decay.get(5, np.array([0.0])), -2, 3), bins=100, color="0.5")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("5 分钟衰减比分布", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "decay_asym.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "decay_asym.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
