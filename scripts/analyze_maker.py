# -*- coding: utf-8 -*-
"""最终一算：门控 + 反转 + maker 费率，能跨过成本线吗？

已知实测：
  - 高波动状态（过去 30 分钟市场波动 > 67 分位）下，截面传导 lift = 0.0339%，
    低波动仅 0.0060%（5.7 倍）-> 门控有效
  - 横截面「买最强」的毛收益为负（-0.010% ~ -0.021%/笔）-> 应该用**反转**
问题：换成 maker 费率后，高波动门控下的反转策略净收益是否为正？
做法：
  1) 门控：只在「高波动」状态交易，其余时间空仓
  2) 信号：买过去 k 分钟跌幅最大的 20 币（反转），持有 k 分钟
  3) 费率三档：taker 0.10%/边、maker 0.02%/边、maker 返佣 -0.01%/边
  4) 对照：不用门控（全程交易）
输出：evidence/maker_test.txt / .png
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
FEE_TIERS = [("taker 0.10%", 0.0010), ("maker 0.02%", 0.0002), ("maker 返佣 -0.01%", -0.0001)]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "high", "low", "close"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
H = df.pivot(index="open_time", columns="symbol", values="high").sort_index()
L = df.pivot(index="open_time", columns="symbol", values="low").sort_index()
r = C.pct_change().iloc[1:]
r = (r + (H.pct_change() + L.pct_change()) / 2 * 0)  # 保持接口，价格仍用 close
V = r.to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

mkt = np.nanmean(V, axis=1)
vol30 = pd.Series(mkt).rolling(30).std().to_numpy()
q67 = np.nanquantile(vol30, 0.67)
gate = vol30 > q67
say("门控：过去30分钟市场波动 > %.4f%%（高波动状态，占 %.1f%% 的时间）"
    % (q67 * 100, 100 * np.nanmean(gate)))


def backtest(k=5, ktop=20, reverse=True, use_gate=True, fee=0.0010, hold=5):
    nav, ntr, wins = 1.0, 0, 0
    prev = np.zeros(N, dtype=np.float32)
    curve = []
    start = 30 + k + 1
    for t in range(start, T - hold, hold):
        if use_gate and not (gate[t] if np.isfinite(gate[t]) else False):
            w = np.zeros(N, dtype=np.float32)
            turn = np.nansum(np.abs(w - prev))
            nav *= (1.0 - turn * fee)
            curve.append(nav)
            prev = w
            continue
        past = np.nanmean(V[t - k:t], axis=0)
        order = np.argsort(np.nan_to_num(past))
        top = order[:ktop] if reverse else order[-ktop:]
        w = np.zeros(N, dtype=np.float32)
        w[top] = 1.0 / len(top)
        turn = np.nansum(np.abs(w - prev))
        gross = np.nansum(w * np.nan_to_num(V[t:t + hold].mean(axis=0)))
        net = gross - turn * fee
        nav *= (1.0 + net)
        if net > 0:
            wins += 1
        ntr += 1
        curve.append(nav)
        prev = w
    return (nav - 1) * 100, ntr, (wins / max(ntr, 1)) * 100, pd.Series(curve)


say("")
say("=== 反转策略：买过去 k 分钟跌幅最大的 20 币，持有 %d 分钟 ===" % 5)
say("%-20s %8s %10s %10s %10s %10s" % ("费率档 / 门控", "k", "总净收益%", "年化%", "胜率%", "交易次数"))
rows = {}
for label, fee in FEE_TIERS:
    for gname, ug in [("高波动门控", True), ("全程", False)]:
        for k in (1, 3, 5):
            tot, ntr, wr, curve = backtest(k=k, reverse=True, use_gate=ug, fee=fee)
            rows[(label, gname, k)] = curve
            yrs = len(curve) / (365 * 24 * 60 / 5)
            ann = ((1 + tot / 100) ** (1 / max(yrs, 1e-9)) - 1) * 100 if tot > -99 else -100
            say("%-20s %8d %9.2f%% %9.2f%% %9.1f%% %10d"
                % ("%s / %s" % (label, gname), k, tot, ann, wr, ntr))

say("")
say("=== 对照：不做反转（买最强的 20 币）===")
say("%-20s %8s %10s %10s" % ("费率档 / 门控", "k", "总净收益%", "胜率%"))
for label, fee in FEE_TIERS[:2]:
    tot, ntr, wr, _ = backtest(k=5, reverse=False, use_gate=True, fee=fee)
    say("%-20s %8d %9.2f%% %9.1f%%" % ("%s / 高波动门控" % label, 5, tot, wr))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for (label, gname, k), curve in rows.items():
    if k != 5:
        continue
    ax.plot(curve.index, curve.values, lw=1.2, label="%s / %s" % (label, gname))
ax.set_yscale("log")
ax.set_title("反转策略净值（k=5，各费率档）", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axes[1]
labels, vals = [], []
for (label, gname, k), curve in rows.items():
    if k != 5:
        continue
    labels.append("%s\n%s" % (label.split()[0], gname))
    vals.append((curve.iloc[-1] - 1) * 100)
ax.bar(range(len(vals)), vals, color="0.45")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=20, fontsize=7, ha="right")
ax.axhline(0, color="0.2", lw=1)
ax.set_ylabel("总净收益 %")
ax.set_title("费率与门控对净收益的影响（k=5）", fontsize=10)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
p1 = os.path.join(EV, "maker_test.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "maker_test.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
