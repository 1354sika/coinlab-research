# -*- coding: utf-8 -*-
"""持有期扫描：预测的涨幅能不能「长到」覆盖成本？

用户判断：分钟级预测的涨幅不够覆盖成本。
对应的正确测试是**拉长持有期**：成本是每笔固定（往返 taker 0.2% / maker 0.04%），
若单笔涨幅随持有期增长，成本占比就会下降——这是唯一能让算术翻正的路径。
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口）
设计：
  - 信号固定：过去 60 分钟跌幅最大的 20 币（反转），高波动门控开关对照
  - 持有期 hold ∈ {1, 5, 15, 30, 60, 240, 720, 1440} 分钟
  - 输出：毛收益/笔、成本/笔、净收益/笔、总净收益、交易次数
输出：evidence/holding_scan.txt / .png
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
HOLDS = [1, 5, 15, 30, 60, 240, 720, 1440]
FEE_T = 0.0010   # taker 单边
FEE_M = 0.0002   # maker 单边
SIG_LB = 60
KTOP = 20
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=["symbol", "open_time", "close"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
V = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
say("矩阵 %d 分钟 x %d 币（约 %.1f 天）" % (T, N, T / 1440))
mkt = np.nanmean(V, axis=1)
vol30 = pd.Series(mkt).rolling(30).std().to_numpy()
q67 = np.nanquantile(vol30, 0.67)
gate = vol30 > q67
say("门控：过去30分钟市场波动 > %.4f%%（占 %.0f%% 时间）" % (q67 * 100, 100 * np.nanmean(gate)))


def run(hold, fee, use_gate):
    nav, ntr, gross_sum, cost_sum = 1.0, 0, 0.0, 0.0
    prev = np.zeros(N, dtype=np.float32)
    eq = []
    start = SIG_LB + 2
    for t in range(start, T - hold, hold):
        if use_gate and not (np.isfinite(vol30[t]) and vol30[t] > q67):
            if prev.any():
                cost_sum += np.nansum(np.abs(prev)) * fee
                nav *= (1.0 - np.nansum(np.abs(prev)) * fee)
                prev = np.zeros(N, dtype=np.float32)
            eq.append(nav)
            continue
        past = np.nanmean(V[t - SIG_LB:t], axis=0)
        order = np.argsort(np.nan_to_num(past))
        top = order[:KTOP]
        w = np.zeros(N, dtype=np.float32)
        w[top] = 1.0 / len(top)
        turn = np.nansum(np.abs(w - prev))
        seg = V[t:t + hold]
        gross = np.nansum(w[None, :] * np.nan_to_num(seg).mean(axis=0))
        cost = turn * fee
        nav *= (1.0 + gross - cost)
        gross_sum += gross
        cost_sum += cost
        ntr += 1
        eq.append(nav)
        prev = w
    return nav, ntr, gross_sum / max(ntr, 1), cost_sum / max(ntr, 1), pd.Series(eq)


say("")
say("=== 持有期扫描：反转信号（过去 60 分钟跌最多 20 币），高波动门控 ===")
say("%-10s %8s %12s %12s %12s %12s" % ("持有期", "交易数", "毛收益/笔%", "成本/笔%", "净收益/笔%", "总净收益%"))
res = {}
for hold in HOLDS:
    nav, ntr, g, c, eq = run(hold, FEE_T, True)
    res[("taker", hold)] = (nav - 1) * 100, g, c, eq
    say("%-10s %8d %11.4f%% %11.4f%% %11.4f%% %11.2f%%"
        % ("%d分钟" % hold, ntr, g * 100, c * 100, (g - c) * 100, (nav - 1) * 100))

say("")
say("=== 同样的扫描，费率换成 maker（单边 %.2f%%）===" % (FEE_M * 100))
say("%-10s %8s %12s %12s %12s %12s" % ("持有期", "交易数", "毛收益/笔%", "成本/笔%", "净收益/笔%", "总净收益%"))
for hold in HOLDS:
    nav, ntr, g, c, eq = run(hold, FEE_M, True)
    res[("maker", hold)] = (nav - 1) * 100, g, c, eq
    say("%-10s %8d %11.4f%% %11.4f%% %11.4f%% %11.2f%%"
        % ("%d分钟" % hold, ntr, g * 100, c * 100, (g - c) * 100, (nav - 1) * 100))

say("")
say("=== 对照：不用门控（全程持有）===")
for hold in [60, 240, 1440]:
    for label, fee in [("taker", FEE_T), ("maker", FEE_M)]:
        nav, ntr, g, c, _ = run(hold, fee, False)
        say("  %s / %d分钟: 毛 %.4f%%  成本 %.4f%%  净 %.4f%%  总 %.2f%%"
            % (label, hold, g * 100, c * 100, (g - c) * 100, (nav - 1) * 100))

# 关键判据：毛收益是否随持有期增长、哪一档超过成本
say("")
say("=== 判据：毛收益/笔 是否随持有期增长（若增长，成本占比会下降）===")
for label, fee in [("taker", FEE_T), ("maker", FEE_M)]:
    gs = [res[(label, h)][1] * 100 for h in HOLDS]
    say("  %s: 毛收益/笔 从 %d分钟 %.4f%% 到 %d分钟 %.4f%%" % (label, HOLDS[0], gs[0], HOLDS[-1], gs[-1]))
    say("     （变化倍数 %.2f）" % (gs[-1] / max(abs(gs[0]), 1e-9)))
    cross = [h for h in HOLDS if res[(label, h)][1] > fee]
    say("     毛收益 > 单边成本(%.2f%%) 的持有期档位: %s"
        % (fee * 100, cross if cross else "无"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
for label in ["taker", "maker"]:
    ax.plot(HOLDS, [res[(label, h)][1] * 100 for h in HOLDS], "o-", label="%s 毛收益/笔" % label)
ax.axhline(FEE_T * 100, color="0.3", ls="--", lw=1, label="taker 成本 %.2f%%" % (FEE_T * 100))
ax.axhline(FEE_M * 100, color="0.6", ls=":", lw=1, label="maker 成本 %.2f%%" % (FEE_M * 100))
ax.set_xscale("log")
ax.set_xlabel("持有期（分钟）")
ax.set_ylabel("%")
ax.set_title("毛收益/笔 vs 持有期（虚线=成本线）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for label in ["taker", "maker"]:
    ax.plot(HOLDS, [res[(label, h)][3].iloc[-1] * 100 - 100 for h in HOLDS], "o-", label=label)
ax.axhline(0, color="0.2", lw=1)
ax.set_xscale("log")
ax.set_xlabel("持有期（分钟）")
ax.set_ylabel("总净收益 %")
ax.set_title("总净收益 vs 持有期", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[2]
for label in ["taker", "maker"]:
    eq = res[(label, 1440)][3]
    ax.plot(eq.index, eq.values, lw=1.2, label="%s 1440分钟" % label)
ax.set_title("持有 1 天的净值曲线（门控）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "holding_scan.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "holding_scan.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
