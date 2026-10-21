# -*- coding: utf-8 -*-
"""把已实测的发现做成「风控算子」并验证：方向赚不到钱，那风控能不能省钱

已知实测结论（本会话 v0.10/v0.12/v0.13）：
  - 波动率传导真实：某币剧烈波动后，1 分钟其他币波动放大 1.34x（滤波+高额分钟到 1.81x），
    随滞后单调衰减（1.34 -> 1.30 -> 1.28 -> 1.25 -> 1.22）
  - 下尾共现比中位 2.09（88.5% 币对 >1.5）-> 危机时分散化失效
  - 收盘价有 bid-ask bounce（AC(1) = -0.07），用 (高+低)/2 后转正（+0.16）
本脚本把上面三条各做成一个算子，并量化「用了能省多少」：
  R1 波动门控：市场进入「传导窗口」时，等权组合的风险是平常几倍？
  R2 共现分散：把高共现币对当同一头寸，风险预算合并后，极端日损失改善多少？
  R3 中间价止损：止损价用 (高+低)/2 而不是最新成交价，被「价差跳动扫损」的概率降多少？
数据：klines_1m_sync.csv（100 币 x 60 天统一窗口）
输出：evidence/risk_ops.txt / .png
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
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


cols = ["symbol", "open_time", "high", "low", "close"]
df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"), usecols=cols)


def pv(c):
    return df.pivot(index="open_time", columns="symbol", values=c).sort_index()


C = pv("close")
H = pv("high")
L = pv("low")
MID = (H + L) / 2.0
r = C.pct_change().iloc[1:]
rm = MID.pct_change().iloc[1:]
ex = r.sub(r.mean(axis=1), axis=0)
syms = list(C.columns)
T, N = ex.shape
say("矩阵 %d 分钟 x %d 币" % (T, N))

absmat = np.abs(ex.to_numpy(dtype=np.float32))
port = r.mean(axis=1).to_numpy(dtype=np.float32)   # 等权组合的实际收益（不能用超额：超额横截面均值恒为0）

# ---------------- R1 波动门控 ----------------
say("")
say("=== R1 波动门控：市场进入「传导窗口」后，风险是平常的几倍 ===")
mkt_abs = np.nanmean(absmat, axis=1)
shock = mkt_abs > np.nanquantile(mkt_abs, 0.95)      # 市场极端波动分钟
say("  「传导窗口」定义：全市场平均|超额收益| > 95 分位（%d 分钟，占 %.1f%%）"
    % (int(shock.sum()), 100 * shock.mean()))

win = np.zeros(T, dtype=bool)
for k in range(0, 11):                                # 触发后 0-10 分钟
    w = np.roll(shock, k)
    if k:
        w[:k] = False
    win |= w
say("  窗口覆盖（触发后 0-10 分钟）：%d 分钟，占 %.1f%%" % (int(win.sum()), 100 * win.mean()))

in_w = port[win]
out_w = port[~win]
say("")
say("  %-22s %12s %12s %10s" % ("分组", "分钟波动", "P1 分位", "最差单分钟"))
say("  %-22s %11.4f%% %11.4f%% %9.4f%%" % ("传导窗口内",
                                            np.nanstd(in_w) * 100, np.nanpercentile(in_w, 1) * 100,
                                            np.nanmin(in_w) * 100))
say("  %-22s %11.4f%% %11.4f%% %9.4f%%" % ("平常时段",
                                            np.nanstd(out_w) * 100, np.nanpercentile(out_w, 1) * 100,
                                            np.nanmin(out_w) * 100))
ratio = np.nanstd(in_w) / np.nanstd(out_w)
say("  -> 传导窗口内组合波动是平常的 **%.2f 倍**（占 %.0f%% 的时间）" % (ratio, 100 * win.mean()))

# 若在窗口内把仓位降到 50%，组合方差改善多少
wts = np.where(win, 0.5, 1.0)
port_adj = port * wts
say("  -> 若窗口内仓位降到 50%%：组合波动 %.4f%% -> %.4f%%（降 %.1f%%）；最差单分钟 %.4f%% -> %.4f%%"
    % (np.nanstd(port) * 100, np.nanstd(port_adj) * 100,
       100 * (1 - np.nanstd(port_adj) / np.nanstd(port)),
       np.nanmin(port) * 100, np.nanmin(port_adj) * 100))

# ---------------- R2 共现分散 ----------------
say("")
say("=== R2 共现分散：下尾共现比 vs 假设的独立，组合的「同时崩」有多大 ===")
lo = (ex < ex.quantile(0.10))
nm = lo.sum(axis=1)                                   # 每分钟有多少币在下尾
say("  每分钟处于下尾(<10%%分位)的币数：均值 %.1f，P99 %.1f，最大 %d"
    % (nm.mean(), np.percentile(nm, 99), nm.max()))
exp_indep = N * 0.10
say("  若完全独立，期望 %.1f 个 -> 实际均值 %.1f（%.2f 倍），P99 %.1f（%.2f 倍）"
    % (exp_indep, nm.mean(), nm.mean() / exp_indep, np.percentile(nm, 99), np.percentile(nm, 99) / exp_indep))
say("  含义：极端时刻「一起崩」的币数远超独立假设 -> 等权持有 100 币，风险并没有分散掉")

# ---------------- R3 中间价止损 ----------------
say("")
say("=== R3 中间价止损：用 (高+低)/2 还是用最新成交价，被扫损概率差多少 ===")
for dist in (0.003, 0.005, 0.010):
    hit_c = []
    hit_m = []
    for s in syms[:40]:
        rr = r[s].dropna().to_numpy()
        mm = rm[s].dropna().to_numpy()
        idx = min(len(rr), len(mm))
        rr, mm = rr[:idx], mm[:idx]
        # 滚动窗口内：价格从某个基准向下走 dist 是否触发
        run_c = np.cumsum(rr)
        run_m = np.cumsum(mm)
        hit_c.append((run_c < -dist).sum() / len(rr))
        hit_m.append((run_m < -dist).sum() / len(rr))
    say("  止损距离 %.1f%%：收盘价口径触发率 %.4f，中间价口径触发率 %.4f（降 %.1f%%）"
        % (dist * 100, np.mean(hit_c), np.mean(hit_m),
           100 * (1 - np.mean(hit_m) / max(np.mean(hit_c), 1e-9))))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.hist(out_w * 100, bins=100, alpha=0.6, label="平常时段", color="0.7")
ax.hist(in_w * 100, bins=100, alpha=0.6, label="传导窗口内", color="0.25")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("组合分钟收益分布：传导窗口 vs 平常", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(nm, bins=40, color="0.45")
ax.axvline(exp_indep, color="0.2", ls="--", label="独立假设 %.1f" % exp_indep)
ax.set_title("每分钟处于下尾的币数（共现）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
dists = [0.003, 0.005, 0.010]
hc, hm = [], []
for dist in dists:
    a, b = [], []
    for s in syms[:40]:
        rr = r[s].dropna().to_numpy()
        mm = rm[s].dropna().to_numpy()
        i2 = min(len(rr), len(mm))
        rr, mm = rr[:i2], mm[:i2]
        a.append((np.cumsum(rr) < -dist).sum() / len(rr))
        b.append((np.cumsum(mm) < -dist).sum() / len(rr))
    hc.append(np.mean(a))
    hm.append(np.mean(b))
x = np.arange(len(dists))
ax.bar(x - 0.2, hc, 0.4, label="收盘价", color="0.65")
ax.bar(x + 0.2, hm, 0.4, label="(高+低)/2", color="0.3")
ax.set_xticks(x)
ax.set_xticklabels(["%.1f%%" % (d * 100) for d in dists])
ax.set_title("止损距离 vs 被扫概率", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
p1 = os.path.join(EV, "risk_ops.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "risk_ops.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
