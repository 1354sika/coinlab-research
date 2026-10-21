# -*- coding: utf-8 -*-
"""用户三点（都对，我之前有盲区）：
  1) 样本少？有 502 个币 —— 我漏掉了**横截面维度**：样本 = 分钟 x 币，不是只有分钟
  2) 要找的是**大涨幅**，不是平均收益（0.0068% 那种）
  3) "同涨同跌"不是每时每刻 —— 若有些时刻是「分化」的，那时结构就不同 → 可能是可观测的信号

本脚本：
  A) 横截面离散度 disp(t) = 每分钟 100 币收益的标准差：量化「同涨同跌 vs 分化」
     看 disp 的时间结构、以及它与市场收益的关系
  B) 按 disp 分层（低/中/高分化），各自测条件收益 —— 检验「不同步的时刻结构不同」
  C) 大涨幅命中率：以**个币**为单位（样本 = 分钟 x 币 ≈ 860 万），
     统计「给定条件，未来 5 分钟涨/跌 > 1% 的概率」，与无条件基准比
数据：klines_1m_sync.csv（100 币 x 60 天）
输出：evidence/dispersion.txt / .png
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


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "close", "quote_volume"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
Qv = df.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
V = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
QV = Qv.iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
hours = pd.DatetimeIndex(C.index).hour.to_numpy()[1:]
say("矩阵 %d 分钟 x %d 币 = %.2f 亿个「分钟x币」样本" % (T, N, T * N / 1e8))

mkt = np.nanmean(V, axis=1)
disp = np.nanstd(V, axis=1)          # 横截面离散度：0 = 完全同涨同跌
say("")
say("=== A) 横截面离散度 disp(t)：同涨同跌有多彻底 ===")
say("  disp 中位 %.4f%%  均值 %.4f%%  P10 %.4f%%  P90 %.4f%%"
    % (np.median(disp) * 100, disp.mean() * 100, np.percentile(disp, 10) * 100, np.percentile(disp, 90) * 100))
say("  与市场收益的相关: %.3f（%s）"
    % (np.corrcoef(np.abs(mkt), disp)[0, 1], "市场动得越大越分化" if np.corrcoef(np.abs(mkt), disp)[0, 1] > 0 else "反向"))
say("  与市场收益(带符号)的相关: %.3f" % np.corrcoef(mkt, disp)[0, 1])
# 极端分化时刻占比
q90 = np.nanquantile(disp, 0.9)
say("  高度分化时刻（disp > P90 = %.4f%%）占 %.1f%%" % (q90 * 100, 100 * np.mean(disp > q90)))

# ---------- B) 按离散度分层 ----------
say("")
say("=== B) 按离散度分三层：同涨同跌程度不同时，结构是否不同 ===")
q33, q67 = np.nanquantile(disp, [0.33, 0.67])
disp_state = np.where(disp < q33, 0, np.where(disp < q67, 1, 2))
NAMES = ["同步(低disp)", "中等", "分化(高disp)"]
mv5 = pd.Series(mkt).rolling(5).sum().to_numpy()
vol30 = pd.Series(mkt).rolling(30).std().to_numpy()
say("%-14s %8s %12s %12s %12s" % ("离散层", "分钟数", "mkt t+5%", "横截面 std", "P(|r|>1%)"))
for k in range(3):
    idx = np.where((disp_state == k) & np.isfinite(mv5))[0]
    idx = idx[idx < T - 6]
    if len(idx) < 100:
        continue
    f5 = [np.nanmean(V[t + 1:t + 6]) for t in idx[:3000]]
    xs = V[idx[:3000]]
    say("%-14s %8d %11.4f%% %11.4f%% %11.4f%%"
        % (NAMES[k], len(idx), np.nanmean(f5) * 100, np.nanstd(xs) * 100,
           100 * np.nanmean(np.abs(xs) > 0.01)))

# ---------- C) 大涨幅命中率（样本 = 分钟 x 币）----------
say("")
say("=== C) 大涨幅：给定条件，未来 5 分钟个币涨幅 > 阈值 的概率（基准=无条件）===")
H = 5
FWD = np.full((T, N), np.nan, dtype=np.float32)
for h in range(1, H + 1):
    FWD[:T - h] = np.nan_to_num(FWD[:T - h]) + np.nan_to_num(np.roll(V, -h, axis=0)[:T - h])
FWD[T - H:] = np.nan
for thr in (0.005, 0.01, 0.02):
    base_up = float(np.nanmean(FWD > thr))
    base_dn = float(np.nanmean(FWD < -thr))
    say("")
    say("  阈值 |未来5分钟| > %.1f%%：无条件命中率 涨 %.4f%%  跌 %.4f%%（基准）"
        % (thr * 100, base_up * 100, base_dn * 100))
    say("  %-14s %9s %11s %11s %9s" % ("条件", "样本数", "涨>阈值%", "跌>阈值%", "提升倍数"))
    # 条件：离散层 x 市场过去5分钟方向
    for k in range(3):
        for ds, dname in [(1, "市场涨"), (0, "市场跌")]:
            m = (disp_state == k) & (np.sign(np.nan_to_num(mv5)) == (1 if ds else -1))
            idx = np.where(m)[0]
            idx = idx[idx < T - H]
            if len(idx) < 200:
                continue
            sub = FWD[idx]
            up = float(np.nanmean(sub > thr))
            dn = float(np.nanmean(sub < -thr))
            say("  %-14s %9d %10.4f%% %10.4f%% %8.2fx"
                % ("%s/%s" % (NAMES[k], dname), len(idx) * N, up * 100, dn * 100,
                   up / max(base_up, 1e-9)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.plot(np.arange(T), disp * 100, lw=0.4, color="0.5")
ax.axhline(q33 * 100, color="0.3", ls="--", lw=1)
ax.axhline(q67 * 100, color="0.3", ls=":", lw=1)
ax.set_title("横截面离散度 disp(t)（0=完全同涨同跌）", fontsize=10)
ax.set_ylabel("std %")
ax.grid(alpha=0.3)
ax = axes[1]
ax.scatter(np.abs(mkt) * 100, disp * 100, s=2, color="0.5", alpha=0.3)
ax.set_xlabel("|市场收益| %")
ax.set_ylabel("disp %")
ax.set_title("市场波动 vs 横截面分化", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[2]
for thr in (0.005, 0.01, 0.02):
    ups, dns = [], []
    for k in range(3):
        m = (disp_state == k)
        idx = np.where(m)[0]
        idx = idx[idx < T - H]
        sub = FWD[idx]
        ups.append(float(np.nanmean(sub > thr)) * 100)
        dns.append(float(np.nanmean(sub < -thr)) * 100)
    ax.plot(range(3), ups, "o-", label="涨>%.1f%%" % (thr * 100))
    ax.plot(range(3), dns, "s--", label="跌<-%.1f%%" % (thr * 100))
ax.set_xticks(range(3))
ax.set_xticklabels(NAMES, fontsize=8)
ax.set_ylabel("命中率 %")
ax.set_title("不同分化程度下的大涨幅概率", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "dispersion.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "dispersion.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
