# -*- coding: utf-8 -*-
"""第2件：分钟级下沉 —— 1m 数据上的 lead-lag 与「可交易性算术」

数据：data/klines_1m_long.csv（12 币 x ~30 天，44000 根/币）
要回答两件事：
  1) 分钟尺度上到底有没有「领先-滞后」（lag != 0 的相关）
  2) 就算有，扣掉手续费后能不能赚（这是决定性的算术）
输出：evidence/leadlag_1m_long.txt / .png
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
FEE_ROUND = 0.002  # 往返 0.2%（现货 taker 0.1% x2）
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1m_long.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
df = df.drop_duplicates(["symbol", "open_time"])
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index().dropna()
ret = px.pct_change().dropna()
syms = list(ret.columns)
say("分钟数据: %d 分钟 x %d 币  %s .. %s" % (len(ret), len(syms), ret.index[0], ret.index[-1]))

std = ret.std()
say("")
say("=== 关键算术：分钟波动 vs 手续费 ===")
say("单币 1 分钟收益标准差：中位 %.4f%%  最大 %.4f%%" % (std.median() * 100, std.max() * 100))
say("往返手续费 %.2f%% = 分钟波动中位的 %.1f 倍" % (FEE_ROUND * 100, FEE_ROUND / std.median()))
say("-> 要盈利，每笔交易的期望毛收益必须超过 %.2f%%，即 %.1f 个标准差"
    % (FEE_ROUND * 100, FEE_ROUND / std.median()))

# ---- lead-lag：对每一对币，扫 lag -30..30 ----
LAGS = list(range(-30, 31))
say("")
say("=== 逐对 lead-lag（%d 对，lag -30..+30 分钟）===" % (len(syms) * (len(syms) - 1) // 2))
pairs = []
prof = {L: [] for L in LAGS}
for i, a in enumerate(syms):
    for b in syms[i + 1:]:
        cs = {}
        for L in LAGS:
            cs[L] = ret[a].shift(L).corr(ret[b])
        for L in LAGS:
            prof[L].append(abs(cs[L]))
        best = max(cs, key=lambda k: abs(cs[k]))
        pairs.append(dict(a=a, b=b, best_lag=best, best_corr=cs[best], lag0=cs[0]))
pd_ = pd.DataFrame(pairs)
say("同期(lag=0) 相关：中位 %.3f" % pd_["lag0"].median())
say("最强 lag 分布：lag=0 的占比 %.0f%%（%d/%d）"
    % (100 * (pd_["best_lag"] == 0).mean(), int((pd_["best_lag"] == 0).sum()), len(pd_)))
say("|lag|<=1 的占比 %.0f%%；|lag|>=5 的占比 %.0f%%"
    % (100 * (pd_["best_lag"].abs() <= 1).mean(), 100 * (pd_["best_lag"].abs() >= 5).mean()))
pd_.to_csv(os.path.join(EV, "leadlag_1m_pairs.csv"), index=False)

mean_c = {L: float(np.mean(prof[L])) for L in LAGS}
say("")
say("各 lag 的平均|相关|（可见领先/滞后有多弱）：")
for L in (-10, -5, -2, -1, 0, 1, 2, 5, 10):
    say("  lag=%+3d 分钟: %.4f" % (L, mean_c[L]))

# ---- 可交易性算术：用 lag=+1 的相关估上界 ----
c1 = mean_c[1]
edge = c1 * std.median()
say("")
say("=== 可交易性（用最强领先相关做乐观上界）===")
say("lag=+1 平均|相关| = %.4f -> 若据此预测对方下一分钟收益，期望幅度 ≈ %.4f%%"
    % (c1, edge * 100))
say("往返成本 %.2f%% 是它的 %.0f 倍 -> 单笔期望净收益 %.4f%%"
    % (FEE_ROUND * 100, FEE_ROUND / max(edge, 1e-12), (edge - FEE_ROUND) * 100))

# ---- 实测：A 涨 -> 下一分钟买 B ----
say("")
say("=== 实测策略：某分钟涨幅 top3 的币，下一分钟做多其余币中上分钟跌幅最大的 3 个 ===")
nav, ntr = 1.0, 0
curve = []
for t in range(1, len(ret)):
    r_prev = ret.iloc[t - 1]
    r_now = ret.iloc[t]
    leaders = r_prev.sort_values(ascending=False).head(3).index
    laggards = r_prev.drop(leaders).sort_values().head(3).index
    g = float(r_now[laggards].mean())
    nav *= (1.0 + g - FEE_ROUND)
    ntr += 1
    curve.append(nav)
curve = pd.Series(curve, index=ret.index[1:])
say("交易 %d 次，净值 %.4f（毛收益均值 %.4f%%/笔，往返成本 %.2f%%）"
    % (ntr, curve.iloc[-1], (curve.iloc[-1] ** (1 / ntr) - 1 + FEE_ROUND) * 100, FEE_ROUND * 100))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
ax.plot(LAGS, [mean_c[L] for L in LAGS], "o-", ms=3, color="0.25")
ax.axvline(0, color="0.8", lw=1)
ax.set_title("平均|相关| vs 滞后（1m，12 币，%d 对）" % len(pd_), fontsize=10)
ax.set_xlabel("lag (分钟，正=BTC领先)"); ax.set_ylabel("mean |corr|"); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(curve.index, curve.values, lw=1.2, color="0.25")
ax.set_yscale("log")
ax.set_title("实测：跟买滞后币，扣往返 0.2%% -> 净值", fontsize=10)
ax.set_ylabel("NAV (log)"); ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "leadlag_1m_long.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "leadlag_1m_long.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
