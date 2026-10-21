# -*- coding: utf-8 -*-
"""全量传导 v4（严格版）：只在长期币、共同交易日上做「冲击 -> 次日响应」+ 同期置换检验

为什么要重做（v1/v3 的两个致命偏误）：
  1) 币池混入新上市币：上市初期日涨跌可达 ±100%，与「源币冲击日」在时间上重叠
     -> 产生大量伪相关（v3 出现的 95%、73% 这类 lift 就是它）
  2) 只在 |lift| 最大的 300 对里做检验 = 选择后再检验（selection bias），必然显著
本版规则：
  - 只用上市 >= 2000 天的长期币（共同交易日 >= 800 天）
  - 每一对都在**两者的共同交易日**上计算：源币超额收益 > 97 分位 的次日，响应币超额收益
  - 用**同期**（同一批共同交易日）随机抽样做置换检验，双侧 p 值
  - 报告「通过 p<0.01 的对数」与随机期望的对比（这是判定真假的唯一标准）
输出：evidence/cascade2.txt / .png / cascade2_pairs.csv
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
MIN_BARS = 2000      # 长期币门槛（天）
MIN_COMMON = 800     # 共同交易日门槛
Q = 0.97
MIN_EV = 25
NPERM = 200
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1d_full.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
df = df.drop_duplicates(["symbol", "open_time"])
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()
ret = ret.loc[ret.notna().sum(axis=1) >= 5]

cnt = ret.notna().sum()
big = list(cnt[cnt >= MIN_BARS].index)
say("样本 %d 天；上市 >= %d 天的长期币 %d 个" % (len(ret), MIN_BARS, len(big)))
say("币池: %s ..." % ", ".join(sorted(big)[:12]))

mkt = ret.mean(axis=1)
ex = ret.sub(mkt, axis=0)
V = ex[big].to_numpy(dtype=float)
valid = np.isfinite(V)
T, N = V.shape
say("市场中性化完成（超额收益 = ret - 当日全市场均值）")

rng = np.random.default_rng(20261015)
rows = []
for a in range(N):
    va = valid[:, a]
    for b in range(N):
        if a == b:
            continue
        nxt = np.zeros(T, dtype=bool)
        nxt[:-1] = valid[1:, b]
        common = va & valid[:, b] & nxt
        idx = np.where(common)[0]
        if len(idx) < MIN_COMMON:
            continue
        sa = V[idx, a]
        rb = V[idx + 1, b]
        m = sa > np.quantile(sa, Q)
        k = int(m.sum())
        if k < MIN_EV:
            continue
        obs = rb[m].mean() - rb.mean()
        cnt_ = 0
        for _ in range(NPERM):
            pick = rng.choice(len(rb), size=k, replace=False)
            if abs(rb[pick].mean() - rb.mean()) >= abs(obs):
                cnt_ += 1
        rows.append(dict(src=big[a], dst=big[b], lift=obs, p=(cnt_ + 1) / (NPERM + 1.0),
                         n_ev=k, n_common=len(idx)))

P = pd.DataFrame(rows).sort_values("p")
P.to_csv(os.path.join(EV, "cascade2_pairs.csv"), index=False)
say("")
say("=== 严格版结果：共检验 %d 对（长期币、共同交易日 >= %d）===" % (len(P), MIN_COMMON))
for a in (0.001, 0.01, 0.05):
    exp = len(P) * a
    got = int((P["p"] < a).sum())
    say("  p < %.3f：实际 %d 对，随机期望 %.1f 对  —— %s"
        % (a, got, exp, "超出" if got > exp * 2 else "与随机无异"))

say("")
say("=== p 最小的 20 对 ===")
say("%-13s %-13s %10s %8s %7s %8s" % ("源", "响应", "超额%", "p值", "冲击数", "共同天数"))
for r_ in P.head(20).itertuples():
    say("%-13s %-13s %9.3f %8.4f %7d %8d"
        % (r_.src, r_.dst, r_.lift * 100, r_.p, r_.n_ev, r_.n_common))

# 非对称性（在有共同历史的对上）
L = P.set_index(["dst", "src"])["lift"]
asy = []
for r_ in P.itertuples():
    try:
        back = L.loc[(r_.src, r_.dst)]
    except KeyError:
        continue
    asy.append(abs(r_.lift - back))
say("")
say("=== 非对称性 ===")
say("可配对比较 %d 对；|A->B 减 B->A| 中位 %.4f 百分点，均值 %.4f 百分点"
    % (len(asy), float(np.median(asy) * 100), float(np.mean(asy) * 100)))

say("")
say("=== 谁最能带动（作为源，平均次日超额 %），只在长期币里 ===")
s_pow = P.groupby("src")["lift"].mean().sort_values(ascending=False)
for c in s_pow.head(12).index:
    say("  %-13s %+8.3f" % (c, s_pow[c] * 100))
say("=== 谁最容易被带动（作为响应方）===")
d_pow = P.groupby("dst")["lift"].mean().sort_values(ascending=False)
for c in d_pow.head(12).index:
    say("  %-13s %+8.3f" % (c, d_pow[c] * 100))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.hist(P["lift"] * 100, bins=80, color="0.55")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("全部 %d 对的 lift 分布（百分点）" % len(P), fontsize=10)
ax.grid(alpha=0.3)
ax = axes[1]
ax.hist(P["p"], bins=40, color="0.55")
ax.axhline(len(P) / 40, color="0.2", ls="--", lw=1)
ax.set_title("p 值分布（虚线=均匀分布期望）", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[2]
tp = s_pow.head(20)
ax.barh([c.replace("USDT", "") for c in tp.index][::-1], (tp.values * 100)[::-1], color="0.35")
ax.set_title("谁最能带动（平均次日超额 %）", fontsize=10)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
p1 = os.path.join(EV, "cascade2.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cascade2.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
