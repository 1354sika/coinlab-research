# -*- coding: utf-8 -*-
"""全量传导挖掘 v2（修正版）：谁带动谁 —— 502 币 x 9 年日线

v1 的三个 bug（已修）：
  1) t 检验分母用错维度（跨币 std 当成标准误）-> 改成**置换检验**（时间维度重采样）
  2) 币池混入代币化美股/ETF（非 7x24 交易，制造假信号）-> 按「日收益=0 的比例」剔除
  3) 字符串 % 与 * 优先级 bug 导致输出重复上百遍
方法：市场中性化的「冲击-响应」
  - 冲击：币 j 当日的**超额收益**（ret - 当日全市场均值）> 它自己的 97 分位
  - 响应：币 i 在次日的**超额收益**
  - 显著性：置换检验（随机抽同样多的日子，500 次）
输出：evidence/cascade.txt / .png / cascade_pairs.csv
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
Q = 0.97
MIN_EV = 30
HOR = 1
NPERM = 500
TOPTEST = 300
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
say("原始: %d 天 x %d 币  %s .. %s" % (len(ret), ret.shape[1], ret.index[0].date(), ret.index[-1].date()))

# --- 剔除非 7x24 交易资产（代币化美股/ETF：非交易时段价格不动，日收益恰好为 0）---
zs = (ret == 0).sum() / ret.notna().sum()
bad = zs[zs > 0.40].index
ret = ret.drop(columns=bad)
say("剔除「日收益=0 占比>40%%」的资产 %d 个（代币化股票/ETF 类），剩余 %d 币" % (len(bad), ret.shape[1]))
say("  被剔除样例: %s" % ", ".join(sorted(bad)[:18]))

mkt = ret.mean(axis=1)
ex = ret.sub(mkt, axis=0)
say("市场中性化: 冲击与响应均用超额收益 = ret - 当日全市场均值")
resp = ex.shift(-HOR)
base = resp.mean(skipna=True)
ret_q = ex.quantile(Q)
S = ex.gt(ret_q)
idx_all = np.arange(len(ex) - HOR)

cols = list(ret.columns)
lift = pd.DataFrame(index=cols, columns=cols, dtype=float)
nev = {}
for j in cols:
    m = S[j].fillna(False).to_numpy()[:len(ex) - HOR]
    idx = np.where(m)[0]
    nev[j] = len(idx)
    if len(idx) < MIN_EV:
        continue
    lift[j] = resp.iloc[idx].mean(skipna=True) - base
nev = pd.Series(nev)
ok = nev[nev >= MIN_EV].index
L = lift[ok].dropna(how="all")
say("有效冲击源 %d 个币（冲击次数中位 %d）；可统计币对 %d"
    % (len(ok), int(nev[ok].median()), int(L.notna().sum().sum())))

# --- 置换检验：对 |lift| 最大的若干对 ---
flat = L.stack()
flat = flat.reindex(flat.abs().sort_values(ascending=False).index).head(TOPTEST)
rng = np.random.default_rng(20261015)
rows = []
for (dst, src), obs in flat.items():
    idx = np.where(S[src].fillna(False).to_numpy()[:len(ex) - HOR])[0]
    if len(idx) < MIN_EV:
        continue
    col = resp[src].to_numpy()  # 占位，防 KeyError
    col = resp[dst].to_numpy()
    cnt = 0
    k = len(idx)
    for _ in range(NPERM):
        pick = rng.choice(idx_all, size=k, replace=False)
        if abs(np.nanmean(col[pick])) >= abs(obs):
            cnt += 1
    rows.append(dict(src=src, dst=dst, lift=obs, p=(cnt + 1) / (NPERM + 1.0), n=k))
PV = pd.DataFrame(rows).sort_values("p")
PV.to_csv(os.path.join(EV, "cascade_pairs.csv"), index=False)
say("")
say("=== 置换检验结果（对 |lift| 最大的 %d 对做 %d 次重采样）===" % (len(PV), NPERM))
for a in (0.001, 0.01, 0.05):
    say("  p < %.3f 的币对数：%d / %d（随机对照下期望 %d）"
        % (a, int((PV["p"] < a).sum()), len(PV), int(len(PV) * a)))
say("")
say("=== p 最小的 20 对（源 -> 响应）===")
say("%-14s %-14s %10s %9s %7s" % ("源", "响应", "超额%", "p值", "冲击次数"))
for r_ in PV.head(20).itertuples():
    say("%-14s %-14s %9.3f %9.4f %7d" % (r_.src, r_.dst, r_.lift * 100, r_.p, r_.n))

# --- 谁最能带动 / 最容易被带动 ---
src_power = L.mean(axis=0).sort_values(ascending=False)
dst_power = L.mean(axis=1).sort_values(ascending=False)
say("")
say("=== 谁最能带动别人（作为源，次日全场平均超额 %%%%）===")
for c in src_power.head(12).index:
    say("  %-14s %8.3f" % (c, src_power[c] * 100))
say("=== 谁最容易被带动（作为响应方）===")
for c in dst_power.head(12).index:
    say("  %-14s %8.3f" % (c, dst_power[c] * 100))

# --- 非对称性 ---
asy = []
for i, a in enumerate(ok):
    for b in ok[i + 1:]:
        if a not in L.columns or b not in L.index:
            continue
        ab, ba = L.loc[b, a], L.loc[a, b]
        if pd.notna(ab) and pd.notna(ba):
            asy.append(dict(a=a, b=b, a2b=ab, b2a=ba, diff=ab - ba))
A = pd.DataFrame(asy)
A["abs_diff"] = A["diff"].abs()
A.to_csv(os.path.join(EV, "cascade_asym.csv"), index=False)
say("")
say("=== 非对称性 ===")
say("全部 %d 对的中位 |A->B - B->A| = %.4f 百分点"
    % (len(A), float(A["abs_diff"].median() * 100)))
say("最强的 12 对不对称：")
for r_ in A.sort_values("abs_diff", ascending=False).head(12).itertuples():
    say("  %-13s -> %-13s %9.3f vs %9.3f  差 %7.3f" % (r_.a, r_.b, r_.a2b * 100, r_.b2a * 100, r_.diff * 100))

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 2, 1)
tp = src_power.head(20)
ax1.barh([c.replace("USDT", "") for c in tp.index][::-1], (tp.values * 100)[::-1], color="0.35")
ax1.set_title("谁最能带动别人（次日全场平均超额 %）", fontsize=10)
ax1.grid(alpha=0.3, axis="x")

ax2 = fig.add_subplot(2, 2, 2)
top = list(src_power.head(40).index)
sub = L.loc[[c for c in top if c in L.index], top].astype(float) * 100
im = ax2.imshow(sub.values, cmap="gray_r", vmin=np.nanpercentile(sub.values, 2),
                vmax=np.nanpercentile(sub.values, 98))
ax2.set_xticks(range(len(top)))
ax2.set_xticklabels([c.replace("USDT", "") for c in top], rotation=90, fontsize=6)
ax2.set_yticks(range(sub.shape[0]))
ax2.set_yticklabels([c.replace("USDT", "") for c in sub.index], fontsize=6)
ax2.set_title("传导矩阵：源(横) -> 响应(纵)，单位百分点", fontsize=10)
plt.colorbar(im, ax=ax2, fraction=0.04)

ax3 = fig.add_subplot(2, 1, 2)
ax3.hist(PV["lift"] * 100, bins=60, color="0.5")
ax3.axvline(0, color="0.2", lw=1)
ax3.set_title("测试过的 %d 个币对「次日超额收益」分布（百分点）" % len(PV), fontsize=10)
ax3.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "cascade.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cascade.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
